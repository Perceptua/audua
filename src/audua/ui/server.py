"""A local, read-only HTTP server over an audua filetree.

Standard library only, on purpose. The pipeline pins torch to CPU wheels to
keep the install small; adding a web framework so a few hundred lines of JSON
could be served would undo that for no gain.

Three decisions worth naming:

* **Read-only, by construction.** Only GET and HEAD are routed at all. There is
  no code path in the UI that writes to the tree, so a mistake here cannot
  damage a recording.
* **Localhost by default.** Clip audio is personal data. Binding to 127.0.0.1
  means the browser can reach it and the network cannot; ``--host`` opens that
  up only when asked.
* **Range requests are real.** Clips run to minutes and browsers seek by asking
  for byte ranges. Answering 206 properly is what makes the scrub bar work
  instead of forcing a whole re-download per drag.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .state import (
    Roots,
    StateError,
    document,
    list_outputs,
    list_sources,
    media_path,
    output_detail,
    overview,
)

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
_CHUNK = 64 * 1024
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".m4b": "audio/mp4",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".webm": "audio/webm",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".wma": "audio/x-ms-wma",
}


def _content_type(path: Path) -> str:
    return _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


class AuduaHandler(BaseHTTPRequestHandler):
    """Routes the handful of GETs the front end makes."""

    # Keep-alive and correct 206s both want HTTP/1.1.
    protocol_version = "HTTP/1.1"
    server_version = "audua-ui"
    sys_version = ""

    roots: Roots = Roots()

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - base class name
        log.debug("%s - %s", self.address_string(), format % args)

    def _write(self, body: bytes) -> None:
        """Write a body, tolerating a browser that hung up mid-stream.

        Aborting a download is how ``<audio>`` seeks: it opens a range, changes
        its mind, and closes the socket. That is normal, not an error.
        """
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            log.debug("client disconnected during response")
            raise _Disconnected from None

    def _head(self, status: int, content_type: str, length: int, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _send(self, status: int, content_type: str, body: bytes, extra: dict | None = None) -> None:
        self._head(status, content_type, len(body), extra)
        if self.command != "HEAD":
            self._write(body)

    def _json(self, payload, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        # The tree is the source of truth and it changes underneath us.
        self._send(status, "application/json; charset=utf-8", body, {"Cache-Control": "no-store"})

    def _error(self, status: int, message: str) -> None:
        if self.path.startswith("/api/"):
            self._json({"error": message, "status": int(status)}, status=status)
        else:
            self._send(status, "text/plain; charset=utf-8", message.encode("utf-8"))

    # ------------------------------------------------------------------
    # files
    # ------------------------------------------------------------------

    def _send_static(self, name: str) -> None:
        try:
            path = (STATIC_DIR / name).resolve()
            if STATIC_DIR.resolve() not in path.parents or not path.is_file():
                raise StateError(name)
            body = path.read_bytes()
        except (StateError, OSError):
            self._error(HTTPStatus.NOT_FOUND, f"no such asset: {name}")
            return
        self._send(HTTPStatus.OK, _content_type(path), body, {"Cache-Control": "no-cache"})

    def _send_media(self, path: Path) -> None:
        """Stream an audio file, honouring a Range request so seeking works."""
        total = path.stat().st_size
        start, end = 0, total - 1
        status = HTTPStatus.OK

        header = self.headers.get("Range")
        if header:
            parsed = self._parse_range(header, total)
            if parsed is None:
                self._head(
                    HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, _content_type(path), 0,
                    {"Content-Range": f"bytes */{total}"},
                )
                return
            start, end = parsed
            status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1
        extra = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
        if status == HTTPStatus.PARTIAL_CONTENT:
            extra["Content-Range"] = f"bytes {start}-{end}/{total}"
        self._head(status, _content_type(path), length, extra)

        if self.command == "HEAD":
            return

        # Chunked so a long clip never has to be held in memory whole.
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                block = handle.read(min(_CHUNK, remaining))
                if not block:
                    break
                self._write(block)
                remaining -= len(block)

    @staticmethod
    def _parse_range(header: str, total: int) -> tuple[int, int] | None:
        """``bytes=start-end`` -> inclusive bounds, or ``None`` if unsatisfiable.

        Only single ranges are supported; a multi-range request falls back to
        the whole file, which is a legal answer and one no audio element asks
        for in practice.
        """
        match = _RANGE_RE.match(header.strip())
        if not match or total == 0:
            return (0, total - 1) if total else None

        first, last = match.group(1), match.group(2)
        if not first and not last:
            return None
        if not first:  # bytes=-N — the final N bytes
            length = min(int(last), total)
            return (total - length, total - 1) if length else None

        start = int(first)
        if start >= total:
            return None
        end = min(int(last), total - 1) if last else total - 1
        return (start, end) if end >= start else None

    # ------------------------------------------------------------------
    # routing
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - base class name
        try:
            self._route()
        except _Disconnected:
            pass
        except StateError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except OSError as exc:
            log.warning("failed serving %s: %s", self.path, exc)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    do_HEAD = do_GET  # noqa: N815 - base class name

    def _route(self) -> None:
        split = urlsplit(self.path)
        path = unquote(split.path)
        query = parse_qs(split.query)
        parts = [p for p in path.split("/") if p]

        if not parts:
            self._send_static("index.html")
            return

        if parts[0] == "static" and len(parts) == 2:
            self._send_static(parts[1])
            return

        if parts[0] == "media" and len(parts) == 3:
            self._send_media(media_path(self.roots, parts[1], parts[2]))
            return

        if parts[0] == "api":
            self._api(parts[1:], query)
            return

        self._error(HTTPStatus.NOT_FOUND, f"no such route: {path}")

    def _api(self, parts: list[str], query: dict[str, list[str]]) -> None:
        match parts:
            case ["health"]:
                self._json({"ok": True, "raw": str(self.roots.raw), "output": str(self.roots.output)})
            case ["overview"]:
                self._json(overview(self.roots))
            case ["sources"]:
                self._json({"sources": list_sources(self.roots)})
            case ["outputs"]:
                self._json({"outputs": list_outputs(self.roots)})
            case ["outputs", name]:
                self._json(output_detail(self.roots, name))
            case ["outputs", name, "doc"]:
                filename = (query.get("file") or [""])[0]
                if not filename:
                    self._error(HTTPStatus.BAD_REQUEST, "doc requires a ?file= parameter")
                    return
                self._json(document(self.roots, name, filename))
            case _:
                self._error(HTTPStatus.NOT_FOUND, f"no such endpoint: /api/{'/'.join(parts)}")


class _Disconnected(Exception):
    """The client closed the connection; stop writing and move on."""


def make_server(
    roots: Roots,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    tls: tuple[Path, Path] | None = None,
) -> ThreadingHTTPServer:
    """Bind a server over ``roots``. Port 0 picks a free one (used by tests).

    ``tls``, if given, is a ``(cert_path, key_path)`` pair — see
    :mod:`audua.ui.tailscale`. The socket is wrapped after binding so a bad
    cert fails at startup, not on the first request.
    """
    handler = type("BoundAuduaHandler", (AuduaHandler,), {"roots": roots})
    server = ThreadingHTTPServer((host, port), handler)
    # Don't hold shutdown hostage to an in-flight audio stream.
    server.daemon_threads = True
    if tls is not None:
        cert_path, key_path = tls
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
        server.socket = context.wrap_socket(server.socket, server_side=True)
    return server


def serve(
    roots: Roots,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    tls: tuple[Path, Path] | None = None,
    shown_host: str | None = None,
) -> int:
    """Run the UI until interrupted.

    ``shown_host`` overrides the hostname printed in the URL — used for
    ``--tailscale``, where the cert is issued for the tailnet MagicDNS name,
    not the bind address itself.
    """
    server = make_server(roots, host, port, tls=tls)
    scheme = "https" if tls is not None else "http"
    shown = shown_host or ("localhost" if host in {"127.0.0.1", "0.0.0.0", "::1"} else host)
    url = f"{scheme}://{shown}:{server.server_address[1]}/"

    print(f"audua UI — {url}")
    print(f"  inbox:   {roots.raw}")
    print(f"  outputs: {roots.output}")
    if host not in {"127.0.0.1", "::1"}:
        print("  note: bound beyond localhost — clip audio is reachable from the network.")
    print("Read-only. Ctrl-C to stop.")

    if open_browser:
        # After the loop is accepting, so the first request cannot race the bind.
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.shutdown()
        server.server_close()
    return 0
