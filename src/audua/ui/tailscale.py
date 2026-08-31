"""Opt-in Tailscale binding for the UI server.

``audua ui --tailscale`` is a second, explicit widening beyond ``--host``: it
binds the machine's Tailscale IP and serves HTTPS with a cert ``tailscale
cert`` issues for the node's MagicDNS name (Let's Encrypt-backed, trusted by
browsers with no manual setup) — the same mechanism nomothetic uses for the
nomon fleet API.

No fallback to a self-signed cert on failure. Clip audio is personal data;
if Tailscale is not available, the right answer is a clear error, not a
quietly weaker cert.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


class TailscaleError(Exception):
    """Tailscale is unavailable, disconnected, or `tailscale cert` failed."""


def tailscale_ip() -> str:
    """The IPv4 address of this node on the tailnet."""
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5
        )
    except FileNotFoundError as exc:
        raise TailscaleError("tailscale CLI not found — install it or drop --tailscale.") from exc
    except subprocess.TimeoutExpired as exc:
        raise TailscaleError("tailscale ip timed out.") from exc
    ip = result.stdout.strip()
    if result.returncode != 0 or not ip:
        raise TailscaleError(
            f"tailscale ip -4 failed (exit {result.returncode}) — is tailscale up? "
            f"stderr: {result.stderr.strip()}"
        )
    return ip


def provision_cert(cert_dir: Path) -> tuple[Path, Path, str]:
    """Issue a Tailscale cert for this node's MagicDNS name.

    Returns ``(cert_path, key_path, fqdn)``. Re-issues on every call —
    ``tailscale cert`` is cheap and idempotent, and the UI is a short-lived
    dev process, so there is no cache-and-check-expiry step to get wrong.
    """
    try:
        status = subprocess.run(
            ["tailscale", "status", "--json"], capture_output=True, text=True, timeout=5
        )
    except FileNotFoundError as exc:
        raise TailscaleError("tailscale CLI not found — install it or drop --tailscale.") from exc
    except subprocess.TimeoutExpired as exc:
        raise TailscaleError("tailscale status timed out.") from exc
    if status.returncode != 0:
        raise TailscaleError(
            f"tailscale status failed (exit {status.returncode}): {status.stderr.strip()}"
        )

    fqdn = json.loads(status.stdout).get("Self", {}).get("DNSName", "").rstrip(".")
    if not fqdn:
        raise TailscaleError(
            "tailscale status returned no DNSName — enable MagicDNS in the "
            "Tailscale admin console."
        )

    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = cert_dir / "cert.pem", cert_dir / "key.pem"
    cert = subprocess.run(
        ["tailscale", "cert", "--cert-file", str(cert_path), "--key-file", str(key_path), fqdn],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if cert.returncode != 0:
        combined = "\n".join(filter(None, [cert.stdout.strip(), cert.stderr.strip()]))
        raise TailscaleError(
            f"tailscale cert failed (exit {cert.returncode}) for {fqdn}:\n  {combined}\n"
            "  Hint: may need elevated permissions — try running as root or add your "
            "user to the 'tailscale' group."
        )
    return cert_path, key_path, fqdn
