# Install — Linux

Two prerequisites: **uv** (manages Python and dependencies) and **ffmpeg** (a
system binary, not a Python package).

## 1. uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via your package manager: `pipx install uv`, `brew install uv`,
`sudo pacman -S uv`. Then open a new shell and check:

```bash
uv --version
```

uv installs the right Python itself — `.python-version` pins 3.11 and `uv sync`
will fetch it if the system does not have it. You do not need to install Python
separately or create a venv by hand.

## 2. ffmpeg

```bash
sudo apt update && sudo apt install -y ffmpeg      # Debian / Ubuntu
sudo dnf install -y ffmpeg-free                    # Fedora
sudo pacman -S ffmpeg                              # Arch
```

Verify:

```bash
ffmpeg -version && ffprobe -version
```

If ffmpeg lives somewhere unusual, point at it explicitly rather than editing
PATH:

```bash
export AUDUA_FFMPEG=/opt/ffmpeg/bin/ffmpeg
export AUDUA_FFPROBE=/opt/ffmpeg/bin/ffprobe
```

## 3. Project dependencies

From the project root:

```bash
uv sync
```

This creates `.venv`, installs everything, and writes `uv.lock`. Commit the
lock file — it is what makes the install reproducible.

Torch is pinned to PyTorch's **CPU** index in `pyproject.toml`, which keeps the
download to a few hundred MB instead of several GB. That is the right default
here: torch is only used to run Silero VAD.

### GPU machines

faster-whisper is the part that benefits from a GPU. Pull the CUDA build and
the runtime libraries it needs:

```bash
uv sync --extra-index-url https://download.pytorch.org/whl/cu124
sudo apt install -y libcublas11 libcudnn8
```

Then run with `--device cuda`, or leave `--device auto`, which detects it.

## 4. First run

Models download on first use — Silero is a few MB, `large-v3` is about 3 GB.
They cache in `~/.cache/huggingface` and `~/.cache/torch`.

```bash
uv run audua plan  /path/to/recording.m4a    # preview the cut, fast
uv run audua run   /path/to/recording.m4a
```

`uv run` handles the venv, so there is nothing to activate.

## Troubleshooting

**`Could not find 'ffmpeg' on PATH`** — install it, or set `AUDUA_FFMPEG`.

**`No interpreter found for Python 3.11`** — uv could not download a managed
Python, usually a network or proxy issue. Install 3.11+ from your package
manager and re-run `uv sync`.

**`Silero VAD is not installed`** — run `uv sync`, and invoke the tool with
`uv run audua ...` rather than a bare `python`.

**`Could not load library libcudnn_ops_infer.so.8`** — cuDNN 8 is missing.
Install it, or run with `--device cpu`.

**Transcription is very slow** — you are on CPU with `large-v3`. Try
`--model medium --compute-type int8`, or use a GPU.

**Killed / out of memory during transcription** — lower the model size. The VAD
stage is memory-bounded by design (`--vad-window`, default 600s) and is rarely
the culprit.
