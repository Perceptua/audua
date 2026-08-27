# Install — Windows

Two prerequisites: **uv** (manages Python and dependencies) and **ffmpeg** (a
system binary, not a Python package).

## 1. uv

```powershell
winget install --id=astral-sh.uv -e
```

Or, in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a **new** terminal (PATH changes do not apply to already-open ones) and
check:

```powershell
uv --version
```

uv installs the right Python itself — `.python-version` pins 3.11 and `uv sync`
will fetch it if the system does not have it. You do not need to install Python
from python.org, create a venv, or deal with PowerShell execution policy for
activation scripts.

## 2. ffmpeg

```powershell
winget install Gyan.FFmpeg
```

Or with Chocolatey (`choco install ffmpeg-full`), or by downloading a build
from <https://www.gyan.dev/ffmpeg/builds/>, unzipping it, and adding its `bin`
folder to PATH.

Open a new terminal and verify:

```powershell
ffmpeg -version
ffprobe -version
```

If you would rather not touch PATH, point at the binaries directly:

```powershell
$env:AUDUA_FFMPEG  = "C:\tools\ffmpeg\bin\ffmpeg.exe"
$env:AUDUA_FFPROBE = "C:\tools\ffmpeg\bin\ffprobe.exe"
```

To make that permanent, use `setx` or System Properties → Environment
Variables.

## 3. Project dependencies

From the project root:

```powershell
uv sync
```

This creates `.venv`, installs everything, and writes `uv.lock`. Commit the
lock file — it is what makes the install reproducible.

Torch is pinned to PyTorch's **CPU** index in `pyproject.toml`, which keeps the
download to a few hundred MB instead of several GB. That is the right default
here: torch is only used to run Silero VAD.

### GPU machines

faster-whisper is the part that benefits from a GPU:

```powershell
uv sync --extra-index-url https://download.pytorch.org/whl/cu124
```

The CUDA torch wheel ships the cuBLAS and cuDNN DLLs faster-whisper needs,
inside `.venv\Lib\site-packages\torch\lib`. If CUDA is still not found, add
that folder to PATH or fall back to `--device cpu`.

## 4. First run

Models download on first use — Silero is a few MB, `large-v3` is about 3 GB.
They cache under `%USERPROFILE%\.cache`.

```powershell
uv run audua plan  "D:\recordings\session one.m4a"
uv run audua run   "D:\recordings\session one.m4a"
```

Quote any path containing spaces. `uv run` handles the venv, so there is
nothing to activate.

## Windows-specific notes

**Long paths.** Windows caps paths at 260 characters unless long-path support
is enabled. A deeply nested output root plus long source filenames can hit
this. Either keep the output root shallow (`-o D:\audua-out`) or enable long
paths:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**Filenames.** Output folders are named after the source file stem. Windows
forbids `< > : " / \ | ? *` in filenames, so a stem containing them (possible
on a file copied from Linux) will fail — rename the source first.

**Antivirus.** Real-time scanning inspects every clip as it is written, which
can dominate runtime on a recording that produces hundreds of clips. Consider
excluding the output folder.

**Console encoding.** If transcripts print as garbage in the terminal, that is
a console codepage issue, not a data one — the `.txt` files are always UTF-8.
`chcp 65001` fixes the display.

## Troubleshooting

**`Could not find 'ffmpeg' on PATH`** — install it and open a new terminal, or
set `AUDUA_FFMPEG`.

**`uv` is not recognized** — open a new terminal after installing; PATH changes
do not reach already-open ones.

**`Silero VAD is not installed`** — run `uv sync`, and invoke the tool with
`uv run audua ...` rather than a bare `python`.

**Transcription is very slow** — CPU with `large-v3`. Try
`--model medium --compute-type int8`.

**A run stopped partway** — just run the same command again. Finished clips are
reused and it resumes where it left off.
