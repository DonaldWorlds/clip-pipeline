"""Source acquisition (local file or YouTube URL via yt-dlp)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .config import Job


def _run(cmd):
    print(f"$ {' '.join(str(c) for c in cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def ensure_source(job: Job) -> Path:
    """Return a Path to a local copy of the source. Downloads via yt-dlp when needed.
    Cached: subsequent calls with the same job return the same path without re-downloading.
    """
    if job.source_type == "local":
        p = Path(job.source).expanduser()
        if not p.is_file():
            sys.exit(f"source file not found: {p}")
        return p
    if job.source_type == "youtube":
        job.work_dir.mkdir(parents=True, exist_ok=True)
        out = job.work_dir / "source.mp4"
        if out.exists():
            return out
        _run([
            "yt-dlp",
            "-f", "bestvideo[height<=1080]+bestaudio/best",
            "--merge-output-format", "mp4",
            "-o", str(out),
            job.source,
        ])
        return out
    sys.exit(f"unknown source_type: {job.source_type!r}")
