"""Job loading, paths, shared dataclasses, env handling."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VARIANTS_PATH = PROJECT_ROOT / "variants.json"

ASPECT_RES = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}


def aspect_slug(aspect: str) -> str:
    """'9:16' -> '9x16' for use in filenames."""
    return aspect.replace(":", "x")


# Source profile classes — drive whether the pipeline treats the source as a
# clean phone-creator recording (whole frame is the subject) vs. a screen
# capture (the subject is a small overlay in a larger UI).
SOURCE_PROFILES = ("phone_vertical", "landscape", "square", "unknown")


def detect_source_profile(source_path: "Path") -> str:
    """Read the source's frame dimensions and classify its shape.

    Returns one of:
      phone_vertical  — aspect ratio ~9:16 (phone recording, TikTok/Reels native)
      landscape       — aspect ratio ~16:9 (camera, screen capture, or YouTube)
      square          — aspect ratio ~1:1 (Instagram feed source)
      unknown         — everything else

    Used by framing.analyze() to short-circuit overlay-class composite logic
    for phone-creator content where the whole frame IS the deliverable.
    """
    import cv2
    cap = cv2.VideoCapture(str(source_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        return "unknown"
    ratio = w / h
    if 0.50 <= ratio <= 0.65:    # 9:16 = 0.5625
        return "phone_vertical"
    if 1.70 <= ratio <= 1.85:    # 16:9 = 1.778
        return "landscape"
    if 0.90 <= ratio <= 1.10:    # 1:1
        return "square"
    return "unknown"


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Candidate:
    """One detected moment. Output of Stage 3 (either backend)."""
    id: str
    start: float
    end: float
    score: float
    rationale: str = ""
    topic: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        return cls(
            id=str(d["id"]),
            start=float(d["start"]),
            end=float(d["end"]),
            score=float(d["score"]),
            rationale=str(d.get("rationale", "")),
            topic=str(d.get("topic", "")),
        )


@dataclass
class Framing:
    """Output of Stage 4.5 — per-clip framing + motion decision."""
    id: str
    cls: str               # multi_subject | talking_head | full_body | mid_shot | no_subject
    static: bool
    zoom: dict             # {enabled, start, end, center_x, center_y}
    aspects: list          # subset of ["9:16", "1:1", "16:9"]
    signals: dict
    dropped: bool = False  # frozen-frame escape hatch fired


@dataclass
class Job:
    source: str
    source_type: str                 # local | youtube
    output_dir: Path
    top_n: int = 3
    detection_backend: str = "auto"  # auto | twelvelabs | local
    llm_model: str = "llama3.1:8b"
    whisper_model: str = "small"
    variants: list = field(default_factory=lambda: ["bloomberg", "tiktok_bro", "documentary"])

    # Hash is computed once on first access then cached. Two reasons:
    # 1. Rehashing a multi-GB source file every time is wasteful.
    # 2. If the user moves or renames the source mid-job (it happens), we
    #    don't want late stages to crash. The hash from when we started is
    #    what matches the cached intermediates anyway.
    _source_hash_cache: str | None = field(default=None, repr=False, compare=False)

    @property
    def work_dir(self) -> Path:
        return self.output_dir / ".work"

    @property
    def source_hash(self) -> str:
        """SHA-256 of the source — file content for local, URL for youtube.
        Cached after first call."""
        if self._source_hash_cache is not None:
            return self._source_hash_cache
        h = hashlib.sha256()
        if self.source_type == "local":
            p = Path(self.source).expanduser().resolve()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        else:
            h.update(self.source.encode("utf-8"))
        self._source_hash_cache = h.hexdigest()[:16]
        return self._source_hash_cache


def load_env() -> None:
    """Load .env at project root if present. Idempotent."""
    load_dotenv(PROJECT_ROOT / ".env")


def resolve_detection_backend(requested: str) -> str:
    """Apply auto-detection logic to job/CLI value.
    auto -> twelvelabs if env key present, else local.
    """
    if requested == "auto":
        return "twelvelabs" if os.environ.get("TWELVELABS_API_KEY") else "local"
    if requested not in ("twelvelabs", "local"):
        sys.exit(f"unknown detection_backend: {requested!r}")
    if requested == "twelvelabs" and not os.environ.get("TWELVELABS_API_KEY"):
        sys.exit(
            "detection_backend=twelvelabs but TWELVELABS_API_KEY is not set. "
            "Add it to .env or fall back with --detection-backend local."
        )
    return requested


def load_job(path: Path) -> Job:
    raw = json.loads(path.read_text())
    return Job(
        source=raw.get("source", ""),
        source_type=raw.get("source_type", "local"),
        output_dir=Path(raw.get("output_dir", "out")).expanduser(),
        top_n=int(raw.get("top_n", 3)),
        detection_backend=raw.get("detection_backend", "auto"),
        llm_model=raw.get("llm_model", "llama3.1:8b"),
        whisper_model=raw.get("whisper_model", "small"),
        variants=raw.get("variants") or ["bloomberg", "tiktok_bro", "documentary"],
    )


def load_variants(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------- JSON cache helpers ----------

def cache_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    elif isinstance(obj, list) and obj and hasattr(obj[0], "__dataclass_fields__"):
        obj = [asdict(o) for o in obj]
    path.write_text(json.dumps(obj, indent=2))


def cache_read(path: Path):
    return json.loads(path.read_text())
