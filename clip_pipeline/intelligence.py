"""Stage 4.75: Variant Intelligence.

Reads framing signals for each selected clip and produces per-(clip × aspect)
render recommendations. The render stage consumes these and adjusts:

  - hero_aspect       — which aspect is the primary deliverable for this clip
  - crop_tightness    — how aggressively to crop toward the speaker (0-1)
  - crf               — quality / bitrate trade
  - sharpen           — unsharp-mask strength for upscale recovery (0-1)
  - preferred_variant — which caption variant fits best (informational, doesn't
                        skip rendering the others)

Decisions are heuristic; based on framing class + speaker_rect dimensions +
motion signals. Pure functions, no I/O, no LLM call. Logged to
intelligence.json for inspection.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .config import Candidate, Framing, Job, cache_read, cache_write


# Default render params for the "easy" content case (full-frame talking head).
DEFAULT = {
    "hero_aspect": "9:16",
    "crop_tightness": 0.5,
    "crf": 20,
    "sharpen": 0.0,
    "use_esrgan": False,
    "esrgan_blend": 0.65,   # 0.65 keeps a natural look on real faces
    "preferred_variant": None,
    "rationale": "",
}


def _recommend_for_clip(framing: Framing, cand: Candidate) -> dict:
    sig = framing.signals if isinstance(framing.signals, dict) else {}
    sr = sig.get("speaker_rect") or {}
    sr_w = float(sr.get("w_frac", 0.0))
    sr_h = float(sr.get("h_frac", 0.0))
    frame_diff = float(sig.get("frame_diff_mean", 0.0))
    static = framing.static
    zoom_on = bool(framing.zoom.get("enabled"))

    rec = dict(DEFAULT)
    notes = []

    if framing.cls == "overlay":
        rec["hero_aspect"] = "9:16" if sr_w < sr_h else "16:9"
        rec["crop_tightness"] = 0.9
        rec["crf"] = 17
        rec["sharpen"] = 0.0   # ESRGAN x4plus + lanczos blend is enough
        rec["use_esrgan"] = True
        rec["esrgan_blend"] = 0.65
        rec["preferred_variant"] = "tiktok_bro"
        notes.append("streamer overlay → ESRGAN x4plus, 65% blend")

    elif framing.cls == "full_frame":
        # Single subject fills the frame. Standard talking-head treatment.
        face_dominant = sr_h >= 0.5
        rec["hero_aspect"] = "9:16"
        rec["crop_tightness"] = 0.75 if face_dominant else 0.6
        rec["crf"] = 18 if (zoom_on or face_dominant) else 20
        rec["sharpen"] = 0.4 if zoom_on else 0.0
        rec["preferred_variant"] = "bloomberg" if not static else "tiktok_bro"
        notes.append("clean talking head" + (" (face dominant)" if face_dominant else ""))

    elif framing.cls == "mid_shot":
        rec["hero_aspect"] = "9:16"
        rec["crop_tightness"] = 0.6
        rec["crf"] = 19
        rec["sharpen"] = 0.3 if zoom_on else 0.0
        rec["preferred_variant"] = "bloomberg"
        notes.append("mid-shot talking head")

    elif framing.cls == "multi_subject":
        rec["hero_aspect"] = "16:9"
        rec["crop_tightness"] = 0.4
        rec["crf"] = 18
        rec["sharpen"] = 0.0
        rec["use_esrgan"] = True
        rec["esrgan_blend"] = 0.55   # heavier blend; interviews need very natural skin
        rec["preferred_variant"] = "documentary"
        notes.append("multi-subject → ESRGAN x4plus, 55% blend (more natural)")

    elif framing.cls == "no_subject":
        # Could be a slide/graphic, gameplay, anything without a face.
        rec["hero_aspect"] = "16:9"
        rec["crop_tightness"] = 0.0    # don't crop in; preserve full source framing
        rec["crf"] = 20
        rec["sharpen"] = 0.0
        rec["preferred_variant"] = "documentary"
        notes.append("no subject detected → preserve original framing")

    # Motion-based adjustments
    if frame_diff > 20.0:
        # Lots of visual motion (gameplay, action) — bump quality.
        rec["crf"] = max(16, rec["crf"] - 1)
        notes.append("high motion → bump quality")

    rec["rationale"] = "; ".join(notes)
    return rec


def recommend(job: Job, selected: list[Candidate],
              framings: list[Framing]) -> dict:
    """Produce intelligence.json — keyed by candidate id."""
    cache = job.work_dir / "intelligence.json"
    if cache.exists():
        return cache_read(cache)

    framing_by_id = {f.id: f for f in framings}
    out: dict[str, dict] = {}
    for c in selected:
        f = framing_by_id[c.id]
        rec = _recommend_for_clip(f, c)
        out[c.id] = rec
        print(
            f"[intelligence] {c.id}: hero={rec['hero_aspect']}, "
            f"crop_tightness={rec['crop_tightness']}, crf={rec['crf']}, "
            f"sharpen={rec['sharpen']}, pref={rec['preferred_variant']} "
            f"({rec['rationale']})",
            file=sys.stderr,
        )
    cache_write(cache, out)
    return out
