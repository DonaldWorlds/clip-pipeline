"""Editorial decisions — replaces the old `intelligence.py`.

Each call produces a per-clip `Decision` dataclass that GROUNDS every render
parameter in either:
  1. A rule from `references/_global/style.md` (loaded via editorial.rules)
  2. A framing signal (face position, motion, source profile)
  3. A retrieved reference clip (Phase 3 — placeholder param `references=None`)

The `Decision.rationale` field is human-readable and cites the rule or signal
behind each choice, so an operator can audit why a clip was rendered the way
it was.

This module is intentionally HEURISTIC + RULE-GROUNDED. No LLM call inside
the function. The LLM-vision pieces live in:
  - editorial/copy.py (post-copy generation, exists)
  - validate/editorial.py (post-render vision QC, Phase 2.4)
  - rag/ingest.py (reference fingerprinting via Pegasus, Phase 3)
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Candidate, Framing, Job, cache_read, cache_write
from . import rules as rules_mod
from .rules import EditorialRules


@dataclass
class Decision:
    """Per-clip render decision. Consumed by render/pool.py (caption + filter
    chain) and validate/editorial.py (rule-compliance check)."""
    clip_id: str

    # Crop
    crop_tightness: float = 0.0           # 0=loose / preserve framing, 1=tight
    use_esrgan: bool = False
    esrgan_blend: float = 0.65

    # Captions — overrides the profile's caption defaults per-clip.
    caption_y_fraction: float = 0.85
    caption_font_px: int = 70
    caption_highlight_color: str = "yellow"

    # Audio / motion
    apply_zoom: bool = False
    apply_filler_trim: bool = True

    # Encode
    crf: int = 20
    sharpen: float = 0.0

    # Trace
    rationale: str = ""

    # Reference IDs that fed into this decision
    reference_ids_used: list[str] = field(default_factory=list)


def _decide_one(
    cand: Candidate,
    framing: Framing,
    rules: EditorialRules,
    references: list | None = None,    # Phase 3: list of ClipFingerprint
    profile=None,                       # Phase 5a: ClientStyleProfile (optional)
) -> Decision:
    """Produce a Decision for ONE clip, grounded in rules + framing (+ refs)."""
    notes: list[str] = []
    sig = framing.signals if isinstance(framing.signals, dict) else {}
    speaker_rect = sig.get("speaker_rect")
    frame_diff = float(sig.get("frame_diff_mean", 0.0))
    static = framing.static
    zoom_on = bool(framing.zoom.get("enabled"))

    d = Decision(clip_id=cand.id)

    # ---- captions (rule-grounded) ----
    d.caption_y_fraction = rules_mod.caption_y_fraction(rules, speaker_rect)
    d.caption_font_px = rules_mod.caption_font_px(rules, rules.output_height)
    d.caption_highlight_color = rules.caption_highlight_color
    notes.append(
        f"caption: rule.position={rules.caption_position}, "
        f"speaker_rect={'present' if speaker_rect else 'absent'} "
        f"→ y={d.caption_y_fraction:.2f}, font={d.caption_font_px}px, "
        f"hl={d.caption_highlight_color}"
    )

    # ---- crop tightness + crf by framing class (style-neutral signals) ----
    if framing.cls == "overlay":
        d.crop_tightness = 0.9
        d.crf = 17
        notes.append("class=overlay → tight crop to speaker_rect")

    elif framing.cls == "full_frame":
        sr_h = float(speaker_rect.get("h_frac", 0.0)) if speaker_rect else 0.0
        face_dominant = sr_h >= 0.5
        d.crop_tightness = 0.75 if face_dominant else 0.6
        d.crf = 18 if (zoom_on or face_dominant) else 20
        d.sharpen = 0.4 if zoom_on else 0.0
        notes.append(f"class=full_frame{' (face dominant)' if face_dominant else ''}")

    elif framing.cls == "mid_shot":
        d.crop_tightness = 0.6
        d.crf = 19
        d.sharpen = 0.3 if zoom_on else 0.0
        notes.append("class=mid_shot")

    elif framing.cls == "multi_subject":
        d.crop_tightness = 0.4
        d.crf = 18
        notes.append("class=multi_subject")

    else:  # no_subject
        d.crop_tightness = 0.0
        d.crf = 20
        notes.append("class=no_subject → preserve original framing")

    # ---- upscaling: PURELY profile-driven creative preference. ----
    # Shape signals (framing.cls) are not quality signals; mixing them is a
    # category error. A future technical-quality validator gates compute
    # justification (bitrate, source blur, face resolution) — separately.
    if profile is not None and profile.use_upscaling:
        d.use_esrgan = True
        # Lower blend when there's no isolated subject region; pure ESRGAN can
        # over-smooth full-frame skin tones, so we blend more bicubic back in
        # for full-frame shots. GFPGAN handles faces regardless.
        if framing.cls in ("overlay", "multi_subject"):
            d.esrgan_blend = 0.65
        else:
            d.esrgan_blend = 0.50
        notes.append(f"profile.use_upscaling=True → ESRGAN blend {d.esrgan_blend}")

    # ---- motion-driven adjustments ----
    if frame_diff > 20.0:
        d.crf = max(16, d.crf - 1)
        notes.append(f"high motion (frame_diff={frame_diff:.1f}) → crf-1 = {d.crf}")

    # ---- rule-grounded zoom + trim ----
    d.apply_zoom = zoom_on and rules.zoom_on_loud_moments
    if d.apply_zoom:
        notes.append("zoom: framing said static AND rule.zoom_on_loud_moments=True → on")

    d.apply_filler_trim = rules.max_silence_s > 0.0
    notes.append(f"filler trim: rule.max_silence_s={rules.max_silence_s}s → "
                 f"{'on' if d.apply_filler_trim else 'off'}")

    # ---- Phase 3 reference blending (placeholder; runs when references provided) ----
    if references:
        # Blend reference medians with rule defaults. Numeric only here;
        # Pegasus descriptions are appended to rationale.
        try:
            from statistics import median
            ref_caption_y = median([
                r.editorial.get("caption_y_frac", d.caption_y_fraction)
                for r in references if hasattr(r, "editorial")
            ])
            ref_font_px = median([
                r.editorial.get("caption_font_px", d.caption_font_px)
                for r in references if hasattr(r, "editorial")
            ])
            # 70% reference, 30% rule
            d.caption_y_fraction = round(0.7 * ref_caption_y + 0.3 * d.caption_y_fraction, 3)
            d.caption_font_px = max(d.caption_font_px, int(ref_font_px))
            d.reference_ids_used = [getattr(r, "id", "?") for r in references]
            notes.append(
                f"refs ({len(references)}): blended caption_y → {d.caption_y_fraction:.2f}, "
                f"font ≥ {d.caption_font_px}px"
            )
        except Exception as e:
            notes.append(f"refs failed to blend: {e}")

    d.rationale = "; ".join(notes)
    return d


def decide_all(
    job: Job,
    selected: list[Candidate],
    framings: list[Framing],
    references_by_clip: dict[str, list] | None = None,
    profile=None,
) -> list[Decision]:
    """Decide per-clip render parameters. Grounded in:
       - profile (ClientStyleProfile): drives caption defaults + use_upscaling
       - rules   (style.md):           the global floor
       - framing:                       per-clip shot signals
       - references_by_clip:            retrieved exemplars (when --client set)
    """
    cache = job.work_dir / "decisions.json"
    if cache.exists():
        raw = cache_read(cache)
        return [Decision(**d) for d in raw]

    rules = rules_mod.load()
    framing_by_id = {f.id: f for f in framings}

    out: list[Decision] = []
    for cand in selected:
        f = framing_by_id.get(cand.id)
        if f is None:
            continue
        refs = (references_by_clip or {}).get(cand.id)
        d = _decide_one(cand, f, rules, references=refs, profile=profile)
        out.append(d)
        print(f"[editorial] {d.clip_id}: "
              f"caption_y={d.caption_y_fraction:.2f}, font={d.caption_font_px}px, "
              f"crf={d.crf}, esrgan={d.use_esrgan}, refs={len(d.reference_ids_used)}",
              file=sys.stderr)
        print(f"[editorial]   rationale: {d.rationale}", file=sys.stderr)

    cache_write(cache, [asdict(d) for d in out])
    return out


def as_dict(decisions: list[Decision]) -> dict[str, dict]:
    """Convert list[Decision] to the dict-keyed-by-id shape that render
    consumes."""
    return {d.clip_id: asdict(d) for d in decisions}
