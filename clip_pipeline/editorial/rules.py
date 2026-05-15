"""Editorial rules — parsed from references/_global/style.md.

This module is the bridge between the human-editable markdown rulebook and the
typed decisions module. The pipeline reads the rules at startup; any time you
edit style.md the next run picks up the new values automatically.

Parsing is intentionally LOOSE — keyword + regex matching with sensible
defaults. Missing rules don't crash, they just use the default.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_STYLE_PATH = Path(__file__).resolve().parents[2] / "references" / "_global" / "style.md"


@dataclass
class EditorialRules:
    """The editorial spec, after parsing style.md. Single source of truth for
    every render decision."""
    # Captions
    caption_position: str = "lower_third"          # lower_third | center | upper_third
    caption_max_lines: int = 2
    caption_min_font_px: int = 70
    caption_highlight_color: str = "yellow"
    safe_margins_pct: float = 5.0                  # inset from frame edges

    # Pacing
    max_silence_s: float = 1.0
    movement_cadence_s: float = 2.0
    hook_first_s: float = 0.5
    zoom_on_loud_moments: bool = True

    # Clip length
    clip_min_s: float = 5.0
    clip_max_s: float = 30.0

    # Encoder
    output_width: int = 1080
    output_height: int = 1920
    pixel_format: str = "yuv420p"
    codec: str = "h264"
    fps: int = 60

    # Raw markdown sections kept around for inspection / Pegasus prompts
    raw_sections: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ----- parsing helpers -----

_INT_RE = re.compile(r"(\d+)")
_FLOAT_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _find_int(text: str, *patterns: str) -> int | None:
    """Return the first integer matched by any of the given patterns.
    Each pattern MUST contain a single (\\d+) capture group."""
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _find_float(text: str, *patterns: str) -> float | None:
    """Each pattern MUST contain a single (\\d+(?:\\.\\d+)?) capture group."""
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def _has(text: str, *needles: str) -> bool:
    return any(n.lower() in text.lower() for n in needles)


def _split_sections(md: str) -> dict[str, str]:
    """Split a markdown file into sections keyed by `##` headers."""
    sections: dict[str, str] = {"_preamble": ""}
    current = "_preamble"
    for line in md.splitlines():
        h = re.match(r"^##\s+(.+?)\s*$", line)
        if h:
            current = h.group(1).strip().lower()
            sections[current] = ""
            continue
        sections[current] = sections.get(current, "") + line + "\n"
    return sections


# ----- parser -----

def parse(md: str) -> EditorialRules:
    """Parse style.md text into EditorialRules. Defaults survive missing rules."""
    rules = EditorialRules()
    sections = _split_sections(md)
    rules.raw_sections = sections

    captions = sections.get("captions", "")
    pacing = sections.get("pacing", "")
    clip_length = sections.get("clip length", "")
    encoder = sections.get("encoder profile", "")

    # Captions — vertical position (horizontal is always centered for vertical
    # short-form). Check in priority order so a "lower third" hint wins over a
    # generic "centered" mention which usually refers to horizontal.
    if captions:
        if _has(captions, "lower third", "lower-third"):
            rules.caption_position = "lower_third"
        elif _has(captions, "upper third", "upper-third"):
            rules.caption_position = "upper_third"
        elif _has(captions, "vertically centered", "middle of frame"):
            rules.caption_position = "center"
        # Otherwise: default "lower_third" already set on the dataclass.

        ml = _find_int(captions, r"max\s+(\d+)\s+lines?", r"(\d+)\s+lines?\s+max")
        if ml is not None:
            rules.caption_max_lines = ml

        fs = _find_int(captions, r"minimum font size\s+(\d+)", r"font size\s+(\d+)")
        if fs is not None:
            rules.caption_min_font_px = fs

        for color in ("yellow", "green", "white", "red", "blue"):
            if f"highlight keywords {color}" in captions.lower() or f"keywords {color}" in captions.lower():
                rules.caption_highlight_color = color
                break

        margin = _find_float(captions, r"(\d+(?:\.\d+)?)\s*%\s*inset", r"safe margins?\s*\((\d+(?:\.\d+)?)\s*%")
        if margin is not None:
            rules.safe_margins_pct = margin

    # Pacing
    if pacing:
        s = _find_float(pacing, r"no silence longer than\s+(\d+(?:\.\d+)?)", r"silence\s*>\s*(\d+(?:\.\d+)?)")
        if s is not None:
            rules.max_silence_s = s
        c = _find_float(pacing, r"every\s+(\d+(?:\.\d+)?)\s*seconds?", r"every\s+(\d+(?:\.\d+)?)\s*s")
        if c is not None:
            rules.movement_cadence_s = c
        h = _find_float(pacing, r"first\s+(\d+(?:\.\d+)?)\s*seconds?", r"first\s+(\d+(?:\.\d+)?)\s*s")
        if h is not None:
            rules.hook_first_s = h
        rules.zoom_on_loud_moments = _has(pacing, "zoom on loud", "zoom loud")

    # Clip length
    if clip_length:
        cmin = _find_float(clip_length, r"minimum\s+(\d+(?:\.\d+)?)\s*seconds?", r"min\s+(\d+(?:\.\d+)?)\s*s")
        if cmin is not None:
            rules.clip_min_s = cmin
        cmax = _find_float(clip_length, r"maximum\s+(\d+(?:\.\d+)?)\s*seconds?", r"max\s+(\d+(?:\.\d+)?)\s*s")
        if cmax is not None:
            rules.clip_max_s = cmax

    # Encoder
    if encoder:
        dims = re.search(r"(\d+)\s*[x×]\s*(\d+)", encoder)
        if dims:
            rules.output_width = int(dims.group(1))
            rules.output_height = int(dims.group(2))
        if "yuv420p" in encoder.lower():
            rules.pixel_format = "yuv420p"
        if "h264" in encoder.lower():
            rules.codec = "h264"
        fps = _find_int(encoder, r"fps\s+(\d+)\b", r"\b(\d+)\s*fps\b")
        if fps is not None:
            rules.fps = fps

    return rules


def load(path: Path | None = None) -> EditorialRules:
    """Load and parse the rulebook. Falls back to defaults if file missing."""
    p = path or DEFAULT_STYLE_PATH
    if not p.is_file():
        return EditorialRules()
    return parse(p.read_text())


# ----- decision helpers used by editorial/decisions.py -----

def caption_y_fraction(rules: EditorialRules, speaker_rect: dict | None) -> float:
    """Compute a safe caption y-position (0.0 = top, 1.0 = bottom) that:
    - Honors rules.caption_position direction
    - Stays inside rules.safe_margins_pct from edges
    - Does NOT overlap the speaker's face if speaker_rect provided
    """
    margin = rules.safe_margins_pct / 100.0
    top_safe = margin
    bottom_safe = 1.0 - margin

    if speaker_rect:
        face_top = float(speaker_rect.get("y_frac", 0.4))
        face_bottom = face_top + float(speaker_rect.get("h_frac", 0.3))
    else:
        face_top, face_bottom = 0.35, 0.65

    if rules.caption_position == "upper_third":
        # Above the face if possible
        candidate = max(top_safe + 0.02, face_top - 0.10)
        return min(candidate, top_safe + 0.20)
    elif rules.caption_position == "center":
        # Center vertically — but if it would overlap the face, push below
        if 0.45 < (face_top + face_bottom) / 2 < 0.55:
            return min(face_bottom + 0.08, bottom_safe - 0.05)
        return 0.5
    else:  # lower_third (default)
        # Below the face if possible
        candidate = max(face_bottom + 0.05, 0.75)
        return min(candidate, bottom_safe - 0.05)


def caption_font_px(rules: EditorialRules, output_height: int | None = None) -> int:
    """Final caption font size: max of rule minimum and 5% of output height."""
    h = output_height or rules.output_height
    return max(rules.caption_min_font_px, int(h * 0.05))
