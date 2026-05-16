"""Color grading derived from the ClientStyleProfile.

The pipeline does not have a color identity of its own. The client's reference
clips define brightness, saturation, contrast, and temperature targets; this
module turns those numeric targets into an ffmpeg filter expression that slots
into the -vf chain AFTER the aspect crop and AFTER sharpening, BEFORE the
caption burn-in.
"""
from __future__ import annotations

from ..profile import ColorGrade


def build_grade_filter(grade: ColorGrade) -> str | None:
    """Translate ColorGrade targets into an ffmpeg filter chain.

    - brightness_target (0-255 mean luminance) → eq=brightness shift
    - saturation_target (1.0 = neutral)         → eq=saturation
    - contrast_target (1.0 = neutral)           → eq=contrast
    - color_temp (warm/neutral/cool)            → colorchannelmixer tint

    Returns None when the profile is purely neutral (no filter needed)."""
    parts: list[str] = []

    # eq= covers contrast + saturation in one expression.
    eq_terms: list[str] = []
    if abs(grade.contrast_target - 1.0) > 0.01:
        eq_terms.append(f"contrast={grade.contrast_target:.2f}")
    if abs(grade.saturation_target - 1.0) > 0.01:
        eq_terms.append(f"saturation={grade.saturation_target:.2f}")
    # Brightness shift: ffmpeg eq=brightness is -1.0..+1.0, neutral 0.
    # The naive math `(target - 128) / 128` assumes the source starts at
    # neutral (128). Real sources rarely do — applying a -0.34 shift to a
    # normally-lit YouTube video crushes it. Cap the shift to ±0.10 so the
    # color grade can nudge toward the reference look without crushing the
    # source. For deeper target-matching we'd need to measure the source
    # brightness first (future work).
    raw_delta = (grade.brightness_target - 128.0) / 128.0
    delta = max(-0.10, min(0.10, raw_delta))
    if abs(delta) > 0.02:
        eq_terms.append(f"brightness={delta:.2f}")
    if eq_terms:
        parts.append("eq=" + ":".join(eq_terms))

    # Color temperature: subtle channel-mixer push. Warm = +R/-B, cool = -R/+B.
    if grade.color_temp == "warm":
        parts.append("colorchannelmixer=rr=1.04:bb=0.97")
    elif grade.color_temp == "cool":
        parts.append("colorchannelmixer=rr=0.97:bb=1.04")

    if not parts:
        return None
    return ",".join(parts)
