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
    # Brightness shift DISABLED. The naive math `(target - 128) / 128` was
    # broken: it assumes the source starts at neutral (128) and shifts toward
    # the target. Real sources don't start at 128. On a Creators.mp4 source
    # measured at brightness ~51 (already dim), the shift was subtracting
    # another 25-87 points and crushing the image to near-black. Even the
    # ±0.10 cap was still making things WORSE because the math always
    # subtracts when the target is below 128.
    #
    # Proper fix requires source-brightness measurement at render time, then
    # computing delta = (target - source_mean) / 256. That's future work.
    # For now: leave brightness alone. Contrast/saturation/color_temp below
    # still apply for styling without the brightness bug.
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


# Premium tone curve — the "cinematic short-form" look:
#   - shadows lifted from 25% → 30%  (preserves shadow detail, less crushy)
#   - midtones preserved (0.5/0.5)
#   - highlights rolled off softly  (75% → 72%)
#   - max white pulled to 96%       (prevents harsh clipping on bright lamps,
#                                    skin highlights, window blowouts)
# Subtle enough to read as "graded" rather than "stylized." Apply AFTER the
# source-aware brightness shift so the curve shapes the already-lifted image.
PREMIUM_TONE_CURVE = (
    "curves=master='0/0 0.25/0.30 0.5/0.5 0.75/0.72 1/0.96'"
)


def build_premium_curve() -> str:
    """Premium tone curve filter expression. Always-on for now; could be
    gated on a profile flag in the future. Slots in AFTER the source-aware
    brightness shift, BEFORE the captions burn-in."""
    return PREMIUM_TONE_CURVE
