"""Per-variant color grading filters.

Each variant gets a small ffmpeg filter expression that slots into the -vf
chain AFTER the aspect crop / composite and AFTER any sharpening, but BEFORE
the caption burn-in. That ordering matters: it means the underlying footage
takes on the variant's color identity while the captions stay at their brand
color and don't get tinted.

No .cube files / external assets. Pure ffmpeg built-ins so the pipeline
stays self-contained. Swap in a `lut3d=path.cube` here if you want a custom
LUT later.
"""
from __future__ import annotations


GRADE_FILTERS: dict[str, str] = {
    # Slight desaturate + faint cool cast → broadcast-news polish.
    "bloomberg":
        "eq=contrast=1.05:saturation=0.85,"
        "colorchannelmixer=bb=1.05:gg=1.0:rr=0.98",

    # Punchy saturation + warm skin push → social-pop look.
    "tiktok_bro":
        "eq=contrast=1.10:saturation=1.20:gamma_r=0.95:gamma_b=1.05",

    # Flat, slightly lifted blacks → restrained filmic.
    "documentary":
        "eq=contrast=0.97:saturation=0.92:gamma=1.05,"
        "curves=preset=lighter",
}


def filter_for(variant_name: str) -> str | None:
    """Return the ffmpeg filter expression for a variant, or None if unknown."""
    return GRADE_FILTERS.get(variant_name)
