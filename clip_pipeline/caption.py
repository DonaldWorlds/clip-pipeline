"""ASS subtitle builder. Captions are verbatim from whisper Word objects.

Three caption modes (from ClientStyleProfile.caption.mode, synthesized by
render.pool._profile_to_caption_dict into a 'variant'-shaped dict):
  - lower_third  : short phrases in a bottom bar
  - karaoke_word : one word at a time, full-screen
  - subtitles    : traditional 2-line subtitles

Caption position, font size, and highlight color can be overridden per-clip
via a Decision dict. The profile-synthesized dict controls font FAMILY, mode,
border, and animation; the Decision controls POSITION/SIZE/COLOR.
"""
from __future__ import annotations

from .config import Word


# Highlight color name → ASS BGR hex.
_HIGHLIGHT_TO_BGR = {
    "yellow": "&H0000FFFF",
    "green":  "&H0000FF00",
    "white":  "&H00FFFFFF",
    "red":    "&H000000FF",
    "blue":   "&H00FF0000",
}


def _y_fraction_to_alignment(y_frac: float) -> tuple[int, int, int]:
    """Convert y_fraction (0=top..1=bottom) to ASS (alignment, margin_v_for_bottom,
    margin_v_for_top). Returns the alignment + the margin to use given the
    output height when render fills it in.

    The caller still needs the output height to compute the final margin_v.
    For alignment=2 (bottom-anchored) margin_v = (1-y_frac)*h
    For alignment=5 (middle-anchored) margin_v = 0 (vertical center)
    For alignment=8 (top-anchored)    margin_v = y_frac*h
    """
    if y_frac <= 0.33:
        return 8, 0, 0  # top-center
    elif y_frac >= 0.67:
        return 2, 1, 0  # bottom-center, use bottom-margin formula
    else:
        return 5, 0, 0  # middle-center, no margin

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Base,{font},{fontsize},{primary},{secondary},{outline},{back},{bold},0,0,0,100,100,0,0,{border_style},{outline_w},{shadow},{alignment},60,60,{margin_v},1
Style: Highlight,{font},{fontsize},{highlight},{secondary},{outline},{back},{bold},0,0,0,100,100,0,0,{border_style},{outline_w},{shadow},{alignment},60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def fmt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:05.2f}"


def ass_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("\n", "\\N")
    )


def group_phrases(words: list[Word], max_words: int, max_dur: float) -> list[list[Word]]:
    phrases: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if not cur:
            cur = [w]
            continue
        if len(cur) >= max_words or (w.end - cur[0].start) > max_dur:
            phrases.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        phrases.append(cur)
    return phrases


def build_ass(words: list[Word], variant: dict, res: tuple[int, int],
              clip_start: float, decision: dict | None = None) -> str:
    """Build an ASS caption file for a clip. Word times are absolute (source-time);
    we subtract clip_start so the .ass timeline starts at 0.

    If `decision` is provided (dict with keys caption_y_fraction,
    caption_font_px, caption_highlight_color), it OVERRIDES the variant's
    position/size/color while keeping the variant's font family, mode, etc.
    """
    res_x, res_y = res
    style = variant["subtitle_style"]
    mode = variant.get("caption_mode", "lower_third")

    # Decision-driven overrides (Phase 2)
    if decision:
        font_px = int(decision.get("caption_font_px", style["font_size"]))
        y_frac = float(decision.get("caption_y_fraction", 0.85))
        align, _, _ = _y_fraction_to_alignment(y_frac)
        if align == 2:
            margin_v = int((1.0 - y_frac) * res_y)
        elif align == 8:
            margin_v = int(y_frac * res_y)
        else:
            margin_v = 0
        hl_name = decision.get("caption_highlight_color", "yellow")
        highlight = _HIGHLIGHT_TO_BGR.get(hl_name, style.get("highlight_color", style["primary_color"]))
    else:
        font_px = style["font_size"]
        align = style.get("alignment", 2)
        margin_v = style.get("margin_v", 120)
        highlight = style.get("highlight_color", style["primary_color"])

    header = ASS_HEADER.format(
        res_x=res_x,
        res_y=res_y,
        font=style["font"],
        fontsize=font_px,
        primary=style["primary_color"],
        secondary=style.get("secondary_color", "&H000000FF"),
        outline=style["outline_color"],
        back=style.get("back_color", "&H80000000"),
        bold=1 if style.get("bold", True) else 0,
        border_style=style.get("border_style", 1),
        outline_w=style.get("outline_width", 3),
        shadow=style.get("shadow", 1),
        alignment=align,
        margin_v=margin_v,
        highlight=highlight,
    )

    # Shift word times to clip-local coordinates and clip to [0, clip_dur].
    local_words = [
        Word(text=w.text, start=max(0.0, w.start - clip_start),
             end=max(0.0, w.end - clip_start))
        for w in words
    ]

    lines: list[str] = []
    if mode == "karaoke_word":
        for w in local_words:
            lines.append(
                f"Dialogue: 0,{fmt_time(w.start)},{fmt_time(w.end)},"
                f"Highlight,,0,0,0,,{ass_escape(w.text)}"
            )
    elif mode == "lower_third":
        for phrase in group_phrases(local_words, max_words=7, max_dur=3.0):
            full = " ".join(w.text for w in phrase)
            lines.append(
                f"Dialogue: 0,{fmt_time(phrase[0].start)},"
                f"{fmt_time(phrase[-1].end)},Base,,0,0,0,,{ass_escape(full)}"
            )
    else:  # "subtitles"
        for phrase in group_phrases(local_words, max_words=12, max_dur=4.5):
            full = " ".join(w.text for w in phrase)
            lines.append(
                f"Dialogue: 0,{fmt_time(phrase[0].start)},"
                f"{fmt_time(phrase[-1].end)},Base,,0,0,0,,{ass_escape(full)}"
            )

    return header + "\n".join(lines) + "\n"
