"""ASS subtitle builder. Captions are verbatim from whisper Word objects.

Three caption modes (from variants.json):
  - lower_third  : short phrases in a bottom bar
  - karaoke_word : one word at a time, full-screen
  - subtitles    : traditional 2-line subtitles
"""
from __future__ import annotations

from .config import Word

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
              clip_start: float) -> str:
    """Build an ASS caption file for a clip. Word times are absolute (source-time);
    we subtract clip_start so the .ass timeline starts at 0."""
    res_x, res_y = res
    style = variant["subtitle_style"]
    mode = variant.get("caption_mode", "lower_third")

    header = ASS_HEADER.format(
        res_x=res_x,
        res_y=res_y,
        font=style["font"],
        fontsize=style["font_size"],
        primary=style["primary_color"],
        secondary=style.get("secondary_color", "&H000000FF"),
        outline=style["outline_color"],
        back=style.get("back_color", "&H80000000"),
        bold=1 if style.get("bold", True) else 0,
        border_style=style.get("border_style", 1),
        outline_w=style.get("outline_width", 3),
        shadow=style.get("shadow", 1),
        alignment=style.get("alignment", 2),
        margin_v=style.get("margin_v", 120),
        highlight=style.get("highlight_color", style["primary_color"]),
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
