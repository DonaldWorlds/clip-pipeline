"""Stage 5.6: filler-word and long-pause trim.

Reads whisper word-level timestamps for each selected clip, identifies stretches
worth cutting (standalone fillers, pauses > threshold), and uses ffmpeg's
select/aselect filters to produce a trimmed clip with synchronized audio.

The trim is conservative on purpose — it only removes obvious dead weight:
  - exact "um", "uh", "ah", "er", "umm", "uhh" (standalone words)
  - pauses > 0.6s between consecutive words (kept 0.2s buffer on each side)
It does NOT remove "like", "actually", "literally", or "you know" since those
are sometimes legitimate speech. Tune via FILLER_WORDS below.

Word timestamps are re-aligned to the post-trim timeline so captions stay in
sync. Cached: re-runs skip the ffmpeg pass.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from .config import Candidate, Word, cache_read, cache_write


FILLER_WORDS = {"um", "uh", "ah", "er", "umm", "uhh", "hmm", "uhm"}
MIN_PAUSE_TO_CUT = 0.6     # seconds
PAUSE_KEEP_BUFFER = 0.2    # seconds preserved on each side of a cut


def _norm(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def compute_keep_ranges(
    words: list[Word], clip_start: float, clip_end: float,
    filler_words: set[str] = FILLER_WORDS,
    min_pause: float = MIN_PAUSE_TO_CUT,
) -> list[tuple[float, float]]:
    """Return keep ranges in SOURCE-TIME (absolute) that the trimmed clip will
    be made of. The clip span [clip_start, clip_end] is the universe; this
    function punches holes for fillers and long pauses, then returns the
    complement."""
    if not words:
        return [(clip_start, clip_end)]

    cuts: list[tuple[float, float]] = []

    # Filler words → cut their entire span with a tiny buffer.
    for w in words:
        if _norm(w.text) in filler_words:
            cuts.append((
                max(clip_start, w.start - 0.05),
                min(clip_end, w.end + 0.05),
            ))

    # Long pauses → cut the inner part of the silence, keeping a buffer.
    for prev, curr in zip(words, words[1:]):
        pause = curr.start - prev.end
        if pause <= min_pause:
            continue
        cut_start = prev.end + PAUSE_KEEP_BUFFER
        cut_end = curr.start - PAUSE_KEEP_BUFFER
        if cut_end > cut_start:
            cuts.append((cut_start, cut_end))

    # Merge overlapping cuts.
    cuts.sort()
    merged: list[tuple[float, float]] = []
    for cs, ce in cuts:
        if merged and cs <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], ce))
        else:
            merged.append((cs, ce))

    # Keep ranges = complement of cuts inside [clip_start, clip_end].
    keep: list[tuple[float, float]] = []
    cursor = clip_start
    for cs, ce in merged:
        if cursor < cs:
            keep.append((cursor, cs))
        cursor = max(cursor, ce)
    if cursor < clip_end:
        keep.append((cursor, clip_end))

    return keep


def retime_words(
    words: list[Word], keep_ranges_source: list[tuple[float, float]],
) -> list[Word]:
    """Return word list with timestamps re-mapped to the trimmed timeline
    (zero-based). Words whose midpoint falls outside any keep range are
    dropped (i.e. they were inside a filler/pause cut)."""
    retimed: list[Word] = []
    elapsed = 0.0
    for r_start, r_end in keep_ranges_source:
        r_dur = r_end - r_start
        for w in words:
            mid = (w.start + w.end) / 2
            if r_start <= mid <= r_end:
                new_s = max(0.0, w.start - r_start) + elapsed
                new_e = min(r_dur, w.end - r_start) + elapsed
                if new_e > new_s:
                    retimed.append(Word(text=w.text, start=new_s, end=new_e))
        elapsed += r_dur
    return retimed


def _trim_with_ffmpeg(
    clip_path: Path, keep_ranges_local: list[tuple[float, float]],
    out_path: Path,
) -> None:
    """ffmpeg select+aselect to keep only the specified clip-local ranges.
    setpts/asetpts re-zero the timestamps so the output is contiguous."""
    # Build the select expression: between(t,X1,X2)+between(t,X3,X4)+...
    expr = "+".join(
        f"between(t,{s:.3f},{e:.3f})" for s, e in keep_ranges_local
    )
    vf = f"select='{expr}',setpts=N/FRAME_RATE/TB"
    af = f"aselect='{expr}',asetpts=N/SR/TB"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(clip_path),
            "-vf", vf,
            "-af", af,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ],
        check=True,
        stderr=subprocess.DEVNULL,
    )


def prepare_trimmed_clip(
    clip_path: Path,
    words_source_time: list[Word],
    cand: Candidate,
    work_dir: Path,
) -> tuple[Path, list[Word]]:
    """Trim fillers + long pauses in-place on clip_path. Returns the (possibly
    same) clip path + word list in TRIMMED-TIMELINE coordinates (zero-based).

    If nothing to trim, the clip is left as-is; we still return zero-based
    words so the caller can always pass clip_start=0 to caption builders.

    Cached via <id>.words_trimmed.json + <id>.trim_report.json:
    - If both exist, skip the ffmpeg pass entirely.
    - <id>.mp4 is the trimmed clip (replaces the original after first run).
    """
    words_path = work_dir / f"{cand.id}.words_trimmed.json"
    report_path = work_dir / f"{cand.id}.trim_report.json"

    if words_path.exists() and report_path.exists():
        return clip_path, [Word(**d) for d in cache_read(words_path)]

    keep_source = compute_keep_ranges(
        words_source_time, cand.start, cand.end,
    )
    retimed = retime_words(words_source_time, keep_source)

    # If we have a single keep range covering the entire clip, no trim needed.
    no_op = (
        len(keep_source) == 1
        and abs(keep_source[0][0] - cand.start) < 1e-3
        and abs(keep_source[0][1] - cand.end) < 1e-3
    )

    saved_secs = (cand.end - cand.start) - sum(e - s for s, e in keep_source)
    n_fillers = sum(
        1 for w in words_source_time if _norm(w.text) in FILLER_WORDS
    )
    n_pauses = sum(
        1 for prev, curr in zip(words_source_time, words_source_time[1:])
        if (curr.start - prev.end) > MIN_PAUSE_TO_CUT
    )
    report = {
        "no_op": no_op,
        "n_fillers_cut": n_fillers if not no_op else 0,
        "n_pauses_cut": n_pauses if not no_op else 0,
        "seconds_saved": round(saved_secs, 2),
        "keep_ranges_source": [list(r) for r in keep_source],
    }
    cache_write(report_path, report)

    if no_op:
        print(f"[trim] {cand.id}: no fillers / no long pauses — skipping trim",
              file=sys.stderr)
        cache_write(words_path, retimed)
        return clip_path, retimed

    # Trim — replace clip_path with the trimmed version.
    keep_local = [(s - cand.start, e - cand.start) for s, e in keep_source]
    temp_path = clip_path.with_suffix(".trimmed.mp4")
    print(f"[trim] {cand.id}: cutting {n_fillers} filler(s) + {n_pauses} "
          f"pause(s), saving {saved_secs:.1f}s", file=sys.stderr)
    try:
        _trim_with_ffmpeg(clip_path, keep_local, temp_path)
    except subprocess.CalledProcessError as e:
        print(f"[trim] {cand.id}: ffmpeg trim failed ({e}) — keeping original",
              file=sys.stderr)
        if temp_path.exists():
            temp_path.unlink()
        # Fall back to no-op: keep clip as-is but use the would-be retimed words.
        cache_write(words_path, retimed)
        return clip_path, retimed

    # Replace original with trimmed.
    clip_path.unlink()
    temp_path.rename(clip_path)
    cache_write(words_path, retimed)
    return clip_path, retimed
