"""Stage 5: render each selected clip × profile.delivery_aspects to MP4.

One profile per client. One output per (clip × aspect). No variants, no
frame-style matrix, no inset-blur — the client's references decided the
look at ingest time and this stage just delivers it.

Content-aware filter selection:
  - framing.cls in {overlay, multi_subject}: speaker-cropped composite with
    blurred-source background. ESRGAN gates on profile.use_upscaling.
  - everything else: simple aspect crop centered on the speaker (or image
    center fallback), optional Ken Burns zoom for static clips.
"""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..caption import build_ass
from ..config import (
    ASPECT_RES,
    Candidate,
    Framing,
    Job,
    Word,
    aspect_slug,
    cache_write,
)
from ..analyze.whisper import words_in_range
from ..profile import ClientStyleProfile
from .enhance import enhance_speaker_region
from .grade import build_grade_filter, build_premium_curve
from .trim import prepare_trimmed_clip, compute_keep_ranges, retime_words


# libx264 is multithreaded internally; keep parallelism mild to avoid
# oversubscription.
PARALLEL_RENDERS = min(4, max(1, (os.cpu_count() or 4) // 2))

# Social-delivery loudness target. -14 LUFS / -1.5 dBTP is the standard
# TikTok/Reels/Shorts spec; applied to every render so audio levels are
# client-grade automatically.
LOUDNORM_FILTER = "loudnorm=I=-14:LRA=11:TP=-1.5"


def _run(cmd):
    print(f"$ {' '.join(str(c) for c in cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def _escape_ass_filter_path(p: Path) -> str:
    return str(p).replace(":", "\\:").replace("'", "\\'")


def _extract_clip(source: Path, cand: Candidate, out_path: Path) -> None:
    if out_path.exists():
        return
    _run([
        "ffmpeg", "-y",
        "-i", str(source),
        "-ss", f"{cand.start:.3f}",
        "-to", f"{cand.end:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-avoid_negative_ts", "make_zero",
        str(out_path),
    ])


def _probe_source_dims(path: Path) -> tuple[int, int]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        raise ValueError(f"could not probe dimensions for {path}")
    return w, h


def _probe_fps(path: Path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)


def _measure_clip_brightness(clip_path: Path) -> float:
    """Sample 5 frames evenly across the clip and return the median mean-Y
    luminance (0-255 scale). Used by the source-aware brightness adjustment:
    the naive `(target - 128) / 128` math assumes the source starts at 128
    (neutral mid-gray), but real sources rarely do — this measures the
    actual source so the shift toward the target is correct in direction
    and magnitude."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return 128.0
    samples = []
    for pos in (0.10, 0.30, 0.50, 0.70, 0.90):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * pos))
        ok, frame = cap.read()
        if ok and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            samples.append(float(np.mean(gray)))
    cap.release()
    if not samples:
        return 128.0
    return float(np.median(samples))


def _source_aware_brightness_filter(
    clip_path: Path, target: float, cap: float = 0.20,
) -> str | None:
    """Compute the actual brightness shift needed to land near `target`
    given the source's measured mean Y. Caps the shift to ±`cap`.

    Pairs the brightness shift with a saturation BOOST to compensate. Without
    this, an additive brightness shift flattens RGB channel ratios — measured
    on user content: shift +0.18 reduced saturation from 88 to 30 (~66% loss),
    making clips look ashy/washed-out. Saturation boost = 1 + |delta| × 5.0
    restores the vibrancy at 0..0.20 brightness range. ffmpeg eq=saturation
    accepts 0.0..3.0 so we stay well within bounds.
    """
    source_mean = _measure_clip_brightness(clip_path)
    delta = (target - source_mean) / 255.0
    delta = max(-cap, min(cap, delta))
    if abs(delta) < 0.02:
        return None
    sat = 1.0 + abs(delta) * 5.0
    return f"eq=brightness={delta:.3f}:saturation={sat:.2f}"


def _measure_brightness_at_source(
    source: Path, start: float, end: float,
) -> float:
    """Same as _measure_clip_brightness but reads from a [start, end] time
    range directly in the source video. Used by the single-pass render path
    where there is no intermediate extracted clip on disk to read from."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(source))
    samples = []
    for pos in (0.10, 0.30, 0.50, 0.70, 0.90):
        t = start + pos * (end - start)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            samples.append(float(np.mean(gray)))
    cap.release()
    return float(np.median(samples)) if samples else 128.0


def _source_aware_brightness_filter_at_source(
    source: Path, start: float, end: float, target: float, cap: float = 0.20,
) -> str | None:
    """Single-pass variant of _source_aware_brightness_filter. Measures the
    source video directly at [start, end] instead of an extracted clip.
    Same saturation-boost compensation."""
    source_mean = _measure_brightness_at_source(source, start, end)
    delta = (target - source_mean) / 255.0
    delta = max(-cap, min(cap, delta))
    if abs(delta) < 0.02:
        return None
    sat = 1.0 + abs(delta) * 5.0
    return f"eq=brightness={delta:.3f}:saturation={sat:.2f}"


# ---------- filter builders ----------

def _simple_aspect_filter(aspect: str, crop_tightness: float = 0.0,
                          speaker_rect: dict | None = None,
                          src_w: int = 0, src_h: int = 0) -> str:
    """Aspect crop. With a speaker_rect and crop_tightness > 0 the crop is
    shifted/tightened so the speaker stays centered."""
    out_w, out_h = (1080, 1920) if aspect == "9:16" else \
                   ((1080, 1080) if aspect == "1:1" else (1920, 1080))

    if not speaker_rect or crop_tightness <= 0 or src_w == 0:
        if aspect == "9:16":
            return "scale=-2:1920:flags=lanczos,crop=1080:1920"
        if aspect == "1:1":
            return "scale=-2:1080:flags=lanczos,crop=1080:1080"
        if aspect == "16:9":
            return "scale=1920:-2:flags=lanczos,crop=1920:1080"

    out_aspect = out_w / out_h
    src_aspect = src_w / src_h
    cx = speaker_rect["x"] + speaker_rect["w"] / 2
    cy = speaker_rect["y"] + speaker_rect["h"] / 2

    if src_aspect > out_aspect:
        loose_h = src_h
        loose_w = loose_h * out_aspect
    else:
        loose_w = src_w
        loose_h = loose_w / out_aspect

    s_w = speaker_rect["w"]
    s_h = speaker_rect["h"]
    pad = 1.15
    if s_w / s_h > out_aspect:
        tight_w = s_w * pad
        tight_h = tight_w / out_aspect
    else:
        tight_h = s_h * pad
        tight_w = tight_h * out_aspect

    t = max(0.0, min(1.0, crop_tightness))
    win_w = loose_w * (1 - t) + tight_w * t
    win_h = loose_h * (1 - t) + tight_h * t
    win_w = min(win_w, src_w)
    win_h = min(win_h, src_h)

    win_x = max(0, min(src_w - win_w, cx - win_w / 2))
    win_y = max(0, min(src_h - win_h, cy - win_h / 2))

    return (
        f"crop={int(win_w)}:{int(win_h)}:{int(win_x)}:{int(win_y)},"
        f"scale={out_w}:{out_h}:flags=lanczos"
    )


def _composite_overlay_filter(
    speaker_rect: dict,
    aspect: str,
    src_w: int,
    src_h: int,
    enhanced_fg: bool,
) -> str:
    """Speaker-cropped composite for overlay/multi_subject sources (subject
    is small in a larger UI like a stream). Foreground = the speaker_rect
    cropped from source; background = blurred full-source-frame stretched
    to fill output."""
    out_w, out_h = ASPECT_RES[aspect]

    if enhanced_fg:
        fg = (
            f"[1:v]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease"
            f":flags=lanczos[fg]"
        )
    else:
        rx = max(0, int(speaker_rect["x"]))
        ry = max(0, int(speaker_rect["y"]))
        rw = max(2, int(speaker_rect["w"]))
        rh = max(2, int(speaker_rect["h"]))
        rw = min(rw, src_w - rx)
        rh = min(rh, src_h - ry)
        fg = (
            f"[0:v]crop={rw}:{rh}:{rx}:{ry},"
            f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease"
            f":flags=lanczos[fg]"
        )

    bg = (
        f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase"
        f":flags=lanczos,crop={out_w}:{out_h},boxblur=30:1"
        f"[bg]"
    )
    composite = "[bg][fg]overlay=(W-w)/2:(H-h)/2[outv]"
    return ";".join([fg, bg, composite])


def _zoompan_filter(framing: Framing, fps: float, duration: float,
                    src_w: int, src_h: int) -> str:
    """zoompan filter. The s= option requires a literal image size — passing
    `iw`/`ih` expressions there doesn't work (ffmpeg parses it as a literal
    string and fails). We pass the actual source dimensions explicitly so
    the zoompan output preserves the source resolution; downstream crop/scale
    handles the aspect conversion."""
    cx = framing.zoom["center_x"]
    cy = framing.zoom["center_y"]
    z_start = framing.zoom["start"]
    z_end = framing.zoom["end"]
    total_frames = max(1, int(round(duration * fps)))
    z_expr = f"{z_start}+({z_end}-{z_start})*on/{total_frames}"
    x_expr = f"iw*{cx}-(iw/zoom/2)"
    y_expr = f"ih*{cy}-(ih/zoom/2)"
    return (
        f"zoompan=z='{z_expr}'"
        f":x='{x_expr}':y='{y_expr}'"
        f":d=1:fps={fps}:s={src_w}x{src_h}"
    )


def _unsharp_filter(strength: float) -> str | None:
    if strength <= 0:
        return None
    s = max(0.0, min(1.0, strength))
    amount = round(s * 1.2, 2)
    return f"unsharp=5:5:{amount}:5:5:0.0"


# ---------- caption-style synthesis ----------

def _profile_to_caption_dict(profile: ClientStyleProfile) -> dict:
    """build_ass still expects a 'variant'-shaped dict with subtitle_style +
    caption_mode keys. Synthesize one from the profile so caption.py doesn't
    need to learn a new schema."""
    c = profile.caption
    return {
        "caption_mode": c.mode,
        "subtitle_style": {
            "font": c.font,
            "font_size": c.font_px_at_1080,
            "primary_color": c.primary_color,
            "secondary_color": "&H000000FF",
            "outline_color": c.outline_color,
            "back_color": "&H80000000",
            "highlight_color": c.highlight_color,
            "bold": True,
            "border_style": 1,
            "outline_width": c.outline_width,
            "shadow": 1,
            "alignment": 2,   # bottom-center; build_ass remaps from y_frac if a Decision is passed
            "margin_v": int((1.0 - c.y_frac) * 1920),
        },
    }


# ---------- render driver ----------

def _render_one(
    job: Job,
    source: Path,
    cand: Candidate,
    framing: Framing,
    aspect: str,
    profile_caption_dict: dict,
    words_for_clip: list[Word],
    recommendation: dict,
    grade_filter: str | None,
    brightness_target: float,
    keep_ranges_local: list[tuple[float, float]] | None,
) -> Path:
    """Render one (clip × aspect) MP4.

    Two paths based on `keep_ranges_local`:
      - not None  → T2 SINGLE-PASS. Read source with -ss/-to + inline trim
                    via select+aselect filters + full render chain + encode.
                    ONE H.264 encode, eliminates the extract step's quality
                    loss. Used for simple-aspect, no-ESRGAN renders.
      - None      → Two-pass path (composite or ESRGAN). Reads from the
                    intermediate `.work/<id>.mp4` written by
                    _prepare_clip_assets.
    """
    work = job.work_dir
    out_dir = job.delivery_dir

    use_composite = framing.cls in ("overlay", "multi_subject")
    speaker_rect = framing.signals.get("speaker_rect") if isinstance(framing.signals, dict) else None
    if use_composite and not speaker_rect:
        use_composite = False

    # Single-pass is available only when no composite (composite needs the
    # multi-input filter graph + on-disk clip path).
    single_pass = (keep_ranges_local is not None) and (not use_composite)

    if not single_pass:
        # Two-pass needs the intermediate; _prepare_clip_assets already wrote it
        # for ESRGAN/composite cases, but defensively extract here too.
        clip_path = work / f"{cand.id}.mp4"
        _extract_clip(source, cand, clip_path)
    else:
        clip_path = None  # not used in single-pass branch

    res = ASPECT_RES[aspect]
    ass_path = work / f"{cand.id}.{aspect_slug(aspect)}.ass"
    if not ass_path.exists():
        ass_path.write_text(
            build_ass(words_for_clip, profile_caption_dict, res, 0.0,
                      decision=recommendation)
        )

    out_path = out_dir / f"{cand.id}_{aspect_slug(aspect)}.mp4"
    if out_path.exists():
        return out_path

    crf = int(recommendation.get("crf", 20))
    sharpen = float(recommendation.get("sharpen", 0.0))
    crop_tightness = float(recommendation.get("crop_tightness", 0.0))
    want_esrgan = bool(recommendation.get("use_esrgan", False))

    sharpen_f = _unsharp_filter(sharpen)

    # ============================================================
    # T2 SINGLE-PASS RENDER: source → render filters → encode (1 pass).
    # Used when there's no composite and no ESRGAN enhanced file. Reads
    # directly from the long-form source with -ss/-to; the trim is inlined
    # into the filter chain via select+aselect+setpts. ONE H.264 encode
    # instead of two (extract + render). Cleaner generation quality.
    # ============================================================
    if single_pass:
        src_w, src_h = _probe_source_dims(source)
        no_op_trim = (
            len(keep_ranges_local) == 1
            and abs(keep_ranges_local[0][0]) < 1e-3
            and abs(keep_ranges_local[0][1] - (cand.end - cand.start)) < 1e-3
        )

        filters = []
        # Inline trim — drop fillers + long pauses before the rest of the
        # filter chain. Skipped when there's nothing to trim.
        if not no_op_trim:
            expr = "+".join(
                f"between(t,{s:.3f},{e:.3f})" for s, e in keep_ranges_local
            )
            filters.append(f"select='{expr}',setpts=N/FRAME_RATE/TB")

        if framing.zoom.get("enabled"):
            fps = _probe_fps(source)
            # Duration in zoom expression should match the TRIMMED clip
            # length, not the original cand duration, since trim shortens.
            trimmed_dur = sum(e - s for s, e in keep_ranges_local)
            filters.append(_zoompan_filter(framing, fps, trimmed_dur, src_w, src_h))

        filters.append(_simple_aspect_filter(
            aspect, crop_tightness=crop_tightness,
            speaker_rect=speaker_rect, src_w=src_w, src_h=src_h,
        ))
        if sharpen_f:
            filters.append(sharpen_f)
        if grade_filter:
            filters.append(grade_filter)
        bright_f = _source_aware_brightness_filter_at_source(
            source, cand.start, cand.end, brightness_target,
        )
        if bright_f:
            filters.append(bright_f)
        filters.append(build_premium_curve())
        filters.append(f"ass={_escape_ass_filter_path(ass_path)}")
        vf = ",".join(filters)

        af_parts = []
        if not no_op_trim:
            expr = "+".join(
                f"between(t,{s:.3f},{e:.3f})" for s, e in keep_ranges_local
            )
            af_parts.append(f"aselect='{expr}',asetpts=N/SR/TB")
        af_parts.append(LOUDNORM_FILTER)

        _run([
            "ffmpeg", "-y",
            "-i", str(source),
            "-ss", f"{cand.start:.3f}",
            "-to", f"{cand.end:.3f}",
            "-avoid_negative_ts", "make_zero",
            "-vf", vf,
            "-af", ",".join(af_parts),
            "-c:v", "libx264", "-preset", "medium",
            "-crf", str(max(15, crf - 2)),
            "-maxrate", "14M", "-bufsize", "20M",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(out_path),
        ])
        return out_path

    if use_composite:
        src_w, src_h = _probe_source_dims(clip_path)
        enhanced_path = None
        if want_esrgan:
            candidate_path = job.work_dir / f"{cand.id}.enhanced.mp4"
            if candidate_path.exists():
                enhanced_path = candidate_path

        vf = _composite_overlay_filter(
            speaker_rect, aspect, src_w, src_h,
            enhanced_fg=enhanced_path is not None,
        )
        suffix_filters = []
        if sharpen_f:
            suffix_filters.append(sharpen_f)
        if grade_filter:
            suffix_filters.append(grade_filter)
        bright_f = _source_aware_brightness_filter(clip_path, brightness_target)
        if bright_f:
            suffix_filters.append(bright_f)
        suffix_filters.append(build_premium_curve())
        suffix_filters.append(f"ass={_escape_ass_filter_path(ass_path)}")
        vf += ";[outv]" + ",".join(suffix_filters) + "[final]"

        cmd = ["ffmpeg", "-y", "-i", str(clip_path)]
        if enhanced_path:
            cmd += ["-i", str(enhanced_path)]
        cmd += [
            "-filter_complex", vf,
            "-map", "[final]",
            "-map", "0:a",
            "-af", LOUDNORM_FILTER,
            "-c:v", "libx264", "-preset", "medium",
            "-crf", str(max(15, crf - 2)),
            "-maxrate", "14M", "-bufsize", "20M",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(out_path),
        ]
        _run(cmd)
    else:
        # Simple-aspect path. When the profile enabled upscaling AND there
        # is no speaker-rect composite, the enhanced full-frame video
        # becomes the visual input; audio still comes from the original clip
        # because enhance.py strips audio (silent .enhanced.mp4).
        enhanced_path = job.work_dir / f"{cand.id}.enhanced.mp4"
        use_enhanced_source = want_esrgan and enhanced_path.exists()

        visual_input = enhanced_path if use_enhanced_source else clip_path
        src_w, src_h = _probe_source_dims(visual_input)

        filters = []
        if framing.zoom.get("enabled"):
            fps = _probe_fps(clip_path)
            duration = cand.end - cand.start
            filters.append(_zoompan_filter(framing, fps, duration, src_w, src_h))

        filters.append(_simple_aspect_filter(
            aspect, crop_tightness=crop_tightness,
            speaker_rect=speaker_rect, src_w=src_w, src_h=src_h,
        ))
        if sharpen_f:
            filters.append(sharpen_f)
        if grade_filter:
            filters.append(grade_filter)
        # Source-aware brightness shift. Measures the clip's actual luminance
        # and shifts toward profile.color_grade.brightness_target. Capped at
        # ±0.20 so the math can't crush or blow out the image. Applied AFTER
        # the color grade (which is just temp+contrast+saturation) so its
        # contribution is independent and easy to disable.
        bright_f = _source_aware_brightness_filter(clip_path, brightness_target)
        if bright_f:
            filters.append(bright_f)
        # Premium tone curve — applied AFTER brightness lift so it shapes
        # the already-lifted image (lifted shadows + soft highlight rolloff).
        filters.append(build_premium_curve())
        filters.append(f"ass={_escape_ass_filter_path(ass_path)}")
        vf = ",".join(filters)

        if use_enhanced_source:
            # Two-input ffmpeg: [0:v] = enhanced (silent) → -vf chain,
            # [1:a] = original clip's audio track.
            _run([
                "ffmpeg", "-y",
                "-i", str(enhanced_path),
                "-i", str(clip_path),
                "-vf", vf,
                "-map", "0:v",
                "-map", "1:a",
                "-af", LOUDNORM_FILTER,
                "-c:v", "libx264", "-preset", "medium",
                "-crf", str(max(15, crf - 2)),
                "-maxrate", "14M", "-bufsize", "20M",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(out_path),
            ])
        else:
            _run([
                "ffmpeg", "-y",
                "-i", str(clip_path),
                "-vf", vf,
                "-af", LOUDNORM_FILTER,
                "-c:v", "libx264", "-preset", "medium",
                "-crf", str(max(15, crf - 2)),
                "-maxrate", "14M", "-bufsize", "20M",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(out_path),
            ])

    return out_path


def _prepare_clip_assets(
    job: Job,
    source: Path,
    cand: Candidate,
    framing: Framing,
    recommendation: dict,
    words_in_source: list[Word],
    profile: ClientStyleProfile,
) -> tuple[list[Word], list[tuple[float, float]] | None]:
    """Serial prep stage. Returns (retimed_words, keep_ranges_local).

    Two paths:

    (T2 single-pass, common case — no ESRGAN, no composite):
       No intermediate `.mp4` extracted. We just compute the trim metadata
       (retimed words + keep-ranges local to the clip) and let _render_one
       do everything in one ffmpeg pass directly from source. Eliminates the
       extract→trim→render double encode that previously cost ~5% generation
       quality.
       Returns (retimed_words, keep_ranges_local). keep_ranges_local may be
       [(0.0, clip_dur)] if no fillers/pauses were found — _render_one skips
       the select+setpts filter in that case.

    (Two-pass path — ESRGAN OR composite):
       Old behavior: extract → trim → (optionally enhance) on disk. ESRGAN
       runs frame-by-frame in Python and needs a file. Composite reads from
       a clip path. Both still need the intermediate.
       Returns (retimed_words, None) — None signals to _render_one to use
       the on-disk clip path with no inline trim filter.
    """
    use_composite = framing.cls in ("overlay", "multi_subject")
    want_esrgan = profile.use_upscaling and recommendation.get("use_esrgan", False)
    # T2 SINGLE-PASS DISABLED for now. Regression: the select+setpts inline
    # trim in the single-pass filter chain caused the ass=<file> burn-in to
    # silently fail — final MP4s had no captions even though the .ass file
    # was correctly generated. Until I can isolate why ass+select interact
    # badly, fall back to the two-pass extract→render path which is known
    # to preserve captions. The quality lift from single-pass was small
    # (~3% generation gain); captions are non-negotiable. Set always-True
    # so we always take the two-pass branch below.
    need_intermediate = True

    if not need_intermediate:
        # T2 SINGLE-PASS: compute retimed words + keep_ranges WITHOUT writing
        # any intermediate clip files. _render_one reads from source directly.
        keep_source = compute_keep_ranges(
            words_in_source, cand.start, cand.end,
        )
        retimed = retime_words(words_in_source, keep_source)
        keep_local = [(s - cand.start, e - cand.start) for s, e in keep_source]
        # Cache so reruns skip recomputing (small JSONs, no MP4s on disk).
        words_path = job.work_dir / f"{cand.id}.words_trimmed.json"
        report_path = job.work_dir / f"{cand.id}.trim_report.json"
        if not words_path.exists():
            cache_write(words_path, retimed)
        if not report_path.exists():
            no_op = (
                len(keep_source) == 1
                and abs(keep_source[0][0] - cand.start) < 1e-3
                and abs(keep_source[0][1] - cand.end) < 1e-3
            )
            saved = (cand.end - cand.start) - sum(e - s for s, e in keep_source)
            cache_write(report_path, {
                "no_op": no_op,
                "seconds_saved": round(saved, 2),
                "keep_ranges_local": [list(r) for r in keep_local],
                "single_pass": True,
            })
        return retimed, keep_local

    # Two-pass path: extract + trim to disk for ESRGAN or composite
    clip_path = job.work_dir / f"{cand.id}.mp4"
    _extract_clip(source, cand, clip_path)

    clip_path, words_trimmed = prepare_trimmed_clip(
        clip_path, words_in_source, cand, job.work_dir,
    )

    if not want_esrgan:
        return words_trimmed, None  # composite path, no enhance

    sig = framing.signals if isinstance(framing.signals, dict) else {}
    speaker_rect = sig.get("speaker_rect")
    if not speaker_rect:
        src_w, src_h = _probe_source_dims(clip_path)
        speaker_rect = {"x": 0, "y": 0, "w": src_w, "h": src_h}

    enhance_speaker_region(
        clip_path, speaker_rect, cand.id, job.work_dir,
        blend=float(recommendation.get("esrgan_blend", 0.65)),
    )
    return words_trimmed, None


def render_all(
    job: Job,
    source: Path,
    words: list[Word],
    selected: list[Candidate],
    framings: list[Framing],
    profile: ClientStyleProfile,
    decisions_by_id: dict,
) -> list[Path]:
    job.delivery_dir.mkdir(parents=True, exist_ok=True)
    job.work_dir.mkdir(parents=True, exist_ok=True)

    framing_by_id = {f.id: f for f in framings}
    caption_dict = _profile_to_caption_dict(profile)
    grade_filter = build_grade_filter(profile.color_grade)
    brightness_target = float(profile.color_grade.brightness_target)

    # Stage A — serial prep (trim metadata + optional ESRGAN). ESRGAN must
    # be serial because GFPGAN/ESRGAN can't share an MPS device across worker
    # processes. For non-ESRGAN clips, Stage A now only computes trim metadata
    # without writing intermediate files (T2 single-pass).
    per_clip_words: dict[str, list[Word]] = {}
    per_clip_keep_ranges: dict[str, list[tuple[float, float]] | None] = {}
    for cand in selected:
        framing = framing_by_id[cand.id]
        if framing.dropped:
            continue
        rec = decisions_by_id.get(cand.id, {})
        words_for_clip = words_in_range(words, cand.start, cand.end)
        words_trimmed, keep_ranges = _prepare_clip_assets(
            job, source, cand, framing, rec, words_for_clip, profile,
        )
        per_clip_words[cand.id] = words_trimmed
        per_clip_keep_ranges[cand.id] = keep_ranges

    # Stage B — build the parallel job list: ONE (clip × aspect) per render.
    jobs_list: list[tuple] = []
    for cand in selected:
        framing = framing_by_id[cand.id]
        if framing.dropped:
            print(f"[render] {cand.id}: source frame is static — dropping clip",
                  file=sys.stderr)
            continue

        rec = decisions_by_id.get(cand.id, {})
        words_for_clip = per_clip_words.get(cand.id, [])
        keep_ranges = per_clip_keep_ranges.get(cand.id)

        for aspect in profile.delivery_aspects:
            jobs_list.append((
                job, source, cand, framing, aspect,
                caption_dict, words_for_clip, rec, grade_filter,
                brightness_target, keep_ranges,
            ))

    print(f"[render] running {len(jobs_list)} renders, "
          f"{PARALLEL_RENDERS}-way parallel "
          f"(client={profile.client}, aspects={profile.delivery_aspects})",
          file=sys.stderr)

    produced: list[Path] = []
    with ProcessPoolExecutor(max_workers=PARALLEL_RENDERS) as ex:
        futures = {ex.submit(_render_one, *args): args for args in jobs_list}
        for fut in as_completed(futures):
            args = futures[fut]
            cand_id, aspect = args[2].id, args[4]
            try:
                p = fut.result()
                produced.append(p)
                print(f"[render]   -> {p}", file=sys.stderr)
            except Exception as e:
                print(f"[render] FAILED {cand_id} × {aspect}: {e}", file=sys.stderr)
                raise

    return produced
