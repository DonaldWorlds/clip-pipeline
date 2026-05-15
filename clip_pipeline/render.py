"""Stage 5: render each selected clip × variant × aspect to MP4.

Content-aware:
  - full_frame / mid_shot / no_subject: simple aspect crop centered on the
    speaker (or image center fallback). Optional Ken Burns zoom for static
    clips.
  - overlay (small webcam + screen-capture): blur-background composite.
    Foreground is the speaker_rect cropped from source and scaled to fit the
    output; background is a heavily-blurred full-source-frame stretched to fill
    the output. Result: subject is sharp and centered; corners are filled with
    soft blurred content rather than upscale pixelation.
  - multi_subject: speaker_rect spans the union of clusters; treated like
    overlay rendering (composite) so both subjects stay in frame.

All three aspects (9:16, 1:1, 16:9) are rendered for every clip.
"""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .caption import build_ass
from .config import (
    ASPECT_RES,
    Candidate,
    Framing,
    Job,
    Word,
    aspect_slug,
)
from .enhance import enhance_speaker_region
from .grade import filter_for as grade_filter_for
from .trim import prepare_trimmed_clip
from .whisper import words_in_range


import os

# Max ffmpeg renders to run in parallel. libx264 already uses multiple threads
# internally, so this is set to keep oversubscription mild on the host CPU.
PARALLEL_RENDERS = min(4, max(1, (os.cpu_count() or 4) // 2))


# Audio normalization target: -14 LUFS / -1.5 dBTP is the social-media
# delivery standard (TikTok, YouTube Shorts, Instagram Reels all target this
# range). Applied to every render so client-grade audio levels are automatic.
LOUDNORM_FILTER = "loudnorm=I=-14:LRA=11:TP=-1.5"


def _run(cmd):
    print(f"$ {' '.join(str(c) for c in cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def _escape_ass_filter_path(p: Path) -> str:
    return str(p).replace(":", "\\:").replace("'", "\\'")


def _extract_clip(source: Path, cand: Candidate, out_path: Path) -> None:
    """Frame-accurate extraction. -ss after -i forces slow seek for accuracy."""
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
    """Return (width, height) of the video, using OpenCV so we get the
    displayed dimensions (with rotation metadata applied). ffprobe's csv
    output trips up on iPhone HEVC files where rotation metadata leaks into
    the field separators."""
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


# ---------- filter builders ----------

def _simple_aspect_filter(aspect: str, crop_tightness: float = 0.0,
                          speaker_rect: dict | None = None,
                          src_w: int = 0, src_h: int = 0) -> str:
    """Aspect crop. If a speaker_rect is provided and crop_tightness > 0, the
    crop is shifted/tightened so the speaker stays centered."""
    out_w, out_h = (1080, 1920) if aspect == "9:16" else \
                   ((1080, 1080) if aspect == "1:1" else (1920, 1080))

    if not speaker_rect or crop_tightness <= 0 or src_w == 0:
        # Default: scale shortest side then center-crop.
        if aspect == "9:16":
            return "scale=-2:1920,crop=1080:1920"
        if aspect == "1:1":
            return "scale=-2:1080,crop=1080:1080"
        if aspect == "16:9":
            return "scale=1920:-2,crop=1920:1080"

    # Speaker-aware crop: pick a crop window in source coords that contains the
    # speaker_rect, with the output aspect ratio. Tighter = window closer to
    # speaker_rect size. Looser = window approaches full source dimensions.
    out_aspect = out_w / out_h
    src_aspect = src_w / src_h
    cx = speaker_rect["x"] + speaker_rect["w"] / 2
    cy = speaker_rect["y"] + speaker_rect["h"] / 2

    # Decide the crop window dimensions in source pixels.
    # Loose window = source filling output aspect at full res.
    if src_aspect > out_aspect:
        loose_h = src_h
        loose_w = loose_h * out_aspect
    else:
        loose_w = src_w
        loose_h = loose_w / out_aspect

    # Tight window = the speaker_rect with output aspect ratio, padded slightly.
    s_w = speaker_rect["w"]
    s_h = speaker_rect["h"]
    pad = 1.15
    if s_w / s_h > out_aspect:
        tight_w = s_w * pad
        tight_h = tight_w / out_aspect
    else:
        tight_h = s_h * pad
        tight_w = tight_h * out_aspect

    # Lerp between loose and tight by crop_tightness.
    t = max(0.0, min(1.0, crop_tightness))
    win_w = loose_w * (1 - t) + tight_w * t
    win_h = loose_h * (1 - t) + tight_h * t
    win_w = min(win_w, src_w)
    win_h = min(win_h, src_h)

    # Position the window centered on the speaker, clamped to source bounds.
    win_x = max(0, min(src_w - win_w, cx - win_w / 2))
    win_y = max(0, min(src_h - win_h, cy - win_h / 2))

    return (
        f"crop={int(win_w)}:{int(win_h)}:{int(win_x)}:{int(win_y)},"
        f"scale={out_w}:{out_h}"
    )


def _inset_full_frame_filter(aspect: str) -> str:
    """Inset-blur composite for FULL-frame content (no speaker_rect crop).
    Used by phone_vertical and clean full_frame sources at 9:16 to produce the
    "polished social" look — source video at ~88% scale, centered, with a
    blurred copy of itself filling the small letterbox margins.

    Different from `_composite_overlay_filter` because there the foreground is
    a small region cropped out of the source; here the entire source IS the
    foreground (creator framed it intentionally, we don't crop their head off).
    """
    out_w, out_h = ASPECT_RES[aspect]
    inset_w = int(out_w * 0.88)
    inset_h = int(out_h * 0.88)
    fg = (
        f"[0:v]scale={inset_w}:{inset_h}:force_original_aspect_ratio=decrease"
        f":flags=lanczos[fg]"
    )
    bg = (
        f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h},boxblur=30:1[bg]"
    )
    composite = "[bg][fg]overlay=(W-w)/2:(H-h)/2[outv]"
    return ";".join([fg, bg, composite])


def _composite_overlay_filter(
    speaker_rect: dict,
    aspect: str,
    src_w: int,
    src_h: int,
    enhanced_fg: bool,
) -> str:
    """Build the composite filter graph.

    Two modes, selected by `enhanced_fg`:
      False (no ESRGAN): single-input.
        [0:v] crop → scale → [fg]
        [0:v] scale → crop → blur → [bg]
        [bg][fg] overlay → [outv]
      True (ESRGAN): two inputs (clip + enhanced.mp4).
        [1:v] scale → [fg]              (already-cropped, already-upscaled FG)
        [0:v] scale → crop → blur → [bg]
        [bg][fg] overlay → [outv]
    """
    out_w, out_h = ASPECT_RES[aspect]

    if enhanced_fg:
        # FG is the already-cropped, ESRGAN-upscaled video at input [1:v].
        # Just rescale it (preserving AR) to fit the output frame.
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
            f"[fg]"
        )

    bg = (
        f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h},boxblur=30:1"
        f"[bg]"
    )
    composite = "[bg][fg]overlay=(W-w)/2:(H-h)/2[outv]"
    return ";".join([fg, bg, composite])


def _zoompan_filter(framing: Framing, fps: float, duration: float) -> str:
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
        f":d=1:fps={fps}:s=iw0xih0"
    )


# ---------- render driver ----------

def _unsharp_filter(strength: float) -> str | None:
    """Strength 0-1 maps to ffmpeg unsharp amount. 0 disables."""
    if strength <= 0:
        return None
    s = max(0.0, min(1.0, strength))
    amount = round(s * 1.2, 2)  # 0..1.2 — above 1.0 starts looking artificial
    return f"unsharp=5:5:{amount}:5:5:0.0"


def frame_styles_for(framing: Framing, aspect: str) -> list[str]:
    """Decide which frame styles to render for a given (clip, aspect) pair.

    - For overlay/multi_subject content the composite path produces an
      inset-like look already; just one variant.
    - For full_frame / mid_shot / no_subject content at 9:16 we offer BOTH
      a 'full' (edge-to-edge) and an 'inset' (~88% centered + blurred
      margins) — the two styles content creators routinely deliver to TikTok
      and Reels.
    - All other (clip, aspect) combinations get only 'full'.
    """
    if framing.cls in ("overlay", "multi_subject"):
        return ["full"]
    if aspect == "9:16":
        return ["full", "inset"]
    return ["full"]


def _render_one(
    job: Job,
    source: Path,
    cand: Candidate,
    framing: Framing,
    variant_name: str,
    variant: dict,
    aspect: str,
    frame_style: str,
    words_for_clip: list[Word],
    recommendation: dict,
) -> Path:
    work = job.work_dir
    out_dir = job.output_dir

    clip_path = work / f"{cand.id}.mp4"
    _extract_clip(source, cand, clip_path)

    res = ASPECT_RES[aspect]
    ass_path = work / f"{cand.id}.{variant_name}.{aspect_slug(aspect)}.ass"
    if not ass_path.exists():
        # words_for_clip is already in trimmed-timeline coordinates
        # (zero-based); see _prepare_clip_assets → prepare_trimmed_clip.
        ass_path.write_text(build_ass(words_for_clip, variant, res, 0.0))

    # Hero aspect's 'full' frame_style gets the _hero suffix.
    is_hero = (
        aspect == recommendation.get("hero_aspect")
        and frame_style == "full"
    )
    suffix_parts = [aspect_slug(aspect)]
    if frame_style != "full":
        suffix_parts.append(frame_style)
    if is_hero:
        suffix_parts.append("hero")
    out_path = out_dir / f"{cand.id}_{variant_name}_{'_'.join(suffix_parts)}.mp4"
    if out_path.exists():
        return out_path

    crf = int(recommendation.get("crf", 20))
    sharpen = float(recommendation.get("sharpen", 0.0))
    crop_tightness = float(recommendation.get("crop_tightness", 0.0))
    want_esrgan = bool(recommendation.get("use_esrgan", False))

    use_composite = framing.cls in ("overlay", "multi_subject")
    speaker_rect = framing.signals.get("speaker_rect") if isinstance(framing.signals, dict) else None
    if use_composite and not speaker_rect:
        use_composite = False

    sharpen_f = _unsharp_filter(sharpen)

    grade_f = grade_filter_for(variant_name)
    use_inset = frame_style == "inset"  # full-frame inset-blur composite

    if use_composite:
        # Overlay/multi_subject: cropped-speaker composite with optional ESRGAN.
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
        if grade_f:
            suffix_filters.append(grade_f)
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
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(out_path),
        ]
        _run(cmd)
    elif use_inset:
        # Full-frame inset-blur composite (no speaker_rect crop — whole source
        # is the foreground, scaled to ~88%, blurred copy fills the margins).
        vf = _inset_full_frame_filter(aspect)
        suffix_filters = []
        if sharpen_f:
            suffix_filters.append(sharpen_f)
        if grade_f:
            suffix_filters.append(grade_f)
        suffix_filters.append(f"ass={_escape_ass_filter_path(ass_path)}")
        vf += ";[outv]" + ",".join(suffix_filters) + "[final]"

        _run([
            "ffmpeg", "-y",
            "-i", str(clip_path),
            "-filter_complex", vf,
            "-map", "[final]",
            "-map", "0:a",
            "-af", LOUDNORM_FILTER,
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(out_path),
        ])
    else:
        # Simple full-frame render — direct aspect crop, no composite.
        filters = []
        if framing.zoom.get("enabled"):
            fps = _probe_fps(clip_path)
            duration = cand.end - cand.start
            filters.append(_zoompan_filter(framing, fps, duration))

        src_w, src_h = _probe_source_dims(clip_path)
        filters.append(_simple_aspect_filter(
            aspect, crop_tightness=crop_tightness,
            speaker_rect=speaker_rect, src_w=src_w, src_h=src_h,
        ))
        if sharpen_f:
            filters.append(sharpen_f)
        if grade_f:
            filters.append(grade_f)
        filters.append(f"ass={_escape_ass_filter_path(ass_path)}")
        vf = ",".join(filters)
        _run([
            "ffmpeg", "-y",
            "-i", str(clip_path),
            "-vf", vf,
            "-af", LOUDNORM_FILTER,
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
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
) -> list[Word]:
    """Run heavy serial prep that the parallel renders will then consume
    from cache:
      1. Extract the clip from source
      2. Filler / long-pause trim (replaces <id>.mp4 with the trimmed version
         and returns the re-timed word list)
      3. ESRGAN + GFPGAN enhancement for overlay-class clips

    Returns: word list in TRIMMED-timeline coordinates (zero-based). The
    parallel render workers consume this list and pass clip_start=0 to the
    caption builder, since the clip is now post-trim.
    """
    clip_path = job.work_dir / f"{cand.id}.mp4"
    _extract_clip(source, cand, clip_path)

    # Filler / long-pause trim — replaces clip in place.
    clip_path, words_trimmed = prepare_trimmed_clip(
        clip_path, words_in_source, cand, job.work_dir,
    )

    if not recommendation.get("use_esrgan"):
        return words_trimmed
    if framing.cls not in ("overlay", "multi_subject"):
        return words_trimmed
    speaker_rect = (
        framing.signals.get("speaker_rect")
        if isinstance(framing.signals, dict)
        else None
    )
    if not speaker_rect:
        return words_trimmed
    enhance_speaker_region(
        clip_path, speaker_rect, cand.id, job.work_dir,
        blend=float(recommendation.get("esrgan_blend", 0.65)),
    )
    return words_trimmed


def render_all(
    job: Job,
    source: Path,
    words: list[Word],
    selected: list[Candidate],
    framings: list[Framing],
    variants: dict,
    intelligence: dict,
) -> list[Path]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.work_dir.mkdir(parents=True, exist_ok=True)

    framing_by_id = {f.id: f for f in framings}

    # Stage A — SERIAL prep. Trim + (optionally) ESRGAN. Both write caches
    # that the parallel workers below consume. Stage A is serial because:
    # ffmpeg trim per clip is fast (1 process at a time); GFPGAN must be
    # serial (one MPS-bound model, can't share across worker processes).
    per_clip_words: dict[str, list[Word]] = {}
    for cand in selected:
        framing = framing_by_id[cand.id]
        if framing.dropped:
            continue
        rec = intelligence.get(cand.id, {})
        words_for_clip = words_in_range(words, cand.start, cand.end)
        words_trimmed = _prepare_clip_assets(
            job, source, cand, framing, rec, words_for_clip,
        )
        per_clip_words[cand.id] = words_trimmed

    # Stage B — build the parallel job list. Each (clip × variant × aspect)
    # may expand to multiple frame_styles (e.g. 'full' + 'inset' at 9:16).
    jobs_list: list[tuple] = []
    for cand in selected:
        framing = framing_by_id[cand.id]
        if framing.dropped:
            print(f"[render] {cand.id}: source frame is static — dropping clip",
                  file=sys.stderr)
            continue

        rec = intelligence.get(cand.id, {})
        words_for_clip = per_clip_words.get(cand.id, [])

        for variant_name in job.variants:
            if variant_name not in variants:
                sys.exit(f"unknown variant: {variant_name}")
            variant = variants[variant_name]
            for aspect in framing.aspects:
                for frame_style in frame_styles_for(framing, aspect):
                    jobs_list.append((
                        job, source, cand, framing,
                        variant_name, variant, aspect, frame_style,
                        words_for_clip, rec,
                    ))

    print(f"[render] running {len(jobs_list)} renders, "
          f"{PARALLEL_RENDERS}-way parallel", file=sys.stderr)

    # Stage C — parallel ffmpeg renders.
    produced: list[Path] = []
    with ProcessPoolExecutor(max_workers=PARALLEL_RENDERS) as ex:
        futures = {
            ex.submit(_render_one, *args): args
            for args in jobs_list
        }
        for fut in as_completed(futures):
            args = futures[fut]
            cand_id, variant_name, aspect, fs = args[2].id, args[4], args[6], args[7]
            try:
                p = fut.result()
                produced.append(p)
                print(f"[render]   -> {p}", file=sys.stderr)
            except Exception as e:
                print(f"[render] FAILED {cand_id} × {variant_name} × {aspect} × {fs}: {e}",
                      file=sys.stderr)
                raise

    return produced
