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
)
from ..analyze.whisper import words_in_range
from ..profile import ClientStyleProfile
from .enhance import enhance_speaker_region
from .grade import build_grade_filter
from .trim import prepare_trimmed_clip


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
            return "scale=-2:1920,crop=1080:1920"
        if aspect == "1:1":
            return "scale=-2:1080,crop=1080:1080"
        if aspect == "16:9":
            return "scale=1920:-2,crop=1920:1080"

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
        f"scale={out_w}:{out_h}"
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
            f"[fg]"
        )

    bg = (
        f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h},boxblur=30:1"
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
) -> Path:
    work = job.work_dir
    out_dir = job.output_dir

    clip_path = work / f"{cand.id}.mp4"
    _extract_clip(source, cand, clip_path)

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

    use_composite = framing.cls in ("overlay", "multi_subject")
    speaker_rect = framing.signals.get("speaker_rect") if isinstance(framing.signals, dict) else None
    if use_composite and not speaker_rect:
        use_composite = False

    sharpen_f = _unsharp_filter(sharpen)

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
                "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
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
                "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
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
) -> list[Word]:
    """Serial prep: extract, trim, optionally ESRGAN-enhance. Returns the
    word list re-timed to the trimmed clip's zero-based timeline.

    ESRGAN gate is purely profile-driven: if `profile.use_upscaling` is True
    AND `recommendation.use_esrgan` agrees, we enhance. For overlay/multi-
    subject sources we pass the speaker_rect (subject-region enhancement);
    for everything else we pass a full-frame synthetic rect so the whole
    frame gets upscaled. The render path consumes the .enhanced.mp4 either
    as the foreground of the composite filter (overlay) or as the direct
    ffmpeg input (simple aspect crop).
    """
    clip_path = job.work_dir / f"{cand.id}.mp4"
    _extract_clip(source, cand, clip_path)

    clip_path, words_trimmed = prepare_trimmed_clip(
        clip_path, words_in_source, cand, job.work_dir,
    )

    if not (profile.use_upscaling and recommendation.get("use_esrgan", False)):
        return words_trimmed

    sig = framing.signals if isinstance(framing.signals, dict) else {}
    speaker_rect = sig.get("speaker_rect")
    if not speaker_rect:
        # Full-frame enhancement: synthesize a rect spanning the whole frame.
        src_w, src_h = _probe_source_dims(clip_path)
        speaker_rect = {"x": 0, "y": 0, "w": src_w, "h": src_h}

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
    profile: ClientStyleProfile,
    decisions_by_id: dict,
) -> list[Path]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.work_dir.mkdir(parents=True, exist_ok=True)

    framing_by_id = {f.id: f for f in framings}
    caption_dict = _profile_to_caption_dict(profile)
    grade_filter = build_grade_filter(profile.color_grade)

    # Stage A — serial prep (trim + optional ESRGAN, must be serial because
    # GFPGAN/ESRGAN can't share an MPS device across worker processes).
    per_clip_words: dict[str, list[Word]] = {}
    for cand in selected:
        framing = framing_by_id[cand.id]
        if framing.dropped:
            continue
        rec = decisions_by_id.get(cand.id, {})
        words_for_clip = words_in_range(words, cand.start, cand.end)
        words_trimmed = _prepare_clip_assets(
            job, source, cand, framing, rec, words_for_clip, profile,
        )
        per_clip_words[cand.id] = words_trimmed

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

        for aspect in profile.delivery_aspects:
            jobs_list.append((
                job, source, cand, framing, aspect,
                caption_dict, words_for_clip, rec, grade_filter,
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
