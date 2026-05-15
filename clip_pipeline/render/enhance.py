"""Stage 4.6: Real-ESRGAN super-resolution of the speaker region.

Used for `overlay` and `multi_subject` content where the relevant subject is a
small portion of the source frame. Without ESRGAN, the composite filter just
bicubic-upscales a 200-400px region to 1080+ px output — that's the soft,
"snapshot" look. With ESRGAN, the upscale produces synthesized detail (sharp
edges, pore texture, eye definition), which is the difference between a
"home stream clip" and a client-grade deliverable.

Output: `<work>/<clip_id>.enhanced.mp4` — silent video at clip fps containing
the upscaled speaker region. The render stage composites this on top of a
blurred copy of the original frame.

Cached: re-runs after the first do not re-upscale.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(cmd):
    print(f"$ {' '.join(str(c) for c in cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


# Two-stage enhancement:
#   1. Real-ESRGAN x2 (fast ncnn-vulkan, runs on Metal): upscale background +
#      non-face regions of the speaker_rect. Uses the anime x2 model because
#      it's the fastest bundled option and faces are restored separately in
#      stage 2, so anime smoothing on the rest of the frame is fine.
#   2. GFPGAN face restoration (PyTorch + MPS): detects face in the upscaled
#      frame, restores it with face-prior-trained model, pastes back. This is
#      the step that makes Akademiks look human, not AI.
#
# Model indices in realesrgan-ncnn-py:
#   0: realesr-animevideov3-x2  (fast, what we use — 2x)
#   1-2: animevideov3 x3/x4     (slower 3x/4x of the same family)
#   3: realesrgan-x4plus-anime  (anime-specific)
#   4: realesrgan-x4plus        (general — slow, replaced by GFPGAN for faces)
DEFAULT_ESRGAN_MODEL = 0
DEFAULT_BLEND = 1.0  # 1.0 = pure ESRGAN; GFPGAN handles the "not AI" look.


_GFPGAN_RESTORER = None


def _get_gfpgan():
    """Lazy-init GFPGAN restorer. Re-uses the same model instance across
    frames and across clips (it caches model weights on disk after first use)."""
    global _GFPGAN_RESTORER
    if _GFPGAN_RESTORER is not None:
        return _GFPGAN_RESTORER
    from gfpgan import GFPGANer
    # GFPGANv1.4 with arch='clean' — best human-face restoration available
    # without the more aggressive RestoreFormer prior. upscale=1 because ESRGAN
    # already handled the geometric upscale; GFPGAN just restores texture.
    _GFPGAN_RESTORER = GFPGANer(
        model_path=(
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/"
            "GFPGANv1.4.pth"
        ),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    return _GFPGAN_RESTORER


def enhance_speaker_region(
    clip_path: Path,
    speaker_rect: dict,
    cand_id: str,
    work_dir: Path,
    model: int = DEFAULT_ESRGAN_MODEL,
    blend: float = DEFAULT_BLEND,
    use_gfpgan: bool = True,
) -> Path:
    """Two-stage enhancement for an overlay/multi_subject clip:

      1. ESRGAN x2 upscales the speaker_rect crop (fast, ncnn-vulkan/Metal).
      2. GFPGAN finds + restores the face in each upscaled frame; pastes back.

    Stage 2 is what keeps the result looking human. ESRGAN alone (especially
    with anime-trained models) over-smooths skin into a plastic look; GFPGAN
    is trained specifically on human-face priors and brings real texture
    back without over-smoothing.

    Output: silent <work>/<cand_id>.enhanced.mp4 at clip fps, cached.
    """
    import cv2
    import numpy as np
    from realesrgan_ncnn_py import Realesrgan

    cache = work_dir / f"{cand_id}.enhanced.mp4"
    if cache.exists():
        return cache

    blend = max(0.0, min(1.0, blend))

    rx = max(0, int(speaker_rect["x"]))
    ry = max(0, int(speaker_rect["y"]))
    rw = max(2, int(speaker_rect["w"]))
    rh = max(2, int(speaker_rect["h"]))

    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rw = min(rw, src_w - rx)
    rh = min(rh, src_h - ry)

    print(f"[enhance] {cand_id}: ESRGAN x2 (model={model}) "
          f"+ {'GFPGAN' if use_gfpgan else 'no GFPGAN'}, "
          f"region {rw}x{rh}, {total} frames @ {fps:.1f}fps", file=sys.stderr)

    upscaler = Realesrgan(gpuid=0, model=model)
    restorer = _get_gfpgan() if use_gfpgan else None

    def _process(crop):
        up = upscaler.process_cv2(crop)
        if blend < 1.0:
            bicubic = cv2.resize(crop, (up.shape[1], up.shape[0]),
                                 interpolation=cv2.INTER_LANCZOS4)
            up = cv2.addWeighted(up, blend, bicubic, 1.0 - blend, 0)
        if restorer is not None:
            try:
                _, _, restored = restorer.enhance(
                    up, has_aligned=False, only_center_face=False,
                    paste_back=True, weight=0.5,
                )
                if restored is not None:
                    return restored
            except Exception as e:
                # If no face detected this frame, GFPGAN may raise — keep ESRGAN.
                print(f"[enhance]   {cand_id} frame skip-gfpgan: {e}", file=sys.stderr)
        return up

    ok, first = cap.read()
    if not ok:
        sys.exit(f"could not read first frame of {clip_path}")
    first_out = _process(first[ry:ry + rh, rx:rx + rw])
    out_h, out_w = first_out.shape[:2]

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{out_w}x{out_h}",
            "-r", f"{fps}",
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-pix_fmt", "yuv420p",
            str(cache),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    proc.stdin.write(first_out.tobytes())

    processed = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = _process(frame[ry:ry + rh, rx:rx + rw])
        proc.stdin.write(out.tobytes())
        processed += 1
        if processed % 30 == 0:
            print(f"[enhance]   {cand_id}: {processed}/{total} frames",
                  file=sys.stderr)

    cap.release()
    proc.stdin.close()
    proc.wait()
    print(f"[enhance]   {cand_id}: done -> {cache} ({out_w}x{out_h}@{fps:.1f})",
          file=sys.stderr)
    return cache
