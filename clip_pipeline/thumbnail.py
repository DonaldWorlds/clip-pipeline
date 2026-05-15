"""Stage 5.5: per-clip thumbnail extraction.

For each rendered clip, sample frames across the clip duration and pick the
"best" thumbnail — heuristic-based:

  - Has a face (so the speaker is on-camera, not mid-cut to b-roll)
  - Face area is large (expressive close-up beats a wide shot)
  - Image is sharp (Laplacian variance — rejects motion-blurred frames)

Output: <out_dir>/<clip_id>_thumb.jpg at the source's native resolution.

These are deliverables you hand to the client alongside the MP4s — every short
needs a thumbnail.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .config import Candidate, Job


SAMPLE_COUNT = 12     # frames to sample per clip
MIN_FACE_AREA_PCT = 0.5  # very permissive — even a tiny webcam-overlay face is usable


def _sharpness(gray) -> float:
    """Laplacian variance — higher = sharper. ~0 means motion blur."""
    import cv2
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _best_thumbnail_frame(source: Path, cand: Candidate, face_cascade,
                          speaker_rect: dict | None = None):
    """Score frames in the clip range, return (BGR frame, score, score_breakdown).
    If speaker_rect is provided, crop to it first (overlay-class content has
    the actual subject in a small region of the frame; a thumbnail of the
    whole 1920x1080 screen recording is useless)."""
    import cv2
    cap = cv2.VideoCapture(str(source))
    duration = cand.end - cand.start
    best_frame = None
    best_score = 0.0
    best_break = {}

    # Two face-detector passes — tight first (high quality), loose fallback
    # (catches small webcam faces). We try tight on every sample; if nothing
    # scores well, we re-pass with loose.
    detect_passes = [
        {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (40, 40), "tag": "tight"},
        {"scaleFactor": 1.1, "minNeighbors": 3, "minSize": (20, 20), "tag": "loose"},
    ]

    for pass_cfg in detect_passes:
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        for i in range(SAMPLE_COUNT):
            t = cand.start + duration * (0.1 + 0.8 * i / max(1, SAMPLE_COUNT - 1))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # For overlay-class sources, crop to the speaker_rect first so the
            # thumbnail is OF THE SUBJECT, not the screen recording UI.
            scan_frame = frame
            if speaker_rect:
                rx = max(0, int(speaker_rect.get("x", 0)))
                ry = max(0, int(speaker_rect.get("y", 0)))
                rw = max(2, int(speaker_rect.get("w", frame.shape[1])))
                rh = max(2, int(speaker_rect.get("h", frame.shape[0])))
                rw = min(rw, frame.shape[1] - rx)
                rh = min(rh, frame.shape[0] - ry)
                scan_frame = frame[ry:ry + rh, rx:rx + rw]

            h, w = scan_frame.shape[:2]
            gray = cv2.cvtColor(scan_frame, cv2.COLOR_BGR2GRAY)
            sharp = _sharpness(gray)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=pass_cfg["scaleFactor"],
                minNeighbors=pass_cfg["minNeighbors"],
                minSize=pass_cfg["minSize"],
            )
            if len(faces) == 0:
                continue
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            face_pct = 100.0 * (fw * fh) / (w * h)
            if face_pct < MIN_FACE_AREA_PCT:
                continue
            score = face_pct * 10 + min(sharp, 1500) / 50
            if score > best_score:
                best_frame = scan_frame.copy()
                best_score = score
                best_break = {"t": round(t, 2), "pass": pass_cfg["tag"],
                              "face_pct": round(face_pct, 1),
                              "sharpness": round(sharp, 1)}
        if best_frame is not None:
            break  # tight pass succeeded — no need for loose

    # Final fallback: take the midpoint frame and crop to speaker_rect (or
    # the full frame). Never return None — every clip gets a thumbnail.
    if best_frame is None:
        cap.set(cv2.CAP_PROP_POS_MSEC, (cand.start + duration / 2) * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            if speaker_rect:
                rx = max(0, int(speaker_rect.get("x", 0)))
                ry = max(0, int(speaker_rect.get("y", 0)))
                rw = min(int(speaker_rect.get("w", frame.shape[1])),
                         frame.shape[1] - rx)
                rh = min(int(speaker_rect.get("h", frame.shape[0])),
                         frame.shape[0] - ry)
                best_frame = frame[ry:ry + rh, rx:rx + rw].copy()
            else:
                best_frame = frame
            best_break = {"fallback": "midpoint"}

    cap.release()
    return best_frame, best_score, best_break


def extract_for_clip(source: Path, cand: Candidate, face_cascade, out_dir: Path,
                     speaker_rect: dict | None = None) -> Path | None:
    """Extract the best thumbnail for one clip."""
    import cv2
    out_path = out_dir / f"{cand.id}_thumb.jpg"
    if out_path.exists():
        return out_path

    frame, score, breakdown = _best_thumbnail_frame(
        source, cand, face_cascade, speaker_rect,
    )
    if frame is None:
        print(f"[thumb] {cand.id}: no usable frame at all (bad source?)",
              file=sys.stderr)
        return None
    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[thumb] {cand.id}: {out_path.name}  {breakdown}", file=sys.stderr)
    return out_path


def extract_all(job: Job, source: Path, selected: list[Candidate],
                framings: list | None = None) -> list[Path]:
    """Stage 5.5: extract per-clip thumbnails. If framings is provided and a
    clip is overlay/multi_subject, the thumbnail is taken from inside the
    speaker_rect rather than the full source frame."""
    import cv2
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        sys.exit(f"Failed to load Haar cascade from {cascade_path}")

    framing_by_id = {f.id: f for f in (framings or [])}

    saved: list[Path] = []
    for cand in selected:
        framing = framing_by_id.get(cand.id)
        speaker_rect = None
        if framing and framing.cls in ("overlay", "multi_subject"):
            sigs = framing.signals if isinstance(framing.signals, dict) else {}
            speaker_rect = sigs.get("speaker_rect")
        p = extract_for_clip(source, cand, face_cascade, job.output_dir,
                             speaker_rect)
        if p:
            saved.append(p)
    return saved
