"""Stage 4.5: framing + motion analysis (OpenCV face + HOG person + frame diff).

Content-aware:
  - Detects a stable speaker_rect (a tight rectangle around the actual subject,
    even if they're a small webcam overlay on a screen recording).
  - Classifies the content type based on speaker_rect size relative to the frame:
    full_frame   — speaker fills most of the frame (clean talking-head/full-body)
    overlay      — speaker is a small region (streamer-overlay, screen capture)
    multi_subject — multiple distinct face clusters of similar size
    no_subject   — no reliable speaker region found
  - Always renders all 3 aspect ratios. The render stage uses speaker_rect to
    crop or composite intelligently so 9:16, 1:1, and 16:9 all look acceptable
    regardless of content type.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..config import Candidate, Framing, Job, cache_read, cache_write


# ---------- thresholds ----------
SAMPLE_FRAMES = 12

# Haar cascade — tighter than v1 to suppress YouTube-thumbnail false positives.
FACE_MIN_SIZE = (40, 40)
FACE_SCALE = 1.15
FACE_NEIGHBORS = 6

# Drop tiny detections in post-process: only count faces ≥ this fraction of the
# frame's shorter side. Filters out background-thumbnail "faces".
MIN_FACE_FRACTION_OF_FRAME = 0.06

# Speaker-rect clustering
CLUSTER_GRID = 12               # 12×12 grid bins for face position binning
MIN_CLUSTER_HITS = 4            # cluster must have ≥4 face hits to be a candidate
SECOND_CLUSTER_RATIO = 0.5      # if 2nd cluster ≥ 50% of 1st in size → multi_subject

# Content-type thresholds (speaker_rect dimensions / frame)
OVERLAY_RECT_FRAC = 0.40        # rect width or height < 40% of frame → overlay
FULL_FRAME_RECT_FRAC = 0.60     # rect ≥ 60% of frame → full_frame

# Motion
STATIC_FRAME_DIFF = 6.0
STATIC_SUBJECT_VAR = 200
FROZEN_FRAME_DIFF = 0.5

ZOOM_START = 1.00
ZOOM_END = 1.08

# All aspects rendered for every clip — client wants delivery flexibility.
ALL_ASPECTS = ["9:16", "1:1", "16:9"]


def _sample_frames(source: Path, t_start: float, t_end: float,
                   n: int = SAMPLE_FRAMES):
    import cv2
    cap = cv2.VideoCapture(str(source))
    frames = []
    for i in range(n):
        t = t_start + (t_end - t_start) * (i + 0.5) / n
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


def _detect_faces(face_cascade, frame, frame_short_side: int):
    """Detect faces, filter out tiny ones (likely background)."""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE,
        minNeighbors=FACE_NEIGHBORS,
        minSize=FACE_MIN_SIZE,
    )
    min_face_side = MIN_FACE_FRACTION_OF_FRAME * frame_short_side
    return [
        (int(x), int(y), int(w), int(h))
        for (x, y, w, h) in faces
        if min(w, h) >= min_face_side
    ]


def _detect_persons(hog, frame):
    """HOG person detection on a downscaled frame for speed."""
    import cv2
    h0, w0 = frame.shape[:2]
    target_w = 640
    if w0 > target_w:
        scale = target_w / w0
        small = cv2.resize(frame, (target_w, int(h0 * scale)))
    else:
        scale = 1.0
        small = frame
    rects, _ = hog.detectMultiScale(small, winStride=(8, 8),
                                    padding=(16, 16), scale=1.05)
    return [
        (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
        for (x, y, w, h) in rects
    ]


def _frame_diff_mean(a, b) -> float:
    import cv2
    import numpy as np
    target_w = 320
    if a.shape[1] > target_w:
        scale = target_w / a.shape[1]
        a = cv2.resize(a, (target_w, int(a.shape[0] * scale)))
        b = cv2.resize(b, (target_w, int(b.shape[0] * scale)))
    if a.shape != b.shape:
        return 0.0
    return float(np.mean(np.abs(a.astype("int16") - b.astype("int16"))))


def _cluster_faces(face_observations: list[tuple[float, float, int, int]],
                   frame_w: int, frame_h: int) -> list[dict]:
    """Cluster face bbox centers across frames into spatial groups. Returns a
    list of clusters, sorted by member count desc. Each cluster has:
        {count, avg_center: (cx, cy), avg_size: (w, h), bbox: (x, y, w, h)}
    """
    import numpy as np
    if not face_observations:
        return []
    bin_w = frame_w / CLUSTER_GRID
    bin_h = frame_h / CLUSTER_GRID
    bins: dict[tuple[int, int], list] = {}
    for (cx, cy, w, h) in face_observations:
        key = (int(cx / bin_w), int(cy / bin_h))
        bins.setdefault(key, []).append((cx, cy, w, h))

    clusters = []
    for key, members in bins.items():
        if len(members) < MIN_CLUSTER_HITS:
            continue
        cxs = [m[0] for m in members]
        cys = [m[1] for m in members]
        ws = [m[2] for m in members]
        hs = [m[3] for m in members]
        avg_cx = float(np.mean(cxs))
        avg_cy = float(np.mean(cys))
        avg_w = float(np.mean(ws))
        avg_h = float(np.mean(hs))
        # Tight bbox around the cluster + 40% padding for hair/shoulders.
        pad_w = avg_w * 0.7
        pad_h = avg_h * 0.7
        x = max(0.0, avg_cx - avg_w / 2 - pad_w)
        y = max(0.0, avg_cy - avg_h / 2 - pad_h)
        w = min(frame_w - x, avg_w + 2 * pad_w)
        h = min(frame_h - y, avg_h + 2 * pad_h)
        clusters.append({
            "count": len(members),
            "avg_center": (avg_cx, avg_cy),
            "avg_size": (avg_w, avg_h),
            "bbox": (x, y, w, h),
        })
    clusters.sort(key=lambda c: -c["count"])
    return clusters


def _classify_content(clusters: list[dict], person_pct: float,
                      frame_w: int, frame_h: int) -> tuple[str, dict | None]:
    """Return (class, speaker_rect_dict-or-None). Speaker rect is in source-frame
    coordinates: {x, y, w, h, x_frac, y_frac, w_frac, h_frac}."""
    if not clusters:
        # Fall back to "no_subject"; renderer will preserve original framing.
        return "no_subject", None

    primary = clusters[0]
    px, py, pw, ph = primary["bbox"]
    primary_short = min(pw, ph) / min(frame_w, frame_h)

    # Multi-subject test: a second cluster with comparable count
    is_multi = (
        len(clusters) >= 2
        and clusters[1]["count"] >= primary["count"] * SECOND_CLUSTER_RATIO
        and clusters[1]["count"] >= MIN_CLUSTER_HITS
    )
    if is_multi:
        # Speaker rect spans the union of both clusters.
        x1, y1, w1, h1 = clusters[1]["bbox"]
        ux1 = min(px, x1)
        uy1 = min(py, y1)
        ux2 = max(px + pw, x1 + w1)
        uy2 = max(py + ph, y1 + h1)
        union_w = ux2 - ux1
        union_h = uy2 - uy1
        rect = {
            "x": ux1, "y": uy1, "w": union_w, "h": union_h,
            "x_frac": ux1 / frame_w, "y_frac": uy1 / frame_h,
            "w_frac": union_w / frame_w, "h_frac": union_h / frame_h,
        }
        return "multi_subject", rect

    rect = {
        "x": px, "y": py, "w": pw, "h": ph,
        "x_frac": px / frame_w, "y_frac": py / frame_h,
        "w_frac": pw / frame_w, "h_frac": ph / frame_h,
    }
    if primary_short >= FULL_FRAME_RECT_FRAC:
        return "full_frame", rect
    if primary_short < OVERLAY_RECT_FRAC:
        # Speaker is a small region — likely a webcam overlay on a screen recording.
        return "overlay", rect
    # In between — mid_shot / partial. Treat like full_frame for rendering, but
    # tag separately so the report shows it.
    return "mid_shot", rect


def analyze_clip(source: Path, cand: Candidate, face_cascade, hog) -> Framing:
    import numpy as np
    frames = _sample_frames(source, cand.start, cand.end)
    if not frames:
        return Framing(
            id=cand.id, cls="no_subject", static=False,
            zoom={"enabled": False, "start": 1.0, "end": 1.0,
                  "center_x": 0.5, "center_y": 0.5},
            aspects=list(ALL_ASPECTS),
            signals={"error": "no frames sampled"},
            dropped=True,
        )

    h0, w0 = frames[0].shape[:2]
    short_side = min(h0, w0)

    # Accumulate face detections (center + size) across all frames.
    face_obs: list[tuple[float, float, int, int]] = []
    person_present = 0
    for fr in frames:
        for (x, y, w, h) in _detect_faces(face_cascade, fr, short_side):
            face_obs.append((x + w / 2, y + h / 2, w, h))
        if _detect_persons(hog, fr):
            person_present += 1

    person_pct = 100.0 * person_present / len(frames)
    clusters = _cluster_faces(face_obs, w0, h0)

    cls, speaker_rect = _classify_content(clusters, person_pct, w0, h0)

    # Motion analysis
    diffs = [_frame_diff_mean(frames[i], frames[i + 1])
             for i in range(len(frames) - 1)]
    frame_diff_mean = float(np.mean(diffs)) if diffs else 0.0

    # Zoom center: speaker rect center; fallback to image center.
    if speaker_rect:
        cx_frac = speaker_rect["x_frac"] + speaker_rect["w_frac"] / 2
        cy_frac = speaker_rect["y_frac"] + speaker_rect["h_frac"] / 2
        subj_var = (
            float(np.var([c["avg_center"][0] for c in clusters[:1]]))
            if clusters else 0.0
        )
    else:
        cx_frac = 0.5
        cy_frac = 0.5
        subj_var = 0.0

    dropped = frame_diff_mean < FROZEN_FRAME_DIFF
    static = (frame_diff_mean < STATIC_FRAME_DIFF) and (subj_var < STATIC_SUBJECT_VAR)
    # Zoom is only safe on full-frame content. Don't zoom overlays (would crop
    # the overlay) or multi-subject (would crop someone out).
    zoom_enabled = static and cls in ("full_frame", "mid_shot") and not dropped

    signals = {
        "face_observations": len(face_obs),
        "face_clusters": len(clusters),
        "primary_cluster_hits": clusters[0]["count"] if clusters else 0,
        "person_present_pct": round(person_pct, 1),
        "frame_diff_mean": round(frame_diff_mean, 2),
        "subject_position_variance": round(subj_var, 1),
    }
    if speaker_rect:
        signals["speaker_rect_w_frac"] = round(speaker_rect["w_frac"], 3)
        signals["speaker_rect_h_frac"] = round(speaker_rect["h_frac"], 3)

    framing = Framing(
        id=cand.id,
        cls=cls,
        static=static,
        zoom={
            "enabled": zoom_enabled,
            "start": ZOOM_START,
            "end": ZOOM_END,
            "center_x": round(cx_frac, 4),
            "center_y": round(cy_frac, 4),
        },
        aspects=list(ALL_ASPECTS),
        signals=signals,
        dropped=dropped,
    )
    # Stash speaker_rect on the framing for the renderer; it's a dict on signals
    # so it survives the dataclass→json round-trip.
    if speaker_rect:
        framing.signals["speaker_rect"] = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in speaker_rect.items()
        }
    return framing


def _apply_source_profile(f: Framing, source_profile: str) -> Framing:
    """Apply source-shape-based overrides to a per-clip Framing.

    The face-cluster classifier doesn't know whether a small face means
    "tiny webcam in a screen recording" (overlay → composite) or "phone
    recording where the creator stood back from the camera" (full_frame →
    direct render). The source aspect ratio resolves the ambiguity.

    phone_vertical → whole frame is the subject. Force class to full_frame
    if the classifier picked overlay/multi_subject (would have triggered
    composite). Output 9:16 only (matches source). No zoom.

    square → similar idea, 1:1 only.

    landscape and unknown → no override; current logic stands.
    """
    if source_profile == "phone_vertical":
        if f.cls in ("overlay", "multi_subject"):
            f.cls = "full_frame"
        f.aspects = ["9:16"]
        # Don't zoom phone content — creator framed it intentionally.
        f.zoom["enabled"] = False
    elif source_profile == "square":
        if f.cls in ("overlay", "multi_subject"):
            f.cls = "full_frame"
        f.aspects = ["1:1"]
        f.zoom["enabled"] = False
    # landscape / unknown: leave as-is.
    return f


def analyze(job: Job, source: Path, selected: list[Candidate],
            source_profile: str = "unknown") -> list[Framing]:
    """Stage 4.5: produce framing.json. Cached per job."""
    import cv2
    cache = job.work_dir / "framing.json"
    if cache.exists():
        return [Framing(**d) for d in cache_read(cache)]

    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        sys.exit(f"Failed to load Haar cascade from {face_cascade_path}")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    out: list[Framing] = []
    for cand in selected:
        print(f"[framing] analyzing {cand.id} [{cand.start:.1f}-{cand.end:.1f}]",
              file=sys.stderr)
        f = analyze_clip(source, cand, face_cascade, hog)
        f = _apply_source_profile(f, source_profile)
        sr = f.signals.get("speaker_rect") if isinstance(f.signals, dict) else None
        sr_desc = (
            f"speaker_rect={sr['w_frac']:.2f}×{sr['h_frac']:.2f}"
            if sr else "no speaker_rect"
        )
        print(f"[framing]   -> {f.cls} (profile={source_profile}), "
              f"{sr_desc}, aspects={f.aspects}, "
              f"zoom={f.zoom['enabled']}, dropped={f.dropped}",
              file=sys.stderr)
        out.append(f)
    cache_write(cache, out)
    return out
