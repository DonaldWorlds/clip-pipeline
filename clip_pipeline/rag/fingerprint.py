"""Reference-clip fingerprinting — the shared feature extractor used by:

  - rag/ingest.py        (when n8n drops a new reference clip into the inbox)
  - validate/editorial.py (later, for comparing OUR render to references)

Three signal layers per video:

  1. Transcript text (faster-whisper, single pass)
  2. Numeric/visual + editorial signatures (OpenCV + numpy on sampled frames)
  3. Pegasus natural-language description (TwelveLabs free tier) — describes
     hook / pacing / captions / payoff in plain English

Plus embeddings of the transcript and the Pegasus description so retrieval
can match on content AND on style/vibe.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class VisualSignature:
    avg_brightness: float = 0.0       # 0-255
    color_temp: str = "neutral"        # warm | neutral | cool
    motion_score: float = 0.0          # mean frame-diff
    cuts_per_sec: float = 0.0
    face_present_pct: float = 0.0
    face_position: str = "center"      # left | center | right | none


@dataclass
class EditorialSignature:
    zoom_cadence_s: float = 0.0
    caption_present: bool = False
    caption_y_frac: float = 0.85
    caption_font_px: int = 0
    max_silence_s: float = 0.0
    dominant_aspect: str = "9:16"
    fps: float = 30.0


@dataclass
class PegasusDescription:
    hook: str = ""
    pacing: str = ""
    captions: str = ""
    payoff: str = ""

    def joined(self) -> str:
        return " ".join([self.hook, self.pacing, self.captions, self.payoff]).strip()


@dataclass
class ClipFingerprint:
    id: str
    duration_s: float = 0.0
    transcript: str = ""
    visual: VisualSignature = field(default_factory=VisualSignature)
    editorial: EditorialSignature = field(default_factory=EditorialSignature)
    pegasus: PegasusDescription = field(default_factory=PegasusDescription)

    def to_dict(self) -> dict:
        return asdict(self)


# -------- Pegasus prompt --------

PEGASUS_DESCRIBE_PROMPT = """This is a short-form video clip in the {niche} niche.
It got {views} views and {likes} likes.

In one paragraph each, describe:
1. THE HOOK — what happens in the first 2 seconds that grabs attention.
2. PACING — speed of cuts, zooms, and visual changes.
3. CAPTION STYLE — caption position, font size relative to frame, color, animation.
4. PAYOFF — the moment of peak engagement and how it's delivered.

Return STRICT JSON only (no markdown, no commentary):
{{"hook": "...", "pacing": "...", "captions": "...", "payoff": "..."}}
""".strip()


def _ffprobe_duration_fps(path: Path) -> tuple[float, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration:stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    duration = 0.0
    fps = 30.0
    for line in out.splitlines():
        if line.startswith("duration="):
            try:
                duration = float(line.split("=", 1)[1])
            except ValueError:
                pass
        elif line.startswith("r_frame_rate="):
            v = line.split("=", 1)[1]
            try:
                num, _, den = v.partition("/")
                fps = float(num) / float(den) if den else float(num)
            except (ValueError, ZeroDivisionError):
                pass
    return duration, fps


def _visual_fingerprint(video_path: Path, n_samples: int = 12) -> VisualSignature:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    duration_s = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / max(cap.get(cv2.CAP_PROP_FPS) or 30, 1)
    frames = []
    for i in range(n_samples):
        t = duration_s * (0.05 + 0.9 * i / max(1, n_samples - 1))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()

    if not frames:
        return VisualSignature()

    h, w = frames[0].shape[:2]
    short_side = min(h, w)

    # Brightness + color temp
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    brightness = float(np.mean([np.mean(g) for g in grays]))
    # crude color temp: avg red vs avg blue across frames
    mean_b = float(np.mean([np.mean(f[:, :, 0]) for f in frames]))
    mean_r = float(np.mean([np.mean(f[:, :, 2]) for f in frames]))
    if mean_r - mean_b > 8:
        color_temp = "warm"
    elif mean_b - mean_r > 8:
        color_temp = "cool"
    else:
        color_temp = "neutral"

    # Motion (frame diff at low res)
    target_w = 320
    scaled = []
    for f in frames:
        scale = target_w / f.shape[1]
        scaled.append(cv2.resize(f, (target_w, int(f.shape[0] * scale))))
    diffs = []
    for i in range(len(scaled) - 1):
        diffs.append(float(np.mean(np.abs(scaled[i].astype("int16") - scaled[i + 1].astype("int16")))))
    motion = float(np.mean(diffs)) if diffs else 0.0

    # Cuts/sec — count adjacent pairs where diff > threshold (heuristic)
    cut_threshold = 35.0  # diff over 35 is likely a hard cut
    n_cuts = sum(1 for d in diffs if d > cut_threshold)
    cuts_per_sec = n_cuts / max(duration_s, 0.1)

    # Face presence + position
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    face_present = 0
    centers = []
    min_face = max(int(short_side * 0.06), 24)
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                              minSize=(min_face, min_face))
        if len(faces) > 0:
            face_present += 1
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            centers.append((x + fw / 2) / w)
    face_pct = 100.0 * face_present / len(frames)
    if centers:
        avg_x = sum(centers) / len(centers)
        if avg_x < 0.35:
            face_position = "left"
        elif avg_x > 0.65:
            face_position = "right"
        else:
            face_position = "center"
    else:
        face_position = "none"

    return VisualSignature(
        avg_brightness=round(brightness, 1),
        color_temp=color_temp,
        motion_score=round(motion, 2),
        cuts_per_sec=round(cuts_per_sec, 2),
        face_present_pct=round(face_pct, 1),
        face_position=face_position,
    )


def _editorial_fingerprint(visual: VisualSignature, duration_s: float, fps: float) -> EditorialSignature:
    # zoom_cadence is approximated by cuts/sec — true zoom detection would need
    # optical flow; we keep it cheap here.
    if visual.cuts_per_sec > 0:
        zoom_cadence_s = round(1.0 / visual.cuts_per_sec, 2)
    else:
        zoom_cadence_s = 0.0
    # The clip's dominant aspect — we'd derive from frame dims; rough fallback.
    aspect = "9:16"
    return EditorialSignature(
        zoom_cadence_s=zoom_cadence_s,
        caption_present=False,            # we don't OCR raw references
        caption_y_frac=0.85,               # default; will get overridden when we have OCR
        caption_font_px=0,
        max_silence_s=0.0,                 # would need transcript-level analysis
        dominant_aspect=aspect,
        fps=fps,
    )


# Module-level cache to suppress repeated Pegasus error noise during a
# batch ingest of references. First failure prints a single explanatory
# message; subsequent calls return silently.
_PEGASUS_AVAILABLE: bool | None = None


def _pegasus_description(
    video_id: str,
    niche: str,
    engagement: dict,
) -> PegasusDescription:
    """Call Pegasus to describe the reference clip. Returns empty PegasusDescription
    on any failure (we don't crash the ingest)."""
    global _PEGASUS_AVAILABLE
    if _PEGASUS_AVAILABLE is False:
        return PegasusDescription()  # silent skip — known unavailable this run
    import os
    if not os.environ.get("TWELVELABS_API_KEY"):
        return PegasusDescription()
    try:
        from twelvelabs import TwelveLabs
        client = TwelveLabs(api_key=os.environ["TWELVELABS_API_KEY"])
        prompt = PEGASUS_DESCRIBE_PROMPT.format(
            niche=niche,
            views=engagement.get("views", "unknown"),
            likes=engagement.get("likes", "unknown"),
        )
        resp = client.analyze(
            video_id=video_id,
            model_name="pegasus1.2",
            prompt=prompt,
            temperature=0.3,
        )
    except Exception as e:
        msg = str(e)
        if "index_not_supported_for_generate" in msg or "video_id is not supported" in msg:
            if _PEGASUS_AVAILABLE is None:
                print(f"[twelvelabs] Pegasus generate not supported on this index — "
                      f"reference descriptions will be left blank for this batch",
                      file=sys.stderr)
            _PEGASUS_AVAILABLE = False
            return PegasusDescription()
        print(f"[fingerprint.pegasus] failed: {e}", file=sys.stderr)
        return PegasusDescription()
    _PEGASUS_AVAILABLE = True

    text = getattr(resp, "data", None) or getattr(resp, "text", None) or str(resp)
    text = text.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return PegasusDescription()
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return PegasusDescription()

    return PegasusDescription(
        hook=str(parsed.get("hook", "")),
        pacing=str(parsed.get("pacing", "")),
        captions=str(parsed.get("captions", "")),
        payoff=str(parsed.get("payoff", "")),
    )


def fingerprint_clip(
    video_path: Path,
    clip_id: str,
    niche: str = "unknown",
    engagement: dict | None = None,
    twelvelabs_video_id: str | None = None,
    transcribe: bool = True,
) -> ClipFingerprint:
    """Full fingerprint for a video file. Resilient — partial signatures are
    returned even if individual layers fail."""
    duration, fps = _ffprobe_duration_fps(video_path)

    # Transcript (optional — may be slow on long files)
    transcript = ""
    if transcribe:
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("small", device="auto", compute_type="auto")
            segments, _ = model.transcribe(
                str(video_path),
                word_timestamps=False,
                vad_filter=True,
                condition_on_previous_text=False,
                beam_size=5,
            )
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            print(f"[fingerprint.transcript] failed: {e}", file=sys.stderr)

    visual = _visual_fingerprint(video_path)
    editorial = _editorial_fingerprint(visual, duration, fps)
    pegasus = (
        _pegasus_description(twelvelabs_video_id, niche, engagement or {})
        if twelvelabs_video_id else PegasusDescription()
    )

    return ClipFingerprint(
        id=clip_id,
        duration_s=round(duration, 2),
        transcript=transcript,
        visual=visual,
        editorial=editorial,
        pegasus=pegasus,
    )
