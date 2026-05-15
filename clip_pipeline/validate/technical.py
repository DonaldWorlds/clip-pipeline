"""Stage 6: verify each rendered MP4."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ..config import Job, cache_write


STILL_FRAME_DIFF_THRESHOLD = 0.5


def _ffprobe(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=codec_type,codec_name",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def _still_image_check(path: Path) -> tuple[bool, float]:
    """Extract 3 frames at 10/50/90% and compute mean abs pixel diff between
    adjacent pairs. Returns (is_still, mean_diff)."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps if fps else 0.0
    if duration <= 0:
        cap.release()
        return True, 0.0

    samples = []
    for frac in (0.1, 0.5, 0.9):
        cap.set(cv2.CAP_PROP_POS_MSEC, frac * duration * 1000.0)
        ok, fr = cap.read()
        if ok and fr is not None:
            samples.append(cv2.resize(fr, (320, 180)))
    cap.release()
    if len(samples) < 2:
        return True, 0.0

    diffs = []
    for i in range(len(samples) - 1):
        diffs.append(float(np.mean(
            np.abs(samples[i].astype("int16") - samples[i + 1].astype("int16"))
        )))
    mean_diff = float(sum(diffs) / len(diffs))
    return mean_diff < STILL_FRAME_DIFF_THRESHOLD, mean_diff


def run(job: Job, produced: list[Path]) -> dict:
    """Run all verification checks on rendered outputs. Returns a report dict
    and writes it to <output_dir>/report.json. Caller decides exit code."""
    report = {"checked": [], "failures": []}

    for p in produced:
        entry = {"file": str(p), "checks": {}}
        try:
            probe = _ffprobe(p)
            has_video = any(s.get("codec_type") == "video" for s in probe.get("streams", []))
            has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
            duration = float(probe.get("format", {}).get("duration", 0))
            entry["checks"]["duration_s"] = round(duration, 2)
            entry["checks"]["has_video"] = has_video
            entry["checks"]["has_audio"] = has_audio

            if duration < 1.0 or not has_video or not has_audio:
                entry["status"] = "FAIL_PROBE"
                report["failures"].append(entry)
                continue

            is_still, mean_diff = _still_image_check(p)
            entry["checks"]["mean_frame_diff"] = round(mean_diff, 3)
            entry["checks"]["is_still"] = is_still
            if is_still:
                entry["status"] = "FAIL_STILL"
                report["failures"].append(entry)
                continue

            entry["status"] = "OK"
            report["checked"].append(entry)
        except Exception as e:
            entry["status"] = "FAIL_ERROR"
            entry["error"] = str(e)
            report["failures"].append(entry)

    report["summary"] = {
        "total": len(produced),
        "ok": len(report["checked"]),
        "failed": len(report["failures"]),
    }
    cache_write(job.output_dir / "report.json", report)
    return report
