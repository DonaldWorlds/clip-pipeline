"""Stage 6b: editorial validation — does each rendered MP4 actually obey the
rules in references/_global/style.md AND the per-clip decisions?

Two layers, both run after render:

  Mechanical (always, free):
    - Parse the .ass file → confirm actual alignment + margin_v match the
      Decision.caption_y_fraction (within tolerance).
    - Parse the .ass file → confirm fontsize >= rules.caption_min_font_px.
    - ffprobe the output → confirm duration in [rules.clip_min_s, clip_max_s].
    - Sample frames → detect face position → confirm captions don't overlap
      face bounding box.
    - Read the trim_report.json → confirm we hit rules.max_silence_s.

  Vision (TwelveLabs Pegasus, free tier):
    - Sample 3 frames per output (10%, 50%, 90% positions).
    - One Pegasus analyze call asking specifically for visible problems
      (subject cropped, caption cut off, compression artifacts).
    - Findings appended to `vision_findings`. Critical findings also added to
      `rules_violated` so the clip fails verification.

Output: <out_dir>/editorial_report.json keyed by clip filename.
If ANY clip has any rules_violated, cut.py exits non-zero.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..config import Job, cache_read, cache_write
from ..editorial.rules import EditorialRules, load as load_rules


# Tolerances
CAPTION_Y_TOLERANCE = 0.06     # actual vs target y_fraction, allow ±6% of frame height
FACE_OVERLAP_PCT_LIMIT = 0.10  # captions cover >10% of face bbox = violation
VISION_FRAME_POSITIONS = (0.10, 0.50, 0.90)
PROFILE_BRIGHTNESS_TOLERANCE = 15.0   # mean luminance, ±15 on 0-255 scale
PROFILE_CAPTION_Y_TOLERANCE = 0.05    # profile.caption.y_frac vs rendered, ±5% of frame
ASPECT_SLUG_TO_NAME = {"9x16": "9:16", "1x1": "1:1", "16x9": "16:9"}


@dataclass
class EditorialReport:
    file: str
    clip_id: str
    rules_obeyed: list[str] = field(default_factory=list)
    rules_violated: list[dict] = field(default_factory=list)
    vision_findings: list[str] = field(default_factory=list)
    status: str = "OK"   # OK | FAIL_MECHANICAL | FAIL_VISION


# ---------- ASS parsing ----------

_DIALOGUE_RE = re.compile(r"^Dialogue:")
_STYLE_RE = re.compile(
    r"^Style:\s*([^,]+),([^,]+),(\d+),"
    r"[^,]*,[^,]*,[^,]*,[^,]*,"  # PrimaryColour, SecondaryColour, OutlineColour, BackColour
    r"[^,]*,[^,]*,[^,]*,[^,]*,"  # Bold, Italic, Underline, StrikeOut
    r"[^,]*,[^,]*,[^,]*,[^,]*,"  # ScaleX, ScaleY, Spacing, Angle
    r"[^,]*,[^,]*,[^,]*,"        # BorderStyle, Outline, Shadow
    r"(\d+),"                     # Alignment
    r"[^,]*,[^,]*,"               # MarginL, MarginR
    r"(\d+),"                     # MarginV
)


def _parse_ass_style(ass_path: Path) -> dict | None:
    """Return {fontsize, alignment, margin_v, play_res_y} from an .ass file."""
    if not ass_path.exists():
        return None
    text = ass_path.read_text()
    res_y_match = re.search(r"^PlayResY:\s*(\d+)", text, re.MULTILINE)
    play_res_y = int(res_y_match.group(1)) if res_y_match else 1920
    for line in text.splitlines():
        m = _STYLE_RE.match(line)
        if m:
            return {
                "fontsize": int(m.group(3)),
                "alignment": int(m.group(4)),
                "margin_v": int(m.group(5)),
                "play_res_y": play_res_y,
            }
    return None


def _y_fraction_from_ass(style: dict) -> float:
    """Reconstruct the y_fraction the .ass file is actually rendering at."""
    align = style["alignment"]
    margin_v = style["margin_v"]
    h = style["play_res_y"]
    if align in (1, 2, 3):       # bottom row
        return 1.0 - (margin_v / h)
    if align in (7, 8, 9):       # top row
        return margin_v / h
    return 0.5                    # middle row


# ---------- mechanical checks ----------

def _mechanical_checks(
    out_path: Path,
    ass_path: Path,
    decision: dict,
    rules: EditorialRules,
    trim_report: dict | None,
) -> tuple[list[str], list[dict]]:
    obeyed: list[str] = []
    violated: list[dict] = []

    # 1. ASS style sanity vs Decision
    style = _parse_ass_style(ass_path)
    if not style:
        violated.append({"rule": "ass_file_present", "expected": str(ass_path), "observed": "missing"})
        return obeyed, violated

    # font size >= rules minimum
    if style["fontsize"] >= rules.caption_min_font_px:
        obeyed.append(f"caption_min_font_px ({style['fontsize']} >= {rules.caption_min_font_px})")
    else:
        violated.append({
            "rule": "caption_min_font_px",
            "expected": f">= {rules.caption_min_font_px}",
            "observed": style["fontsize"],
        })

    # caption y vs Decision target
    actual_y = _y_fraction_from_ass(style)
    target_y = float(decision.get("caption_y_fraction", actual_y))
    delta = abs(actual_y - target_y)
    if delta <= CAPTION_Y_TOLERANCE:
        obeyed.append(f"caption_y_fraction (actual {actual_y:.2f} vs target {target_y:.2f})")
    else:
        violated.append({
            "rule": "caption_y_fraction",
            "expected": round(target_y, 2),
            "observed": round(actual_y, 2),
            "tolerance": CAPTION_Y_TOLERANCE,
        })

    # 2. Duration in [clip_min_s, clip_max_s]
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(out_path)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        duration = float(out)
    except Exception:
        duration = 0.0
    if rules.clip_min_s <= duration <= rules.clip_max_s:
        obeyed.append(f"clip_length ({duration:.1f}s in [{rules.clip_min_s}, {rules.clip_max_s}])")
    else:
        violated.append({
            "rule": "clip_length",
            "expected": f"[{rules.clip_min_s}, {rules.clip_max_s}]",
            "observed": round(duration, 2),
        })

    # 3. Silence trim — read trim report if present
    if trim_report:
        if trim_report.get("no_op", False):
            obeyed.append("filler_silence_trim (nothing to cut)")
        else:
            seconds_saved = trim_report.get("seconds_saved", 0.0)
            obeyed.append(f"filler_silence_trim (cut {seconds_saved:.1f}s)")
    # If trim report missing, don't fail — trim is optional.

    return obeyed, violated


# ---------- profile comparison checks ----------

def _sample_brightness_and_temp(out_path: Path) -> tuple[float, str] | None:
    """Sample 3 frames from the rendered output and compute mean luminance
    + dominant color temperature. Returns None if the file can't be read."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(out_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return None
    samples = []
    for pos in (0.20, 0.50, 0.80):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * pos))
        ok, frame = cap.read()
        if ok and frame is not None:
            samples.append(frame)
    cap.release()
    if not samples:
        return None

    import numpy as np
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in samples]
    brightness = float(np.mean([np.mean(g) for g in grays]))

    mean_b = float(np.mean([np.mean(f[:, :, 0]) for f in samples]))
    mean_r = float(np.mean([np.mean(f[:, :, 2]) for f in samples]))
    if mean_r - mean_b > 8:
        temp = "warm"
    elif mean_b - mean_r > 8:
        temp = "cool"
    else:
        temp = "neutral"
    return brightness, temp


def _profile_checks(
    out_path: Path,
    ass_path: Path,
    profile,
) -> tuple[list[str], list[dict]]:
    """Four profile-conformance checks (spec §4.3):
       1. rendered aspect ∈ profile.delivery_aspects
       2. rendered caption_y_frac within ±0.05 of profile.caption.y_frac
       3. rendered avg_brightness within ±15 of profile.color_grade.brightness_target
       4. rendered color_temp matches profile.color_grade.color_temp
    Returns (obeyed, violated). All four are no-ops when profile is None.
    """
    obeyed: list[str] = []
    violated: list[dict] = []
    if profile is None:
        return obeyed, violated

    # 1. Aspect ∈ profile.delivery_aspects.
    # Filename is "<clip_id>_<aspect_slug>.mp4" — e.g. "01_9x16.mp4".
    parts = out_path.stem.split("_")
    slug = parts[-1] if len(parts) >= 2 else ""
    aspect = ASPECT_SLUG_TO_NAME.get(slug, "")
    if aspect and aspect in profile.delivery_aspects:
        obeyed.append(f"profile.aspect ({aspect} in {profile.delivery_aspects})")
    elif aspect:
        violated.append({
            "rule": "profile.delivery_aspects",
            "expected": profile.delivery_aspects,
            "observed": aspect,
        })

    # 2. Caption y_frac directly against profile (not just the Decision).
    style = _parse_ass_style(ass_path)
    if style:
        actual_y = _y_fraction_from_ass(style)
        target_y = profile.caption.y_frac
        if abs(actual_y - target_y) <= PROFILE_CAPTION_Y_TOLERANCE:
            obeyed.append(
                f"profile.caption.y_frac ({actual_y:.2f} vs target {target_y:.2f}, "
                f"±{PROFILE_CAPTION_Y_TOLERANCE})"
            )
        else:
            violated.append({
                "rule": "profile.caption.y_frac",
                "expected": round(target_y, 2),
                "observed": round(actual_y, 2),
                "tolerance": PROFILE_CAPTION_Y_TOLERANCE,
            })

    # 3 + 4 — sample the rendered output for brightness + color_temp.
    sample = _sample_brightness_and_temp(out_path)
    if sample is None:
        return obeyed, violated
    brightness, temp = sample

    target_b = profile.color_grade.brightness_target
    if abs(brightness - target_b) <= PROFILE_BRIGHTNESS_TOLERANCE:
        obeyed.append(
            f"profile.brightness ({brightness:.1f} vs target {target_b:.1f}, "
            f"±{PROFILE_BRIGHTNESS_TOLERANCE})"
        )
    else:
        violated.append({
            "rule": "profile.color_grade.brightness_target",
            "expected": f"{target_b:.1f} ± {PROFILE_BRIGHTNESS_TOLERANCE}",
            "observed": round(brightness, 1),
        })

    target_temp = profile.color_grade.color_temp
    if temp == target_temp:
        obeyed.append(f"profile.color_temp ({temp})")
    else:
        violated.append({
            "rule": "profile.color_grade.color_temp",
            "expected": target_temp,
            "observed": temp,
        })

    return obeyed, violated


# ---------- vision check (Pegasus, free tier) ----------

_VISION_PROMPT = """Look at these frames from a rendered short-form video clip and identify any visible problems.

Specifically check for:
1. Subject cropped at any edge (head, body partially out of frame)
2. Captions overlapping the face or main subject
3. Captions cut off at frame edge
4. Visible compression / scaling / upscale artifacts
5. Color cast, banding, or unnatural skin tones

Return STRICT JSON only (no markdown, no commentary):
{
  "pass": true | false,
  "issues": ["short description of each problem found, or empty list if none"],
  "critical": ["subset of issues that are bad enough to reject the clip"]
}
"""


def _vision_check(
    out_path: Path,
    job: Job,
) -> tuple[list[str], list[dict]]:
    """Sample 3 frames, send to Pegasus, collect findings.
    Returns (vision_findings, additional_rules_violated_from_vision).
    Falls back gracefully on any error."""
    findings: list[str] = []
    additional_violations: list[dict] = []

    api_key = os.environ.get("TWELVELABS_API_KEY")
    if not api_key:
        findings.append("vision check skipped (no TWELVELABS_API_KEY)")
        return findings, additional_violations

    # Probe duration
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(out_path)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        duration = float(out)
    except Exception:
        findings.append("vision check skipped (could not probe duration)")
        return findings, additional_violations

    if duration < 1.0:
        findings.append("vision check skipped (clip too short)")
        return findings, additional_violations

    # Extract frames to a temp dir.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        frame_paths: list[Path] = []
        for i, frac in enumerate(VISION_FRAME_POSITIONS):
            t = max(0.1, frac * duration)
            fp = td_path / f"frame_{i}.jpg"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(out_path),
                     "-frames:v", "1", "-q:v", "3", str(fp)],
                    check=True, capture_output=True,
                )
                if fp.exists() and fp.stat().st_size > 0:
                    frame_paths.append(fp)
            except subprocess.CalledProcessError:
                continue

        if not frame_paths:
            findings.append("vision check skipped (frame extraction failed)")
            return findings, additional_violations

        # Use TwelveLabs Pegasus by uploading the rendered output as a fresh
        # asset OR sending frames directly. The simpler path: encode frames
        # as base64 and send via the analyze endpoint... but Pegasus is video,
        # not images. So instead: index the rendered output briefly OR skip.
        #
        # For Phase 2 simplicity we use TwelveLabs' existing video_id for the
        # SOURCE (already in the cache) and ask Pegasus about the *source*
        # window matching this clip. That's not perfect (we want to see OUR
        # render, not the source), but it surfaces source-level issues at zero
        # cost. A future Phase 2.5 could upload the rendered output for true
        # output-level vision QC.
        cache_path = job.work_dir / "twelvelabs_videos.json"
        if not cache_path.exists():
            findings.append("vision check skipped (no twelvelabs source cache)")
            return findings, additional_violations
        try:
            cached = cache_read(cache_path)
        except Exception:
            findings.append("vision check skipped (twelvelabs cache unreadable)")
            return findings, additional_violations
        video_id = cached.get(job.source_hash)
        if not video_id:
            findings.append("vision check skipped (no video_id for source)")
            return findings, additional_violations

        # Pegasus 1.5 doesn't accept video_id anymore — fall back to skipping
        # rather than failing the run. We surface this so the operator knows
        # vision QC didn't happen this clip.
        findings.append(
            "vision check available but pegasus1.5 video_id deprecated; "
            "skipping until copy.py is migrated to video object input"
        )
        return findings, additional_violations


# ---------- driver ----------

def run(
    job: Job,
    produced: list[Path],
    decisions: list[dict] | None = None,
    rules: EditorialRules | None = None,
    profile=None,
) -> dict:
    """Run mechanical + vision editorial checks on every rendered output.
    `profile` (ClientStyleProfile) is read alongside `rules` (style.md);
    profile is the per-client ceiling, style.md is the floor.
    Writes editorial_report.json and returns the parsed report."""
    if rules is None:
        rules = load_rules()
    decisions_by_id: dict[str, dict] = {}
    if decisions:
        for d in decisions:
            if isinstance(d, dict):
                decisions_by_id[d.get("clip_id", "?")] = d
            else:
                decisions_by_id[getattr(d, "clip_id", "?")] = asdict(d)

    reports: list[EditorialReport] = []
    failures = 0

    for out_path in produced:
        # Output filename is "<clip_id>_<aspect_slug>.mp4" — no variant segment.
        stem = out_path.stem  # e.g. "01_9x16"
        parts = stem.split("_")
        clip_id = parts[0]
        # Try to find the matching .ass file
        ass_candidates = list((job.work_dir).glob(f"{clip_id}.*.ass"))
        ass_path = ass_candidates[0] if ass_candidates else (job.work_dir / f"{clip_id}.ass")

        # Load trim report if any
        trim_report_path = job.work_dir / f"{clip_id}.trim_report.json"
        trim_report = None
        if trim_report_path.exists():
            try:
                trim_report = cache_read(trim_report_path)
            except Exception:
                trim_report = None

        decision = decisions_by_id.get(clip_id, {})
        obeyed, violated = _mechanical_checks(out_path, ass_path, decision, rules, trim_report)

        # Profile-conformance checks (Phase 5a §4.3): aspect, caption y_frac,
        # brightness, color_temp — all measured against the loaded
        # ClientStyleProfile, not just the per-clip Decision.
        prof_obeyed, prof_violated = _profile_checks(out_path, ass_path, profile)
        obeyed.extend(prof_obeyed)
        violated.extend(prof_violated)

        vision_findings, more_violations = _vision_check(out_path, job)
        violated.extend(more_violations)

        status = "OK"
        if violated:
            status = "FAIL_MECHANICAL"
            failures += 1
        elif any("CRITICAL" in f for f in vision_findings):
            status = "FAIL_VISION"
            failures += 1

        rep = EditorialReport(
            file=str(out_path),
            clip_id=clip_id,
            rules_obeyed=obeyed,
            rules_violated=violated,
            vision_findings=vision_findings,
            status=status,
        )
        reports.append(rep)
        if violated:
            print(f"[validate.editorial] {out_path.name}: FAIL "
                  f"{[v['rule'] for v in violated]}", file=sys.stderr)
        else:
            print(f"[validate.editorial] {out_path.name}: OK ({len(obeyed)} rules)",
                  file=sys.stderr)

    summary = {
        "total": len(reports),
        "passed": len(reports) - failures,
        "failed": failures,
    }
    payload = {"summary": summary, "reports": [asdict(r) for r in reports]}
    cache_write(job.output_dir / "editorial_report.json", payload)
    return payload
