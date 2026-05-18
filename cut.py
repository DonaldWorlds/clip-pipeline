#!/usr/bin/env python3
"""cut.py — orchestrator for the clip pipeline.

Simplest usage (no setup required):
    python cut.py /path/to/your-video.mp4

Power-user usage (per-client style learned from references):
    python cut.py /path/to/your-video.mp4 --client fitness_creator
    python cut.py jobs/myjob.json   # legacy: job file still works

Style is learned from references; this script is a delivery mechanism.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from clip_pipeline import config as cfg
from clip_pipeline import profile as profile_mod
from clip_pipeline import source
from clip_pipeline.analyze import detect, framing, moment_intelligence, whisper
from clip_pipeline.editorial import decisions as editorial_decisions
from clip_pipeline.rag import retrieve as rag_retrieve
from clip_pipeline.render import pool as render
from clip_pipeline.validate import editorial as editorial_validate
from clip_pipeline.validate import technical as verify


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


# ============================================================
# Progress indicator
# ============================================================

@contextlib.contextmanager
def _stage(name: str, idx: int, total: int):
    """Wrap a pipeline stage with a labeled start/duration line.
    Inner logs from the stage (whisper progress, ffmpeg, etc) still print
    between the start and end markers — they're not suppressed, just framed.
    """
    print(f"[{idx}/{total}] {name}...", file=sys.stderr, flush=True)
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[{idx}/{total}] {name} done in {dt:.1f}s", file=sys.stderr, flush=True)


# ============================================================
# Source probe — duration, quality, recommended top_n
# ============================================================

def _probe_source(src_path: Path) -> dict:
    """ffprobe the source: return dict with duration_s, video specs, audio specs,
    and a recommended top_n based on duration. Prints warnings for substandard
    source quality. Returns {} if probe fails."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(src_path)],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
    except Exception as e:
        print(f"[source] quality probe failed: {e}", file=sys.stderr)
        return {}

    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})

    total_kbps = int(fmt.get("bit_rate", 0)) // 1000
    duration_s = float(fmt.get("duration") or 0)

    info: dict = {"duration_s": duration_s, "total_kbps": total_kbps}

    if v:
        w = int(v.get("width", 0))
        h = int(v.get("height", 0))
        codec = v.get("codec_name", "?")
        fps_str = v.get("r_frame_rate", "0/1")
        try:
            num, den = fps_str.split("/")
            fps = round(float(num) / float(den), 1) if float(den) else 0
        except (ValueError, ZeroDivisionError):
            fps = 0
        info["video"] = {"w": w, "h": h, "codec": codec, "fps": fps}
        print(f"[source] video: {w}x{h} {codec} @ {fps}fps, "
              f"total bitrate ~{total_kbps}kbps, duration {duration_s:.1f}s",
              file=sys.stderr)
        if h and h < 720:
            print(f"[source] WARNING: source is {h}p (< 720p). Output quality "
                  f"will be limited by source resolution.", file=sys.stderr)
        if total_kbps and total_kbps < 2000:
            print(f"[source] WARNING: total bitrate {total_kbps}kbps is below "
                  f"2000kbps — likely visible compression artifacts.", file=sys.stderr)

    if a:
        codec = a.get("codec_name", "?")
        sr = a.get("sample_rate", "?")
        ch = a.get("channels", 0)
        a_kbps = int(a.get("bit_rate", 0)) // 1000
        info["audio"] = {"codec": codec, "sample_rate": sr, "channels": ch, "kbps": a_kbps}
        print(f"[source] audio: {codec} {ch}ch @ {sr}Hz, ~{a_kbps}kbps",
              file=sys.stderr)
        if a_kbps and a_kbps < 96:
            print(f"[source] WARNING: audio bitrate {a_kbps}kbps is below "
                  f"96kbps — captions will be fine but audio fidelity is low.",
                  file=sys.stderr)

    # Duration recommendations
    if duration_s > 0:
        if duration_s < 30:
            print(f"[source] very short source ({duration_s:.0f}s). At most 1 useful "
                  f"clip possible — recommend longer source (2-5 min ideal).",
                  file=sys.stderr)
        elif duration_s > 900:
            print(f"[source] long source ({duration_s/60:.1f} min). Render will be slow. "
                  f"Recommended sweet spot is 2-5 min.", file=sys.stderr)
        elif duration_s < 60:
            print(f"[source] short source ({duration_s:.0f}s). Sweet spot is 2-5 min.",
                  file=sys.stderr)

    info["recommended_top_n"] = _smart_top_n(duration_s)
    return info


def _smart_top_n(duration_s: float) -> int:
    """Auto-pick top_n based on source duration. Bounded 1..10.
       <60s  → 1 clip
       60-180s → 3 clips
       180-300s → 5 clips
       300-600s → 7 clips
       >600s → 10 clips
    """
    if duration_s <= 0:
        return 3
    minutes = duration_s / 60.0
    if minutes < 1:
        return 1
    if minutes < 3:
        return 3
    if minutes < 5:
        return 5
    if minutes < 10:
        return 7
    return 10


# ============================================================
# Job construction
# ============================================================

def _build_job_from_video(video_path: Path) -> cfg.Job:
    """Auto-build a Job dataclass for a direct video file input. Used when
    the user runs `cut.py video.mp4` instead of `cut.py jobs/foo.json`.
    output_dir is derived from the video's stem so multiple sources don't
    collide in one folder."""
    return cfg.Job(
        source=str(video_path.expanduser().resolve()),
        source_type="local",
        output_dir=Path("out") / video_path.stem,
        top_n=3,  # will be overridden later by smart_top_n or --top-n flag
        detection_backend="auto",
        whisper_model="small",
    )


def _select_top_n(cands, n: int):
    """Pick the top-N candidates by final_score, then renumber them
    sequentially (01, 02, 03, ...) for clean output filenames.

    The original Marengo-rank IDs are preserved in `graded_reranked.json`;
    the renumber mapping is logged so cross-referencing is one scroll away.
    """
    sorted_cs = sorted(cands, key=lambda c: -c.score)
    top = sorted_cs[: max(1, n)]
    if top:
        print("[select] renumbering top-N to sequential output IDs:", file=sys.stderr)
    for i, c in enumerate(top, 1):
        original_id = c.id
        c.id = f"{i:02d}"
        print(f"        {c.id} <- (was {original_id})  "
              f"[{c.start:.1f}-{c.end:.1f}]  score={c.score:.4f}  "
              f"rationale={c.rationale!r}", file=sys.stderr)
    return top


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Clip pipeline: long-form video → short-form clips.",
    )
    ap.add_argument(
        "source_or_job",
        type=Path,
        help="EITHER a path to a video file (.mp4/.mov/etc) for the simplest "
             "usage, OR a path to a job JSON for power-user / multi-source workflows.",
    )
    ap.add_argument(
        "--source",
        help="Override the source from the job file (only meaningful when "
             "source_or_job is a job JSON).",
    )
    ap.add_argument(
        "--source-type",
        choices=["local", "youtube"],
        help="Force the source type (otherwise inferred from --source prefix).",
    )
    ap.add_argument(
        "--detection-backend",
        choices=["auto", "twelvelabs", "local"],
        help="Override job['detection_backend'].",
    )
    ap.add_argument(
        "--no-twelvelabs",
        action="store_true",
        help="Skip TwelveLabs entirely — use local-only moment detection "
             "(transcript-density based). Useful when the user can't or "
             "doesn't want to upload footage to a third party.",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        help="How many clips to produce. Valid range 1..10. If omitted, "
             "auto-picked from source duration (1min→1, 3min→3, 5min→5, "
             "10min→7, >10min→10).",
    )
    ap.add_argument("--keep-temp", action="store_true",
                    help="Retain .work/ intermediates after success.")
    ap.add_argument(
        "--client",
        help="Client name. Resolves to references/<client>/profile.json — the "
             "ClientStyleProfile that drives caption / aspect / color / upscaling. "
             "If omitted, runs in QUICKSTART mode (floor defaults from style.md).",
    )
    ap.add_argument(
        "--quickstart",
        action="store_true",
        help="Explicit no-client mode (same as omitting --client). Kept as a "
             "flag for backward compatibility.",
    )
    ap.add_argument(
        "--no-moment-intelligence",
        action="store_true",
        help="Skip the RAG-driven candidate re-rank step. Pegasus's "
             "original ordering decides top-N.",
    )
    args = ap.parse_args()

    cfg.load_env()

    # --- 0a. Resolve input: video file or job JSON ---
    in_path = args.source_or_job
    if not in_path.exists():
        sys.exit(f"input not found: {in_path}")
    if in_path.suffix.lower() == ".json":
        job = cfg.load_job(in_path)
        if args.source:
            job.source = args.source
            job.source_type = args.source_type or (
                "youtube" if args.source.startswith(("http://", "https://"))
                else "local"
            )
    elif in_path.suffix.lower() in VIDEO_SUFFIXES:
        job = _build_job_from_video(in_path)
        job_auto_built = True
        if args.source:
            print(f"[cut] WARNING: --source ignored when input is a video file",
                  file=sys.stderr)
    else:
        sys.exit(f"unknown input type {in_path.suffix!r}; expected .json or a "
                 f"video file ({', '.join(sorted(VIDEO_SUFFIXES))})")
    job_auto_built = locals().get("job_auto_built", False)

    if args.detection_backend:
        job.detection_backend = args.detection_backend
    if args.no_twelvelabs:
        job.detection_backend = "local"

    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.work_dir.mkdir(parents=True, exist_ok=True)

    # --- 0b. Client / quickstart mode ---
    if args.client and args.quickstart:
        sys.exit("--client and --quickstart are mutually exclusive.")

    if args.client:
        profile = profile_mod.load_strict(args.client)
        client_label = args.client
    else:
        profile = profile_mod.load(None)
        profile.client = "_quickstart"
        client_label = "_quickstart"
        print(f"[cut] running in QUICKSTART mode (no --client passed). "
              f"Using floor defaults from references/_global/style.md. "
              f"To use a custom client style: ingest references and pass --client <name>.",
              file=sys.stderr)

    # --- 0c. Per-source delivery folder ---
    # Path shape: out/<job.output_dir>/<source_stem>/run_<timestamp>/
    # So multiple source videos under the same client don't collide, and
    # multiple runs of the same source are preserved side-by-side.
    src_path_for_naming = Path(job.source).expanduser()
    source_stem = (
        src_path_for_naming.stem
        if src_path_for_naming.suffix.lower() in VIDEO_SUFFIXES
        else "remote"
    )
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Skip the source_stem inner folder when the job was AUTO-built from a
    # video path — the source_stem is already in job.output_dir (`out/<stem>`),
    # nesting it again produces `out/<stem>/<stem>/run_<ts>` which was the
    # double-nesting bug. When the job came from a JSON file, output_dir is
    # client-named (`out/phone_creator`) and we DO want source_stem nesting
    # so multiple sources don't collide.
    if job_auto_built:
        job._delivery_dir = job.output_dir / f"run_{run_id}"
    else:
        job._delivery_dir = job.output_dir / source_stem / f"run_{run_id}"
    job.delivery_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cut] delivery dir: {job.delivery_dir}", file=sys.stderr)

    # --- 0d. Acquire + probe source ---
    src_path = source.ensure_source(job)
    source_profile = cfg.detect_source_profile(src_path)
    print(f"[cut] source: {src_path} (profile={source_profile})", file=sys.stderr)
    src_info = _probe_source(src_path)

    # --- 0e. Smart top_n if not explicitly set ---
    if args.top_n is not None:
        if not (1 <= args.top_n <= 10):
            sys.exit(f"--top-n must be in 1..10 (got {args.top_n})")
        job.top_n = args.top_n
        print(f"[cut] top_n = {job.top_n} (from --top-n)", file=sys.stderr)
    else:
        job.top_n = src_info.get("recommended_top_n", job.top_n or 3)
        print(f"[cut] top_n = {job.top_n} (auto from "
              f"{src_info.get('duration_s', 0):.0f}s source duration)",
              file=sys.stderr)

    backend = cfg.resolve_detection_backend(job.detection_backend)
    print(f"[cut] backend={backend}, client={client_label}, "
          f"aspects={profile.delivery_aspects}",
          file=sys.stderr)

    # ---- 8-stage pipeline begins ----
    TOTAL_STAGES = 8

    # 1. Whisper full-video transcription.
    with _stage("whisper transcription", 1, TOTAL_STAGES):
        words = whisper.transcribe(job, src_path)
        print(f"        extracted {len(words)} words", file=sys.stderr)

    # 2. Detection + grading.
    with _stage("candidate detection", 2, TOTAL_STAGES):
        cands = detect.find_candidates(job, backend, src_path, words)
        print(f"        {len(cands)} candidates", file=sys.stderr)
    if not cands:
        sys.exit("no candidates produced; refusing to render")

    # 3. Moment Intelligence re-rank.
    with _stage("moment intelligence re-rank", 3, TOTAL_STAGES):
        if args.client and not args.no_moment_intelligence:
            cands = moment_intelligence.rerank(
                job, cands, words, args.client, profile,
            )
        else:
            reason = "no --client" if not args.client else "--no-moment-intelligence"
            print(f"        skipped ({reason})", file=sys.stderr)

    # 4. Top-N selection.
    with _stage("top-N selection", 4, TOTAL_STAGES):
        selected = _select_top_n(cands, job.top_n)
        cfg.cache_write(job.work_dir / "selected.json", selected)

    # 5. Framing + motion.
    with _stage("framing analysis", 5, TOTAL_STAGES):
        framings = framing.analyze(job, src_path, selected, source_profile)

    # 6. Editorial decisions.
    with _stage("editorial decisions", 6, TOTAL_STAGES):
        references_by_clip: dict[str, list] = {}
        if args.client:
            for c in selected:
                try:
                    refs = rag_retrieve.retrieve(
                        client=args.client,
                        niche=None,
                        k=5,
                        query_transcript=c.rationale or c.topic,
                        query_description=c.rationale or c.topic,
                    )
                except Exception as e:
                    print(f"[rag] retrieve failed for {c.id}: {e}", file=sys.stderr)
                    refs = []
                if refs:
                    references_by_clip[c.id] = refs
        decisions = editorial_decisions.decide_all(
            job, selected, framings,
            references_by_clip=references_by_clip or None,
            profile=profile,
        )
        recs = editorial_decisions.as_dict(decisions)

    # 7. Render.
    with _stage(f"render ({len(selected)} clips × {len(profile.delivery_aspects)} aspects)",
                7, TOTAL_STAGES):
        produced = render.render_all(job, src_path, words, selected, framings,
                                     profile, recs)
        print(f"        produced {len(produced)} MP4s", file=sys.stderr)

    # 8. Verification.
    with _stage("verification (technical + editorial)", 8, TOTAL_STAGES):
        report = verify.run(job, produced)
        print(f"        technical: {report['summary']}", file=sys.stderr)
        editorial_report = editorial_validate.run(job, produced, decisions,
                                                  profile=profile)
        print(f"        editorial: {editorial_report['summary']}", file=sys.stderr)

    has_failures = bool(report["failures"]) or editorial_report["summary"]["failed"] > 0

    if not args.keep_temp and not has_failures:
        import shutil
        shutil.rmtree(job.work_dir, ignore_errors=True)

    if has_failures:
        msgs = []
        if report["failures"]:
            msgs.append(f"technical failures: {len(report['failures'])}")
        if editorial_report["summary"]["failed"]:
            msgs.append(f"editorial failures: {editorial_report['summary']['failed']}")
        sys.exit(
            f"verification failed ({'; '.join(msgs)}); "
            f"see {job.delivery_dir}/report.json and editorial_report.json"
        )

    print(f"\n[cut] done. {len(produced)} clips in {job.delivery_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
