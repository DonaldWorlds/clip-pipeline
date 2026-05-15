#!/usr/bin/env python3
"""cut.py — v2 orchestrator for the clip pipeline.

Pipeline:
    1.   whisper.transcribe         (full-video word transcription)
    3.   detect.find_candidates     (pluggable backend → graded.json)
    4.   top-N selection
    4.5. framing.analyze            (face + HOG + motion)
    5.   render.render_all          (multi-aspect, optional zoom)
    6.   verify.run                 (ffprobe + still-frame check)

Run:
    python cut.py jobs/myjob.json --source /path/to/video.mp4

See docs/superpowers/specs/2026-05-13-clip-pipeline-v2-design.md for details.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clip_pipeline import config as cfg
from clip_pipeline import (
    copy as post_copy,
    detect,
    framing,
    intelligence,
    render,
    source,
    thumbnail,
    verify,
    whisper,
)


def _select_top_n(cands, n: int):
    sorted_cs = sorted(cands, key=lambda c: -c.score)
    return sorted_cs[: max(1, n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", type=Path, help="Path to job JSON")
    ap.add_argument("--variants", type=Path, default=cfg.DEFAULT_VARIANTS_PATH)
    ap.add_argument(
        "--source",
        help="Override job['source']. Local path or URL. Type auto-detected from prefix.",
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
    ap.add_argument("--top-n", type=int, help="Override job['top_n'].")
    ap.add_argument("--keep-temp", action="store_true",
                    help="Retain .work/ intermediates after success.")
    args = ap.parse_args()

    cfg.load_env()

    job = cfg.load_job(args.job)
    variants = cfg.load_variants(args.variants)

    if args.source:
        job.source = args.source
        if args.source_type:
            job.source_type = args.source_type
        else:
            job.source_type = (
                "youtube" if args.source.startswith(("http://", "https://"))
                else "local"
            )
    elif args.source_type:
        sys.exit("--source-type requires --source")

    if args.detection_backend:
        job.detection_backend = args.detection_backend
    if args.top_n:
        job.top_n = args.top_n

    missing = [v for v in job.variants if v not in variants]
    if missing:
        sys.exit(f"unknown variants: {missing}")

    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.work_dir.mkdir(parents=True, exist_ok=True)

    backend = cfg.resolve_detection_backend(job.detection_backend)
    print(f"[cut] backend={backend}, top_n={job.top_n}, "
          f"output_dir={job.output_dir}", file=sys.stderr)

    # 0. Acquire source + detect its shape.
    src_path = source.ensure_source(job)
    source_profile = cfg.detect_source_profile(src_path)
    print(f"[cut] source: {src_path} (profile={source_profile})",
          file=sys.stderr)

    # 1. Whisper full-video transcription (always — needed for captions and
    #    sentence-boundary snapping regardless of detection backend).
    words = whisper.transcribe(job, src_path)
    print(f"[cut] whisper: {len(words)} words", file=sys.stderr)

    # 3. Detection + grading.
    cands = detect.find_candidates(job, backend, src_path, words)
    print(f"[cut] {len(cands)} candidates", file=sys.stderr)
    if not cands:
        sys.exit("no candidates produced; refusing to render")

    # 4. Top-N selection.
    selected = _select_top_n(cands, job.top_n)
    cfg.cache_write(job.work_dir / "selected.json", selected)
    print(f"[cut] selected {len(selected)} clips:", file=sys.stderr)
    for c in selected:
        print(f"        {c.id}: [{c.start:.1f}-{c.end:.1f}] "
              f"score={c.score:.2f} {c.rationale}", file=sys.stderr)

    # 4.5. Framing + motion. source_profile lets phone_vertical content skip
    # the streamer-overlay composite path entirely.
    framings = framing.analyze(job, src_path, selected, source_profile)

    # 4.75. Variant Intelligence: per-clip render parameter decisions.
    recs = intelligence.recommend(job, selected, framings)

    # 5. Render every (clip × variant × aspect), parameterized by intelligence.
    produced = render.render_all(job, src_path, words, selected, framings,
                                 variants, recs)
    print(f"[cut] rendered {len(produced)} MP4s", file=sys.stderr)

    # 5.5. Thumbnails — one JPG per clip. Pass framings so overlay-class clips
    # get cropped to the speaker_rect (a thumbnail of the full screen recording
    # is useless for streamer content).
    thumbs = thumbnail.extract_all(job, src_path, selected, framings)
    print(f"[cut] thumbnails: {len(thumbs)}", file=sys.stderr)

    # 5.7. Social post copy — TikTok / IG / YT captions + hashtags per clip.
    posts = post_copy.generate_all(job, selected)
    print(f"[cut] post copy generated for {len(posts)} clip(s)", file=sys.stderr)

    # 6. Verification.
    report = verify.run(job, produced)
    print(f"[cut] verify: {report['summary']}", file=sys.stderr)

    if not args.keep_temp and not report["failures"]:
        import shutil
        shutil.rmtree(job.work_dir, ignore_errors=True)

    if report["failures"]:
        sys.exit(f"verification failed for {len(report['failures'])} output(s); "
                 f"see {job.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
