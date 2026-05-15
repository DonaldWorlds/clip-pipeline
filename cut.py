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
from clip_pipeline import source, thumbnail
from clip_pipeline.analyze import detect, framing, whisper
from clip_pipeline.editorial import copy as post_copy
from clip_pipeline.editorial import decisions as editorial_decisions
from clip_pipeline.rag import retrieve as rag_retrieve
from clip_pipeline.render import pool as render
from clip_pipeline.validate import editorial as editorial_validate
from clip_pipeline.validate import technical as verify


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
    ap.add_argument(
        "--client",
        help="(Phase 3) Client name. When set, the RAG Librarian retrieves "
             "indexed reference clips for this client and grounds editorial "
             "decisions in them. Falls back to global rulebook if client has "
             "no references yet.",
    )
    ap.add_argument(
        "--verify-before-render",
        action="store_true",
        help="(Phase 2) Vision sanity-check each clip's decision BEFORE rendering. "
             "Saves time on clips that would fail editorial validation anyway. "
             "Pegasus free tier. Off by default.",
    )
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

    # 4.6. RAG retrieve (Phase 3) — if --client passed AND that client has
    # indexed references, retrieve top-K per clip to ground decisions.
    references_by_clip: dict[str, list] = {}
    if args.client:
        for c in selected:
            # Query: the candidate's rationale (Pegasus "why it works") + its
            # transcript snippet. retrieve.py will embed both client-side.
            try:
                refs = rag_retrieve.retrieve(
                    client=args.client,
                    niche=None,                  # niche-agnostic for now
                    k=5,
                    query_transcript=c.rationale or c.topic,
                    query_description=c.rationale or c.topic,
                )
            except Exception as e:
                print(f"[rag] retrieve failed for {c.id}: {e}", file=sys.stderr)
                refs = []
            if refs:
                references_by_clip[c.id] = refs
                print(f"[rag] {c.id}: retrieved {len(refs)} reference(s) for "
                      f"client={args.client}", file=sys.stderr)
        if not references_by_clip:
            print(f"[rag] client={args.client} has no indexed references; "
                  f"falling back to global rules only", file=sys.stderr)

    # 4.75. Editorial decisions: per-clip render parameter decisions, grounded
    # in references/_global/style.md (loaded by editorial.rules) and optionally
    # in retrieved reference clips.
    decisions = editorial_decisions.decide_all(
        job, selected, framings,
        references_by_clip=references_by_clip or None,
    )
    recs = editorial_decisions.as_dict(decisions)

    # 5. Render every (clip × variant × aspect), parameterized by decisions.
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

    # 6a. Technical verification (ffprobe + still-frame).
    report = verify.run(job, produced)
    print(f"[cut] verify (technical): {report['summary']}", file=sys.stderr)

    # 6b. Editorial verification (rules-compliance + Pegasus vision QC).
    editorial_report = editorial_validate.run(job, produced, decisions)
    print(f"[cut] verify (editorial): {editorial_report['summary']}", file=sys.stderr)

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
            f"see {job.output_dir}/report.json and editorial_report.json"
        )


if __name__ == "__main__":
    main()
