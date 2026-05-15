#!/usr/bin/env python3
"""cut.py — orchestrator for the client-style clip pipeline.

Style is learned from references; this script is a delivery mechanism.

Pipeline:
    1.   whisper.transcribe         (full-video word transcription)
    3.   detect.find_candidates     (TwelveLabs Pegasus / local backend → graded.json)
    4.   top-N selection
    4.5. framing.analyze            (face + HOG + motion)
    4.6. profile.load(--client)     (ClientStyleProfile from references/<client>/profile.json)
    4.7. rag.retrieve               (per-clip retrieved references, when --client passed)
    4.8. editorial.decisions        (per-clip render parameters, grounded in profile + refs)
    5.   render.render_all          (one MP4 per clip × profile.delivery_aspects)
    6.   verify.run + editorial.run (technical + editorial validation)

Run:
    python cut.py jobs/myjob.json --client fitness_creator_v1 --source video.mp4

Details: docs/superpowers/specs/2026-05-15-phase5a-client-style-profile-design.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clip_pipeline import config as cfg
from clip_pipeline import profile as profile_mod
from clip_pipeline import source
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
        help="Client name. Resolves to references/<client>/profile.json — the "
             "ClientStyleProfile that drives caption / aspect / color / upscaling. "
             "REQUIRED unless --quickstart is used.",
    )
    ap.add_argument(
        "--quickstart",
        action="store_true",
        help="Explicit escape hatch: render with floor defaults from "
             "references/_global/style.md (one 9:16 output per clip, neutral grade, "
             "no upscaling). Use for dev/demo runs with no client references yet.",
    )
    args = ap.parse_args()

    cfg.load_env()

    job = cfg.load_job(args.job)

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

    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.work_dir.mkdir(parents=True, exist_ok=True)

    if not args.client and not args.quickstart:
        sys.exit(
            "no --client supplied. The pipeline is reference-conditioned: "
            "every run targets a specific client whose references define the "
            "style. Either:\n"
            "  • pass --client <name> (requires references/<name>/ already "
            "ingested via `python -m clip_pipeline.rag.ingest --client <name>`)\n"
            "  • or pass --quickstart to render with floor defaults from "
            "references/_global/style.md (dev/demo only)."
        )
    if args.client and args.quickstart:
        sys.exit("--client and --quickstart are mutually exclusive.")

    if args.client:
        profile = profile_mod.load_strict(args.client)
    else:
        profile = profile_mod.load(None)
        profile.client = "_quickstart"

    backend = cfg.resolve_detection_backend(job.detection_backend)
    print(f"[cut] backend={backend}, top_n={job.top_n}, "
          f"client={profile.client}, aspects={profile.delivery_aspects}, "
          f"output_dir={job.output_dir}", file=sys.stderr)

    # 0. Acquire source + detect its shape.
    src_path = source.ensure_source(job)
    source_profile = cfg.detect_source_profile(src_path)
    print(f"[cut] source: {src_path} (profile={source_profile})", file=sys.stderr)

    # 1. Whisper full-video transcription.
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

    # 4.5. Framing + motion.
    framings = framing.analyze(job, src_path, selected, source_profile)

    # 4.7. RAG retrieve — per-clip references for grounding editorial decisions.
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
                print(f"[rag] {c.id}: retrieved {len(refs)} reference(s) for "
                      f"client={args.client}", file=sys.stderr)
        if not references_by_clip:
            print(f"[rag] client={args.client} has no indexed references; "
                  f"editorial decisions fall back to profile + global rules",
                  file=sys.stderr)

    # 4.8. Editorial decisions, grounded in profile + retrieved refs.
    decisions = editorial_decisions.decide_all(
        job, selected, framings,
        references_by_clip=references_by_clip or None,
        profile=profile,
    )
    recs = editorial_decisions.as_dict(decisions)

    # 5. Render: one MP4 per clip × profile.delivery_aspects.
    produced = render.render_all(job, src_path, words, selected, framings,
                                 profile, recs)
    print(f"[cut] rendered {len(produced)} MP4s", file=sys.stderr)

    # 5.7. Social post copy — gated on profile.generate_post_copy.
    if profile.generate_post_copy:
        posts = post_copy.generate_all(job, selected)
        print(f"[cut] post copy generated for {len(posts)} clip(s)",
              file=sys.stderr)

    # 6a. Technical verification.
    report = verify.run(job, produced)
    print(f"[cut] verify (technical): {report['summary']}", file=sys.stderr)

    # 6b. Editorial verification against profile + global rules.
    editorial_report = editorial_validate.run(job, produced, decisions,
                                              profile=profile)
    print(f"[cut] verify (editorial): {editorial_report['summary']}",
          file=sys.stderr)

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
