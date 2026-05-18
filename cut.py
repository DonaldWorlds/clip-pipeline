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
from clip_pipeline.analyze import detect, framing, moment_intelligence, whisper
from clip_pipeline.editorial import decisions as editorial_decisions
from clip_pipeline.rag import retrieve as rag_retrieve
from clip_pipeline.render import pool as render
from clip_pipeline.validate import editorial as editorial_validate
from clip_pipeline.validate import technical as verify


def _probe_source_quality(src_path: Path) -> None:
    """Print source video/audio specs at startup. Warns if the source is
    below thresholds that limit output quality (resolution, bitrate, audio
    spec). Diagnostic only — does not refuse to render. Helps the operator
    set client expectations: low-quality input → limits on output quality."""
    import subprocess
    import json as _json
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(src_path)],
            capture_output=True, text=True, check=True,
        )
        data = _json.loads(result.stdout)
    except Exception as e:
        print(f"[source] quality probe failed: {e}", file=sys.stderr)
        return

    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})
    total_kbps = int(fmt.get("bit_rate", 0)) // 1000

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
        print(f"[source] video: {w}x{h} {codec} @ {fps}fps, "
              f"total bitrate ~{total_kbps}kbps", file=sys.stderr)
        if h < 720:
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
        print(f"[source] audio: {codec} {ch}ch @ {sr}Hz, ~{a_kbps}kbps",
              file=sys.stderr)
        if a_kbps and a_kbps < 96:
            print(f"[source] WARNING: audio bitrate {a_kbps}kbps is below "
                  f"96kbps — captions will be fine but audio fidelity is low.",
                  file=sys.stderr)


def _select_top_n(cands, n: int):
    """Pick the top-N candidates by final_score, then renumber them
    sequentially (01, 02, 03, ...) for clean output filenames.

    The original Marengo-rank IDs (e.g. "07" if Marengo's 7th-ranked moment
    won the rerank) are still preserved in `graded_reranked.json` keyed by
    their original IDs. The renumber-to-sequential mapping is logged to
    stderr so cross-referencing is one log scroll away."""
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
    ap.add_argument(
        "--no-moment-intelligence",
        action="store_true",
        help="(Phase 5b) Skip the RAG-driven candidate re-rank step. Pegasus's "
             "original ordering decides top-N. Useful for A/B comparing reference-"
             "conditioned selection against generic Pegasus taste.",
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

    # Per-run delivery folder: each invocation gets its own timestamped subdir
    # so old runs aren't overwritten and side-by-side comparison is possible.
    # The .work cache stays at job.output_dir/.work (shared across runs).
    from datetime import datetime as _dt
    run_id = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    job._delivery_dir = job.output_dir / f"run_{run_id}"
    job.delivery_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cut] this run's delivery dir: {job.delivery_dir}", file=sys.stderr)

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

    # 0. Acquire source + detect its shape + probe its quality.
    src_path = source.ensure_source(job)
    source_profile = cfg.detect_source_profile(src_path)
    print(f"[cut] source: {src_path} (profile={source_profile})", file=sys.stderr)
    _probe_source_quality(src_path)

    # 1. Whisper full-video transcription.
    words = whisper.transcribe(job, src_path)
    print(f"[cut] whisper: {len(words)} words", file=sys.stderr)

    # 3. Detection + grading.
    cands = detect.find_candidates(job, backend, src_path, words)
    print(f"[cut] {len(cands)} candidates", file=sys.stderr)
    if not cands:
        sys.exit("no candidates produced; refusing to render")

    # 3.5. Phase 5b — Moment Intelligence re-rank. Off in --quickstart mode
    # (no client refs to compare against) or when explicitly overridden.
    if args.client and not args.no_moment_intelligence:
        cands = moment_intelligence.rerank(
            job, cands, words, args.client, profile,
        )

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

    # 5.7. Social post copy — REMOVED. Was a Pegasus.analyze call that uses
    # video_id, deprecated in pegasus1.5. Re-add via the video-object API if
    # this ever becomes a priority. For now: copy can be written by hand.

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
            f"see {job.delivery_dir}/report.json and editorial_report.json"
        )


if __name__ == "__main__":
    main()
