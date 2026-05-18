"""Stage 3: candidate detection + grading.

Pluggable backends:
  - twelvelabs: upload to TwelveLabs, run search queries, map results
  - local: algorithmic scoring (audio energy + transcript signals) + Ollama grade

Both produce the same graded.json (list of Candidate dicts).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from ..config import Candidate, Job, Word, cache_read, cache_write


# ---------- shared: sentence-boundary refinement ----------

def snap_clip(start: float, end: float, words: list[Word],
              lookaround: float = 3.0,
              min_len: float = 15.0,
              max_len: float = 45.0) -> tuple[float, float]:
    """Refine raw [start, end] to land on natural sentence boundaries from the
    whisper transcript, then enforce min/max length."""
    if not words:
        return start, end

    # Snap start: prefer the latest word.start <= start that's preceded by either
    # a pause >0.6s or end-of-sentence punctuation. Look back up to `lookaround`.
    best_start = start
    for i, w in enumerate(words):
        if w.start < start - lookaround or w.start > start + lookaround:
            continue
        prev_end = words[i - 1].end if i > 0 else 0.0
        pause = w.start - prev_end
        prev_text = words[i - 1].text if i > 0 else ""
        sentence_break = (pause >= 0.6) or prev_text.endswith((".", "!", "?"))
        if sentence_break and abs(w.start - start) < abs(best_start - start):
            best_start = w.start
    start = best_start

    # Snap end: prefer the earliest word.end >= end that itself ends a sentence
    # (its text ends in . ! ? OR the next word is >0.6s away).
    best_end = end
    for i, w in enumerate(words):
        if w.end < end - lookaround or w.end > end + lookaround:
            continue
        next_start = words[i + 1].start if i + 1 < len(words) else w.end + 10
        pause_after = next_start - w.end
        is_sent_end = w.text.endswith((".", "!", "?")) or pause_after >= 0.6
        if is_sent_end and abs(w.end - end) < abs(best_end - end):
            best_end = w.end
    end = best_end

    # Enforce length bounds.
    if end - start < min_len:
        # Extend the end first, then the start, to reach min_len.
        deficit = min_len - (end - start)
        end = min(end + deficit, words[-1].end)
        if end - start < min_len:
            start = max(start - (min_len - (end - start)), words[0].start)
    if end - start > max_len:
        end = start + max_len

    return start, end


def merge_overlapping(cands: list[Candidate], overlap_threshold: float = 0.25) -> list[Candidate]:
    """If two candidates overlap by >= overlap_threshold of the shorter clip's
    duration, keep only the higher-scoring one. Threshold lowered from 0.5 to
    0.25 because TwelveLabs's different queries often surface the same moment
    with slightly different boundaries (rank-1 from query A overlaps rank-1 from
    query B by 30-40%)."""
    sorted_cs = sorted(cands, key=lambda c: -c.score)
    kept: list[Candidate] = []
    for c in sorted_cs:
        keep = True
        for k in kept:
            inter = max(0.0, min(c.end, k.end) - max(c.start, k.start))
            shorter = min(c.end - c.start, k.end - k.start)
            if shorter > 0 and inter / shorter >= overlap_threshold:
                keep = False
                break
        if keep:
            kept.append(c)
    return kept


# ============================================================
# Backend: TwelveLabs
# ============================================================

# Tunable in code. Each query yields up to `page_limit` moments.
TWELVELABS_QUERIES = [
    "high energy moment",
    "emotional reaction",
    "loud or excited reaction",
    "surprising or unexpected statement",
    "punchline or joke",
    "intense delivery",
    "key insight or strong opinion",
]
TWELVELABS_PAGE_LIMIT = 10
TWELVELABS_MIN_RANK_SCORE = 0.15  # candidates below this drop (was 0.30; too strict on short/B-roll content)
# Floor: if score-filtering kills everything, keep at least this many of the
# highest-scoring raw moments anyway so the pipeline always produces output.
TWELVELABS_FALLBACK_KEEP = 30   # Phase 5b: give the re-ranker more raw material


def _twelvelabs_client():
    from twelvelabs import TwelveLabs
    key = os.environ.get("TWELVELABS_API_KEY")
    if not key:
        sys.exit("TWELVELABS_API_KEY not set; cannot use twelvelabs backend")
    return TwelveLabs(api_key=key)


def _twelvelabs_get_index_id(client) -> str:
    """Return the user's first index (Marengo 3.0)."""
    indexes = client.indexes.list()
    items = list(indexes)
    if not items:
        sys.exit("No TwelveLabs indexes found on this account.")
    return items[0].id


def _twelvelabs_upload(job: Job, source_path: Path) -> str:
    """Upload source if not already cached. Returns video_id."""
    cache = job.work_dir / "twelvelabs_videos.json"
    cached = cache_read(cache) if cache.exists() else {}
    h = job.source_hash
    if h in cached:
        print(f"[twelvelabs] reusing cached video_id for source hash {h}", file=sys.stderr)
        return cached[h]

    client = _twelvelabs_client()
    index_id = _twelvelabs_get_index_id(client)
    cached["_index_id"] = index_id  # informational

    print(f"[twelvelabs] uploading source ({source_path.stat().st_size // (1<<20)} MB)…", file=sys.stderr)
    with open(source_path, "rb") as f:
        task = client.tasks.create(index_id=index_id, video_file=f)
    task_id = getattr(task, "id", None) or getattr(task, "_id", None)
    if not task_id:
        sys.exit(f"TwelveLabs tasks.create returned no task id: {task}")
    print(f"[twelvelabs] task {task_id} created; waiting for indexing…", file=sys.stderr)

    def _progress(t):
        status = getattr(t, "status", "?")
        print(f"[twelvelabs] indexing… status={status}", file=sys.stderr)

    done = client.tasks.wait_for_done(task_id, sleep_interval=8.0, callback=_progress)
    status = getattr(done, "status", None)
    video_id = getattr(done, "video_id", None)
    if status != "ready" or not video_id:
        sys.exit(f"TwelveLabs indexing failed: status={status} video_id={video_id}")

    cached[h] = video_id
    cache_write(cache, cached)
    print(f"[twelvelabs] indexed: video_id={video_id}", file=sys.stderr)
    return video_id


def _item_score(item) -> float:
    """Best-effort score extraction from a SearchItem. SDK typed schema only
    exposes rank, but the raw API sometimes returns score; check both."""
    # Try score / confidence attributes first.
    for attr in ("score", "confidence"):
        v = getattr(item, attr, None)
        if isinstance(v, (int, float)):
            return float(v) / 100.0 if v > 1.0 else float(v)
    # Fall back to rank inversion.
    rank = getattr(item, "rank", None)
    if isinstance(rank, int) and rank > 0:
        return 1.0 / (1.0 + (rank - 1) * 0.5)
    return 0.5


def _twelvelabs_search(client, index_id: str, video_id: str) -> list[dict]:
    """Run all queries; return a flat list of moments tied to our video.
    Each moment is a dict {start, end, score, rationale}.

    Score is computed from the item's POSITION in its query's page (1st hit gets
    the highest score, 10th gets the lowest), not from item.rank which is often
    just '1' for every item in the SDK's typed schema."""
    moments: list[dict] = []
    for query in TWELVELABS_QUERIES:
        try:
            page = client.search.query(
                index_id=index_id,
                search_options=["visual", "audio"],
                query_text=query,
                group_by="clip",
                page_limit=TWELVELABS_PAGE_LIMIT,
            )
        except Exception as e:
            print(f"[twelvelabs] query {query!r} failed: {e}", file=sys.stderr)
            continue
        items = list(page)
        print(f"[twelvelabs] query {query!r} -> {len(items)} hits", file=sys.stderr)
        for position, it in enumerate(items):
            vid = getattr(it, "video_id", None)
            if vid and vid != video_id:
                continue
            # Position-based score: 1st = 1.0, 2nd = 0.85, 3rd = 0.72... never < 0.2
            pos_score = max(0.2, 1.0 - position * 0.15)
            # If the SDK exposes a real score/confidence, blend it.
            api_score = _item_score(it)
            score = (api_score + pos_score) / 2 if api_score < 0.95 else pos_score

            start = getattr(it, "start", None)
            end = getattr(it, "end", None)
            if start is None or end is None:
                clips = getattr(it, "clips", None) or []
                for cpos, clip in enumerate(clips):
                    cs = getattr(clip, "start", None)
                    ce = getattr(clip, "end", None)
                    if cs is not None and ce is not None:
                        moments.append({
                            "start": float(cs), "end": float(ce),
                            "score": score * (1.0 - cpos * 0.1),
                            "rationale": query,
                        })
                continue
            moments.append({
                "start": float(start), "end": float(end),
                "score": score, "rationale": query,
            })
    return moments


PEGASUS_PICK_MOMENTS_PROMPT = """You are watching a long-form video. Identify the top 30 best moments to clip for short-form social posts. Be generous — return all plausibly-good moments; a downstream re-ranker will narrow to the best few.

Judge by:
- emotional peaks
- quotable spoken lines
- visual events (cuts, reveals, expressions)
- narrative payoff (the moment a setup pays off)
- surprise or strong contrast

For each moment, return:
  start_s: float — when the moment begins (seconds)
  end_s:   float — when the moment ends   (seconds)
  why_it_works: one short sentence explaining what makes it good
  confidence:  float 0.0 to 1.0
  suggested_clip_length_s: float — typical short-form length for this beat
  dominant_signal: "verbal" | "visual" | "both"

Return STRICT JSON only (no markdown, no commentary):
{"moments": [{"start_s": 0.0, "end_s": 0.0, "why_it_works": "", "confidence": 0.0, "suggested_clip_length_s": 0.0, "dominant_signal": ""}]}
""".strip()


# Module-level cache: True/False once we know, None = haven't probed.
# Prevents repeating multi-line 400 errors on every Pegasus call for
# Marengo-only accounts.
_PEGASUS_AVAILABLE: bool | None = None


def _pegasus_pick_moments(client, video_id: str) -> list[dict]:
    """Single Pegasus call replaces the 7-keyword search. Returns the raw
    moment dicts (start, end, score, rationale). Returns [] on failure so the
    caller can fall back to keyword search."""
    global _PEGASUS_AVAILABLE
    if _PEGASUS_AVAILABLE is False:
        return []  # silent skip — already known unavailable this run
    import json
    import re
    try:
        resp = client.analyze(
            video_id=video_id,
            model_name="pegasus1.2",
            prompt=PEGASUS_PICK_MOMENTS_PROMPT,
            temperature=0.3,
        )
    except Exception as e:
        msg = str(e)
        if "index_not_supported_for_generate" in msg or "video_id is not supported" in msg:
            if _PEGASUS_AVAILABLE is None:
                print(f"[twelvelabs] Pegasus generate not supported on this index — "
                      f"falling back to Marengo keyword search for the rest of the run",
                      file=sys.stderr)
            _PEGASUS_AVAILABLE = False
            return []
        # Other errors are still worth surfacing.
        print(f"[pegasus.pick_moments] failed: {e}", file=sys.stderr)
        return []
    _PEGASUS_AVAILABLE = True
    text = getattr(resp, "data", None) or getattr(resp, "text", None) or str(resp)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return []
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    moments = parsed.get("moments", [])
    out: list[dict] = []
    for m in moments:
        try:
            start = float(m.get("start_s", 0))
            end = float(m.get("end_s", 0))
            conf = float(m.get("confidence", 0.5))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        out.append({
            "start": start,
            "end": end,
            "score": max(0.1, min(1.0, conf)),
            "rationale": str(m.get("why_it_works", "pegasus pick")),
        })
    return out


def detect_twelvelabs(job: Job, source_path: Path, words: list[Word]) -> list[Candidate]:
    """Stage 3 — TwelveLabs backend. Tries Pegasus.analyze first (one call,
    semantic moment selection). Falls back to keyword-query search if Pegasus
    fails or returns nothing."""
    cache = job.work_dir / "graded.json"
    if cache.exists():
        return [Candidate.from_dict(d) for d in cache_read(cache)]

    client = _twelvelabs_client()
    index_id = _twelvelabs_get_index_id(client)
    video_id = _twelvelabs_upload(job, source_path)

    # Try Pegasus first
    print(f"[detect] trying Pegasus.analyze for moment selection...", file=sys.stderr)
    raw_moments = _pegasus_pick_moments(client, video_id)
    pegasus_used = bool(raw_moments)
    if pegasus_used:
        print(f"[detect] Pegasus returned {len(raw_moments)} moments", file=sys.stderr)
    else:
        print(f"[detect] Pegasus path failed/empty — falling back to keyword search",
              file=sys.stderr)
        raw_moments = _twelvelabs_search(client, index_id, video_id)
    if not raw_moments:
        sys.exit("TwelveLabs returned no moments via either Pegasus or keyword search")

    # Diagnostic: surface the score distribution so a "0 candidates" bug is
    # immediately obvious in the log instead of silent.
    raw_moments.sort(key=lambda m: -m["score"])
    sample_scores = [round(m["score"], 3) for m in raw_moments[:5]]
    print(f"[twelvelabs] {len(raw_moments)} raw moments; "
          f"top-5 scores: {sample_scores}", file=sys.stderr)

    # Probe source duration so we can adapt min clip length for short content.
    src_duration = _probe_duration(source_path)
    min_clip_len = 5.0 if src_duration < 60.0 else 15.0

    # Build candidates with sentence-snapped boundaries.
    def _build(raw):
        out: list[Candidate] = []
        for i, m in enumerate(raw):
            s, e = snap_clip(m["start"], m["end"], words, min_len=min_clip_len)
            out.append(Candidate(
                id=f"tl{i:03d}",
                start=s, end=e, score=m["score"],
                rationale=m["rationale"], topic="",
            ))
        return out

    cands = _build([m for m in raw_moments if m["score"] >= TWELVELABS_MIN_RANK_SCORE])

    # Fallback: if the score filter killed everything (common on short or
    # mostly-visual sources), keep the highest-scoring raw moments anyway.
    if not cands:
        print(f"[twelvelabs] all moments below score threshold "
              f"({TWELVELABS_MIN_RANK_SCORE}); falling back to top-"
              f"{TWELVELABS_FALLBACK_KEEP}", file=sys.stderr)
        cands = _build(raw_moments[:TWELVELABS_FALLBACK_KEEP])

    cands = merge_overlapping(cands)
    cands.sort(key=lambda c: -c.score)
    for n, c in enumerate(cands, 1):
        c.id = f"{n:02d}"

    cache_write(cache, cands)
    print(f"[twelvelabs] {len(cands)} candidates after dedup", file=sys.stderr)
    return cands


def _probe_duration(source_path: Path) -> float:
    """ffprobe seconds. Returns 0 if it fails."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(source_path)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


# ============================================================
# Backend: local — transcript-density-based, no third-party API
# ============================================================

# Words that tend to mark interesting moments (reactions, opinions, emphasis).
# Tunable. Match is case-insensitive, punctuation-stripped.
_LOCAL_EMPHASIS_WORDS = {
    "literally", "actually", "really", "wow", "amazing", "incredible",
    "crazy", "insane", "wild", "damn", "honestly", "seriously", "exactly",
    "obviously", "basically", "definitely", "absolutely", "best", "worst",
    "love", "hate", "never", "always", "fucking", "shit",
}

_LOCAL_WINDOW_S = 15.0          # candidate window length in seconds
_LOCAL_STRIDE_S = 5.0           # how far to slide per scoring step
_LOCAL_MIN_WORDS_IN_WINDOW = 8  # below this, window is too sparse to bother
_LOCAL_OVERLAP_THRESHOLD = 0.30  # candidates overlapping by >30% are deduped


def _norm_word(text: str) -> str:
    import re
    return re.sub(r"[^a-z]", "", text.lower())


def _score_local_window(words_in_window: list[Word], window_s: float) -> tuple[float, str]:
    """Score a transcript window for "is this likely a clippable moment."
    Returns (score, rationale). Pure transcript signals — no audio energy
    needed. Heuristic but useful: dense + emphatic windows score higher."""
    if not words_in_window:
        return 0.0, ""
    n = len(words_in_window)
    excl = sum(1 for w in words_in_window if w.text.rstrip().endswith("!"))
    quest = sum(1 for w in words_in_window if w.text.rstrip().endswith("?"))
    emph = sum(1 for w in words_in_window if _norm_word(w.text) in _LOCAL_EMPHASIS_WORDS)
    density = n / window_s  # words per second
    # Weighted score. Density carries the baseline; punctuation + emphasis
    # add multipliers for "this is a reaction or strong claim" moments.
    score = (
        density * 1.0
        + excl * 0.8
        + quest * 0.5
        + emph * 0.6
    )
    notes = []
    if emph > 0:
        notes.append(f"{emph} emphasis")
    if excl > 0:
        notes.append(f"{excl} excl")
    if quest > 0:
        notes.append(f"{quest} quest")
    notes.append(f"{density:.1f} wps")
    return score, ", ".join(notes)


def detect_local(job: Job, source_path: Path, words: list[Word]) -> list[Candidate]:
    """Pure-local moment detection — no TwelveLabs, no API calls.

    Algorithm:
      1. Slide a 15-second window across the whisper transcript with a 5s stride.
      2. Score each window by: word density, exclamation/question marks,
         emphasis-word hits.
      3. Sort windows by score, then greedily pick non-overlapping top ones.

    Less sophisticated than Pegasus/Marengo (no visual signal, no semantic
    understanding) but works without a network or third-party service. Good
    enough for creators who don't want to upload footage to TwelveLabs OR
    for offline use.
    """
    cache = job.work_dir / "graded.json"
    if cache.exists():
        return [Candidate.from_dict(d) for d in cache_read(cache)]

    if not words:
        print("[local-detect] no whisper words — no candidates can be produced",
              file=sys.stderr)
        return []

    source_end = max(w.end for w in words)
    # Slide windows
    proposals: list[dict] = []
    t = 0.0
    while t + _LOCAL_WINDOW_S <= source_end:
        win_start = t
        win_end = t + _LOCAL_WINDOW_S
        win_words = [w for w in words if w.start >= win_start and w.end <= win_end]
        if len(win_words) >= _LOCAL_MIN_WORDS_IN_WINDOW:
            score, notes = _score_local_window(win_words, _LOCAL_WINDOW_S)
            proposals.append({
                "start": win_start,
                "end": win_end,
                "score": score,
                "rationale": f"local: {notes}",
            })
        t += _LOCAL_STRIDE_S

    if not proposals:
        print(f"[local-detect] no windows met the minimum word threshold "
              f"({_LOCAL_MIN_WORDS_IN_WINDOW} words / {_LOCAL_WINDOW_S}s). "
              f"Source may be silent or non-speech.", file=sys.stderr)
        return []

    # Sort by score, greedily pick non-overlapping.
    proposals.sort(key=lambda p: -p["score"])
    picked: list[dict] = []
    for p in proposals:
        ok = True
        for q in picked:
            inter = max(0.0, min(p["end"], q["end"]) - max(p["start"], q["start"]))
            shorter = min(p["end"] - p["start"], q["end"] - q["start"])
            if shorter > 0 and inter / shorter >= _LOCAL_OVERLAP_THRESHOLD:
                ok = False
                break
        if ok:
            picked.append(p)
        if len(picked) >= 30:
            break

    # Normalize scores to 0..1 for downstream consistency with TL scoring.
    if picked:
        max_score = max(p["score"] for p in picked)
        for p in picked:
            p["score"] = p["score"] / max_score if max_score > 0 else 0.5

    # Snap to sentence boundaries
    cands: list[Candidate] = []
    for i, p in enumerate(picked, 1):
        s, e = snap_clip(p["start"], p["end"], words)
        cands.append(Candidate(
            id=f"{i:02d}", start=s, end=e, score=p["score"],
            rationale=p["rationale"], topic="",
        ))

    print(f"[local-detect] {len(cands)} candidates from transcript-density "
          f"scoring (no TwelveLabs)", file=sys.stderr)
    cache_write(cache, cands)
    return cands


# ============================================================
# Dispatcher
# ============================================================

def find_candidates(job: Job, backend: str, source_path: Path,
                    words: list[Word]) -> list[Candidate]:
    if backend == "twelvelabs":
        return detect_twelvelabs(job, source_path, words)
    if backend == "local":
        return detect_local(job, source_path, words)
    sys.exit(f"unknown detection backend: {backend!r}")
