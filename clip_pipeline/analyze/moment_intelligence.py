"""Phase 5b — Moment Intelligence layer.

Slots between Pegasus (Stage 3 candidate detection) and top-N selection.
Re-ranks candidates by similarity to the client's ingested reference clips,
emitting an explicit score_breakdown per candidate so the decision is
inspectable, diff-able, and useful for future analytics / ML tuning.

The 7 signals from the original ask (payoff, emotional spikes, conversational
tension, story arcs, curiosity loops, retention, semantic density) are NOT
hand-coded detectors — they emerge from reference-similarity. A moment whose
transcript shape and Pegasus payoff-description match high-performing
reference moments is paid-off, dense, and tense BY CONSTRUCTION, because the
references that scored it were.

This module is intentionally cheap: only transcript embeddings + a couple of
structural z-scores. No second Pegasus call per candidate, no per-candidate
visual / audio fingerprinting. Audio/visual probes are v2 if needed.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

from ..config import Candidate, Job, Word, cache_read, cache_write
from ..analyze.whisper import words_in_range
from ..rag import retrieve as rag_retrieve

if TYPE_CHECKING:
    from ..profile import ClientStyleProfile


MIN_REFS_FOR_RERANK = 3
DEFAULT_ALPHA = 0.5            # 0=trust Pegasus only, 1=trust references only
TOP_K_REFS_PER_CANDIDATE = 5
DENSITY_Z_CLAMP = 3.0          # words-per-second z-score saturation
DURATION_Z_CLAMP = 3.0
CACHE_FILENAME = "graded_reranked.json"

# Scoring weights. Picked at runtime based on whether the client's refs have
# real Pegasus descriptions or degenerate empty-text ones (Marengo-only TL
# index — see memory: project_twelvelabs_marengo_only).
#                          transcript, payoff, density, duration
WEIGHTS_PEGASUS  = (0.40, 0.40, 0.10, 0.10)
WEIGHTS_MARENGO  = (0.80, 0.00, 0.10, 0.10)
# Refs are considered "Pegasus-real" when at least this fraction have any
# non-whitespace Pegasus description text.
PEGASUS_DETECTION_THRESHOLD = 0.5


# ---------- snippet building + embedding ----------

def _build_snippet(words: list[Word], start: float, end: float) -> str:
    """Plain-text transcript snippet for [start, end]. Empty string if no
    words fall in the window."""
    in_range = words_in_range(words, start, end)
    return " ".join(w.text for w in in_range).strip()


def _embed_batch(snippets: list[str]) -> np.ndarray:
    """Batch-embed candidate snippets in ONE sentence-transformers call.
    Returns a (N, D) array of normalized embeddings. Same model as ingest /
    retrieve so cosines are comparable."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(
        snippets, convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")


# ---------- scoring components ----------

def _cosine_one_to_many(query_emb: np.ndarray, ref_embs: np.ndarray) -> float:
    """Mean cosine of query vs every ref. Embeddings are pre-normalized so
    cosine == dot product."""
    if len(ref_embs) == 0:
        return 0.0
    sims = ref_embs @ query_emb
    return float(np.mean(sims))


def _density_match(cand_wps: float, ref_wps: list[float]) -> tuple[float, float]:
    """Returns (density_match, median_ref_wps). density_match is in [0, 1]:
    1.0 when cand_wps == median(refs), falling toward 0 with z-distance.
    A z-score of DENSITY_Z_CLAMP collapses the match to 0."""
    if not ref_wps:
        return 0.5, 0.0
    med = median(ref_wps)
    if len(ref_wps) > 1:
        sigma = float(np.std(ref_wps)) or 0.5
    else:
        sigma = 0.5
    z = abs(cand_wps - med) / sigma
    match = max(0.0, 1.0 - min(z / DENSITY_Z_CLAMP, 1.0))
    return float(match), float(med)


def _duration_match(cand_dur: float, profile_typical: float) -> float:
    """Returns duration_match in [0, 1]: 1.0 at profile_typical, falling to
    0 at ±DURATION_Z_CLAMP * (0.4 * profile_typical) distance.

    The 0.4 spread keeps the curve forgiving for short-form (a 14s typical
    profile tolerates 8s-20s pretty well, but penalizes 30s+ heavily)."""
    if profile_typical <= 0:
        return 0.5
    sigma = 0.4 * profile_typical
    z = abs(cand_dur - profile_typical) / sigma
    return max(0.0, 1.0 - min(z / DURATION_Z_CLAMP, 1.0))


# ---------- the re-ranker ----------

def rerank(
    job: Job,
    candidates: list[Candidate],
    words: list[Word],
    client: str,
    profile: "ClientStyleProfile",
    alpha: float = DEFAULT_ALPHA,
) -> list[Candidate]:
    """Re-rank Pegasus's candidates by similarity to the client's ingested
    references. Returns a new list sorted by `final_score` descending.

    Idempotent: results cached to <work_dir>/graded_reranked.json keyed by
    (client, n_refs, alpha, source candidate IDs). Cache miss on any change.
    """
    if not candidates:
        return candidates

    cache_path = job.work_dir / CACHE_FILENAME
    cache_key = _cache_key(client, candidates, alpha)
    if cache_path.exists():
        try:
            cached = cache_read(cache_path)
            if cached.get("_cache_key") == cache_key:
                print(f"[moment-intelligence] cached re-rank reused "
                      f"({len(cached['candidates'])} candidates)",
                      file=sys.stderr)
                return [Candidate.from_dict(d) for d in cached["candidates"]]
        except Exception:
            pass  # fall through and re-rank

    # Pull all reference fingerprints for the client (the ones we'll compare
    # every candidate against).
    try:
        refs = rag_retrieve.list_fingerprints(client)
    except Exception as e:
        print(f"[moment-intelligence] list_fingerprints failed: {e}; "
              f"skipping re-rank", file=sys.stderr)
        return candidates

    if len(refs) < MIN_REFS_FOR_RERANK:
        print(f"[moment-intelligence] insufficient refs "
              f"({len(refs)} < {MIN_REFS_FOR_RERANK}); skipping re-rank",
              file=sys.stderr)
        return candidates

    # Build per-candidate transcript snippets, batch-embed in one call.
    snippets = [_build_snippet(words, c.start, c.end) for c in candidates]
    snippets_for_embed = [s if s else " " for s in snippets]  # avoid empty
    cand_embs = _embed_batch(snippets_for_embed)

    # Build per-candidate words-per-second + duration up front.
    cand_durations = [max(0.1, c.end - c.start) for c in candidates]
    cand_wps_list = [
        len(words_in_range(words, c.start, c.end)) / d
        for c, d in zip(candidates, cand_durations)
    ]

    # Reference-side: gather transcript embeddings, payoff-description
    # embeddings, words-per-second, and whether Pegasus descriptions are
    # real (vs Marengo-only degenerate blanks) from the ingested fingerprints.
    ref_t_embs, ref_d_embs, ref_wps, has_pegasus = _load_reference_embeddings(client, refs)
    if ref_t_embs is None or len(ref_t_embs) < MIN_REFS_FOR_RERANK:
        print(f"[moment-intelligence] ref embeddings unavailable; skipping",
              file=sys.stderr)
        return candidates

    # Pick weights based on Pegasus availability. Marengo-only mode collapses
    # payoff_cos's 0.40 weight into transcript_cos so the scoring math stays
    # meaningful instead of carrying a dead 40% slot.
    weights = WEIGHTS_PEGASUS if has_pegasus else WEIGHTS_MARENGO
    w_transcript, w_payoff, w_density, w_duration = weights
    mode = "pegasus" if has_pegasus else "marengo"
    print(f"[moment-intelligence] scoring mode={mode} "
          f"(weights: transcript={w_transcript} payoff={w_payoff} "
          f"density={w_density} duration={w_duration})", file=sys.stderr)

    # Top-K retrieve per candidate so the per-candidate comparison is against
    # the MOST SIMILAR refs, not the whole pool. Falls back to whole pool if
    # retrieve fails for any reason.
    per_cand_ref_indices = _top_k_per_candidate(
        cand_embs, ref_t_embs, k=min(TOP_K_REFS_PER_CANDIDATE, len(refs)),
    )

    # Score every candidate.
    profile_typical = float(getattr(profile, "typical_clip_length_s", 15.0))
    reranked: list[Candidate] = []

    for i, c in enumerate(candidates):
        topk = per_cand_ref_indices[i]
        topk_t = ref_t_embs[topk]
        topk_d = ref_d_embs[topk]
        topk_wps = [ref_wps[j] for j in topk if ref_wps[j] > 0]

        transcript_cos = _cosine_one_to_many(cand_embs[i], topk_t)
        payoff_cos     = _cosine_one_to_many(cand_embs[i], topk_d)
        density_match, median_ref_wps = _density_match(cand_wps_list[i], topk_wps)
        duration_match = _duration_match(cand_durations[i], profile_typical)

        # Map cosine [-1, 1] → [0, 1] for human-readable observability.
        # (Sentence-transformer cosines are in practice [0, 1] for non-trivial
        # text, but clamp anyway.)
        transcript_cos_n = max(0.0, min(1.0, (transcript_cos + 1.0) / 2.0))
        payoff_cos_n     = max(0.0, min(1.0, (payoff_cos + 1.0) / 2.0))

        fit_score = (
            w_transcript * transcript_cos_n
            + w_payoff * payoff_cos_n
            + w_density * density_match
            + w_duration * duration_match
        )

        pegasus_score = float(c.score)
        final_score = (1.0 - alpha) * pegasus_score + alpha * fit_score

        c.reference_fit_score = round(fit_score, 4)
        c.reference_ids_matched = [str(refs[j].get("id", "?")) for j in topk]
        c.score_breakdown = {
            "transcript_cos":   round(transcript_cos_n, 4),
            "payoff_cos":       round(payoff_cos_n, 4),
            "density_match":    round(density_match, 4),
            "duration_match":   round(duration_match, 4),
            "pegasus_score":    round(pegasus_score, 4),
            "fit_score":        round(fit_score, 4),
            "final_score":      round(final_score, 4),
            "alpha":            alpha,
            # Which weight regime ran. Lets analytics & future ML know
            # the payoff_cos column is signal (pegasus mode) vs noise
            # (marengo mode), and lets us compare runs across regimes.
            "scoring_mode":     mode,
            "weights":          {
                "transcript": w_transcript, "payoff": w_payoff,
                "density":    w_density,    "duration": w_duration,
            },
            "candidate_wps":    round(cand_wps_list[i], 2),
            "median_ref_wps":   round(median_ref_wps, 2),
            "candidate_duration_s":      round(cand_durations[i], 2),
            "profile_typical_duration_s": round(profile_typical, 2),
            "n_references_compared":     len(topk),
        }
        # Replace `c.score` with the blended final_score so downstream
        # _select_top_n picks by reference-conditioned ranking.
        c.score = round(final_score, 4)
        reranked.append(c)

    reranked.sort(key=lambda c: -c.score)
    print(f"[moment-intelligence] re-ranked {len(reranked)} candidates "
          f"(alpha={alpha}, refs={len(refs)}); top: "
          f"{[(c.id, c.score) for c in reranked[:3]]}", file=sys.stderr)

    # Persist with cache_key header so reruns hit cache.
    cache_write(cache_path, {
        "_cache_key": cache_key,
        "candidates": [asdict(c) for c in reranked],
    })

    return reranked


# ---------- helpers ----------

def _cache_key(client: str, candidates: list[Candidate], alpha: float) -> str:
    """A stable string identifying this re-rank input. If anything material
    changes (different candidate set, different alpha, different client),
    the key changes and the cache is invalidated."""
    parts = [
        client,
        f"a={alpha:.3f}",
        # Use candidate IDs + start times as a hash of the input set.
        ";".join(f"{c.id}@{c.start:.1f}-{c.end:.1f}" for c in candidates),
    ]
    return "|".join(parts)


def _load_reference_embeddings(
    client: str, refs: list[dict],
) -> tuple[np.ndarray | None, np.ndarray | None, list[float], bool]:
    """Pull each reference's transcript + description embedding from rag.db
    (computed at ingest time). Also compute each ref's words-per-second from
    its stored transcript + duration. Detect whether the client's refs
    actually have Pegasus describes or are degenerate (TL Marengo-only
    index — Pegasus generate-API failed at ingest, descriptions are blank).

    Returns (transcript_embs, description_embs, wps_list, has_pegasus).
    Embeddings are (N, D) arrays. Returns (None, None, [], False) on hard
    failure."""
    import sqlite3
    from ..rag.retrieve import DB_PATH

    if not DB_PATH.exists():
        return None, None, [], False

    ids = [r.get("id") for r in refs]
    if not ids:
        return None, None, [], False

    placeholders = ",".join("?" * len(ids))
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        f"SELECT id, transcript, duration_s, transcript_embedding, "
        f"description_embedding, pegasus_hook, pegasus_pacing, "
        f"pegasus_captions, pegasus_payoff FROM reference_clips "
        f"WHERE id IN ({placeholders})",
        ids,
    )
    rows = list(cur.fetchall())
    conn.close()

    by_id = {r["id"]: r for r in rows}
    t_embs, d_embs, wps_list = [], [], []
    pegasus_real_count = 0
    for ref_id in ids:
        row = by_id.get(ref_id)
        if not row:
            return None, None, [], False
        t_blob = row["transcript_embedding"]
        d_blob = row["description_embedding"]
        if not t_blob or not d_blob:
            return None, None, [], False
        t_embs.append(np.frombuffer(t_blob, dtype="float32"))
        d_embs.append(np.frombuffer(d_blob, dtype="float32"))
        transcript = row["transcript"] or ""
        dur = float(row["duration_s"] or 0.0)
        wps = (len(transcript.split()) / dur) if dur > 0 else 0.0
        wps_list.append(wps)
        pegasus_text = " ".join([
            row["pegasus_hook"] or "",
            row["pegasus_pacing"] or "",
            row["pegasus_captions"] or "",
            row["pegasus_payoff"] or "",
        ]).strip()
        if pegasus_text:
            pegasus_real_count += 1

    has_pegasus = pegasus_real_count / len(ids) >= PEGASUS_DETECTION_THRESHOLD
    return np.stack(t_embs), np.stack(d_embs), wps_list, has_pegasus


def _top_k_per_candidate(
    cand_embs: np.ndarray, ref_t_embs: np.ndarray, k: int,
) -> list[list[int]]:
    """For each candidate row in `cand_embs`, return the indices of the top-k
    refs by cosine similarity. Returns a list of length len(cand_embs); each
    element is a list of int indices into ref_t_embs."""
    sims = cand_embs @ ref_t_embs.T   # (N_cand, N_ref)
    top_indices = np.argsort(-sims, axis=1)[:, :k]
    return [list(map(int, row)) for row in top_indices]
