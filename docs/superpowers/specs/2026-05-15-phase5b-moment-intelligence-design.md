# Phase 5b — Moment Intelligence Layer (RAG-driven candidate re-rank)

**Status:** Draft — awaiting user review
**Date:** 2026-05-15
**Builds on:** [2026-05-15-phase5a-client-style-profile-design.md](./2026-05-15-phase5a-client-style-profile-design.md)

## 1. What this is

The selection-intelligence layer the project has been missing. Pegasus already proposes candidate moments (Stage 3 of the pipeline). This layer **re-ranks Pegasus's output by similarity to the client's ingested reference clips** — so the moments that make it to render are not just "interesting moments" by generic taste, but "moments shaped like the ones this client's references prove work."

Today's failure mode: beautifully-formatted mediocre clips. Pegasus picks "high-energy moment" + "punchline" with the same prompt for every client. Phase 5a fixed how clips *look*; Phase 5b fixes which moments *become* clips.

The 7 signals the user originally listed (payoff detection, emotional spikes, conversational tension, story arcs, curiosity loops, retention modeling, semantic density) are **not** hand-coded as separate detectors. They emerge from reference-similarity: a moment whose transcript shape and energy curve match high-performing reference moments is — by construction — paid-off, dense, and tense, because the references that match it were. The references *are* the taxonomy.

## 2. Non-goals (for 5b)

- Replacing Pegasus. It stays as the candidate generator (Stage 3 of `analyze/detect.py`).
- Hand-coded "what is payoff" / "what is curiosity loop" rules. Refs define the shape; we measure fit.
- Audio-energy feature extraction (would need a second pass over the source audio). Out of scope until v2.
- Visual-fingerprint extraction per candidate. Each candidate would need cv2 sampling — too slow for the cheap-first version.
- An LLM-as-judge re-ranker (would mean a Pegasus call per candidate — quota cost). Cheap embeddings only in v1.
- Distribution layer, post-copy improvements, anything Phase 6+.

## 3. Algorithm

```
INPUT:
  - candidates: list[Candidate]    (Pegasus output from Stage 3)
  - words:      list[Word]          (whisper full-video transcription)
  - client:     str                 (drives which refs to retrieve)
  - profile:    ClientStyleProfile  (passed through; informs scoring later)

FOR EACH candidate c:
  1. Build c.transcript_snippet = " ".join(w.text for w in words_in_range(words, c.start, c.end))
  2. Embed snippet (sentence-transformers, same model as ingest)
  3. Retrieve top-K=5 reference fingerprints for the client, ranked by snippet-similarity
     (uses existing rag.retrieve.retrieve(client, query_transcript=snippet))
  4. Compute fit_score:
       - transcript_cos:  mean cosine(snippet_emb, ref.transcript_emb) over retrieved
       - density_z:       z-distance of c's words-per-second vs median(refs' words-per-second)
       - duration_z:      z-distance of (c.end - c.start) vs profile.typical_clip_length_s
       - payoff_cos:      mean cosine(snippet_emb, ref.pegasus_description_emb)
     Combined:
       fit_score = 0.40 * transcript_cos
                 + 0.10 * (1 - clamp(density_z / 3, 0, 1))
                 + 0.10 * (1 - clamp(duration_z / 3, 0, 1))
                 + 0.40 * payoff_cos
  5. final_score = (1 - α) * c.score + α * fit_score
     α = 0.5 default (equal weight to Pegasus's confidence and the references' fit)
  6. Attach c.reference_ids_matched = [ref.id for ref in retrieved top-K]

SORT candidates by final_score descending.
RETURN re-ranked list.
```

**Why these weights:**
- `transcript_cos` + `payoff_cos` carry most of the weight (0.80 combined) because they're the strongest signal of "this moment is shaped like one of theirs."
- Density (words-per-second) and duration are smaller correctives — they catch "the candidate is the right vibe but way too long/short or way too dense/sparse."
- 50/50 Pegasus-vs-fit blend on `final_score` means: if references and Pegasus agree, the candidate dominates; if they disagree, the candidate falls behind one that both sides like better.

**Why no audio/visual per-candidate features in v1:** each would mean a cv2 pass over a 5-45s window per candidate. For 30 candidates that's 1-2 minutes of overhead. Transcript embeddings are sub-second per candidate (batch-encode the lot). We can add the heavier features later if transcript-only re-ranking turns out insufficient.

## 4. Schema delta

`Candidate` dataclass gains two optional fields (default empty; populated only by the re-ranker):

```python
@dataclass
class Candidate:
    # ... existing fields ...
    reference_fit_score: float = 0.0
    reference_ids_matched: list[str] = field(default_factory=list)
```

Re-ranker writes `graded_reranked.json` alongside `graded.json` in the work dir. Schema: same as `graded.json` plus the two new fields. Cache hit on rerun.

## 5. Wiring

```python
# clip_pipeline/analyze/moment_intelligence.py (NEW)
def rerank(
    job: Job,
    candidates: list[Candidate],
    words: list[Word],
    client: str,
    profile: ClientStyleProfile,
    alpha: float = 0.5,
) -> list[Candidate]: ...
```

```python
# cut.py — between Stage 3 (detect) and Stage 4 (top-N selection)
cands = detect.find_candidates(job, backend, src_path, words)

if args.client:
    cands = moment_intelligence.rerank(job, cands, words, args.client, profile)
    print(f"[cut] moment-intelligence: re-ranked {len(cands)} candidates against "
          f"client={args.client}'s references", file=sys.stderr)

selected = _select_top_n(cands, job.top_n)
```

**`--quickstart` skips re-ranking** — no client refs to compare against. The CLI gating mirrors how the rest of the Phase 5a fall-back logic works.

## 6. CLI

Add one flag (off-by-default override):

```
--no-moment-intelligence      Skip the re-rank step even when --client is set.
                              Useful for A/B comparing Pegasus's taste against
                              reference-conditioned re-ranking.
```

No change to `--client` / `--quickstart` semantics.

## 7. Acceptance criteria

1. **With --client and ≥ 3 ingested references**, `cut.py` logs `[cut] moment-intelligence: re-ranked N candidates` and writes `graded_reranked.json` in the work dir.
2. **The top-N selection consumes the re-ranked list**, not Pegasus's original ordering. Verified by: the same source run with `--client X` vs `--client X --no-moment-intelligence` produces a different `selected.json` (assuming the client has refs that differ from Pegasus's taste).
3. **`--quickstart` skips re-ranking silently.** No `graded_reranked.json` written, no `[cut] moment-intelligence:` log line.
4. **Re-rank is idempotent.** Re-running the same job with cached `graded_reranked.json` skips the embedding work.
5. **The re-rank fingerprint extraction is cheap.** On a 10-minute source with 30 candidates, the re-rank step adds < 15 seconds to wall-clock (transcript embeddings batched in one sentence-transformers call).
6. **Each kept candidate has `reference_ids_matched` populated** with the IDs of the top-5 retrieved refs.
7. **A clear failure mode when refs are insufficient:** if the client has < 3 refs in the RAG db, `rerank` returns the input list unchanged and logs `[moment-intelligence] insufficient refs; skipping re-rank`. Does not crash.

## 8. Out of scope (explicit)

- **Audio-energy + visual signatures per candidate.** Future iteration if transcript-only re-rank proves insufficient.
- **An LLM judge over the candidate pool.** A second Pegasus call per candidate is too costly; the embedding-based cosine + structural z-scores carry the load in v1.
- **Learning per-client α weight.** Hard-coded 0.5 blend. Per-client tuning is Phase 6.
- **Hand-coded "what is payoff" detectors.** Decided against in the phase 5b architecture call (RAG-driven, not taxonomy-driven).
- **Re-generating candidates.** This layer only re-ranks what Pegasus surfaced. If Pegasus misses a moment entirely, this layer can't recover it.

## 9. Open risks

- **Embedding-only signal may miss prosody / pacing nuance.** Transcript text-alone can't distinguish "delivered as a punchline" from "delivered as filler." Mitigation: the `payoff_cos` term against the reference Pegasus *description* embedding captures some of this (Pegasus described "punchline" in the ref's payoff field). If insufficient, add audio-energy features in v2.
- **Pegasus's candidate set is the upstream bottleneck.** If Pegasus's prompt misses a moment that would have scored highest on reference fit, the re-ranker has nothing to rank. Mitigation: increase Pegasus's `top_n` ask (currently 10) to ~30 so the re-ranker has more raw material to work with. Tradeoff: more candidates = more transcript embedding work, but still cheap.
- **Reference clips are typically much shorter than candidate windows.** A 14s reference clip embedded vs a 30s candidate snippet — embeddings of different lengths can drift. Mitigation: when scoring, clip the candidate snippet to roughly the median reference length before embedding.
- **Cold-start with few refs is noisy.** With 3-5 refs the median / cosines are unstable. Mitigation: log per-candidate `reference_fit_score` so the operator can see when fit is uniformly low (i.e., refs don't really discriminate) and decide to add more refs or trust Pegasus more (set alpha lower).
- **`graded_reranked.json` cache invalidation.** If the operator ingests new refs for the same client mid-iteration, the cached re-rank is stale. Mitigation: include the source `client` and ref-fingerprint count in the cache file's header; invalidate on mismatch.

## 10. Deliverables

- New: `clip_pipeline/analyze/moment_intelligence.py` — the re-rank function plus internal helpers (snippet embedding, fit scoring, cache I/O).
- Modified: `clip_pipeline/config.py:Candidate` — add the two optional fields.
- Modified: `cut.py` — insert the re-rank call between Stage 3 and Stage 4-selection; gate on `args.client and not args.no_moment_intelligence`; add the CLI flag.
- Modified: `clip_pipeline/analyze/detect.py` — bump `TWELVELABS_FALLBACK_KEEP` from 10 to 30 (gives the re-ranker more raw material).
