# Content Lab Pipeline — Implementation Plan

> **Status:** Awaiting your review. No code changes have been made yet. Once you sign off (or push back on specific phases), implementation starts.
> **Date:** 2026-05-14
> **Supersedes:** the v2 design ([2026-05-13-clip-pipeline-v2-design.md](../specs/2026-05-13-clip-pipeline-v2-design.md))

## Vision (locked in this session)

Multi-tenant **AI Content Lab** for a marketing agency.
- **n8n** is a narrowly-scoped Scout: one workflow per client niche, scrapes trending content, filters by engagement, drops top picks into `references/inbox/<client>/`. Nothing else.
- **Custom Python** owns everything else: ingest references → local RAG → editorial decisions grounded in retrieved references → render → editorial validation → (later) analytics feedback.
- **No managed services**. SQLite + numpy + polars + sentence-transformers locally. Target ongoing cost $0–15/mo (just an optional scraper plan + optional VPS for n8n).

## Architecture, one frame

```
[n8n Scout — one workflow per client niche, runs daily]
     │ scrape trending → filter by engagement → download top-N
     ▼
references/inbox/<client>/*.mp4 + sidecar.json
     │ Python ingest:
     │   - extract numeric fingerprint (OpenCV/numpy)
     │   - VISION: Pegasus describes "what makes this clip work"   ◀── eyes
     │   - embed transcript + description
     ▼
data/rag.db (SQLite) + data/embeddings/*.npy
     │ Python retrieve queries this
     ▼
cut.py <source.mp4> --client <name>
   ├─ analyze
   │    ├─ whisper                                          [exists]
   │    ├─ detect — VISION: Pegasus picks moments by content [Phase 3 upgrade]  ◀── eyes
   │    └─ framing                                          [exists]
   │
   ├─ editorial  [NEW Phase 2 + 3]
   │    ├─ rules        ← style.md                           Phase 2
   │    ├─ decisions    ← rules + retrieved refs + numeric   Phase 2 (refs in Phase 3)
   │    ├─ pre-render check — VISION (optional)              Phase 2  ◀── eyes
   │    └─ copy         ← Pegasus social post copy           [exists]  ◀── eyes
   │
   ├─ render     (trim + enhance + grade + ass + ffmpeg)    [exists, untouched]
   │
   ├─ validate
   │    ├─ technical   (ffprobe + still-frame)               [exists]
   │    └─ editorial   — rules-check + VISION divergence     Phase 2  ◀── eyes
   │
   ├─ thumbnail                                              [exists]
   └─ (copy already in editorial above)
     │
     ▼
out/<job>/   delivery folder: mp4s + thumb + post.json + editorial_report.json
```

Five workers (Scout, Librarian, Rulebook, Studio, Inspector). Only three live in this Python repo (Librarian = `clip_pipeline/rag/`, Rulebook = `references/_global/style.md`, Studio+Inspector = `cut.py` + `clip_pipeline/`).

**Vision-LLM (eyes) is wired in at 5 distinct stages:**
1. Reference fingerprinting (ingest) — Pegasus describes each reference clip in natural language
2. Moment detection — Pegasus picks moments by content understanding, not keyword queries
3. Pre-render sanity check (optional flag) — Pegasus/Claude vision predicts pass/fail before render
4. Editorial validation (post-render) — Pegasus compares output frames to reference frames
5. Social caption copy — Pegasus writes TikTok/IG/YouTube copy (already wired)

Costs: stages 1, 2, 4, 5 use TwelveLabs Pegasus free tier. Stage 3 optional Claude Sonnet vision at ~1¢/clip when enabled.

---

## Phase 1 — Cleanup + Restructure (½ day, mechanical)

**Goal:** Delete the 23 dead Layer 2 files, consolidate the 7 markdown editorial files, restructure `clip_pipeline/` into clear subfolders. **No behavior change** to the running pipeline.

### Files to delete

```
agent.py
enum.py                                  ← timebomb that shadows stdlib
metrics.py
parallell_manager.py
queue_manager.py
cut.py.v1                                ← stale v1 backup

RAG/build_rag.py                         ← empty placeholder
RAG/query_rag.py                         ← empty placeholder

mcp/filesystem_tools.py
mcp/rag_tools.py                         ← syntax error
mcp/sqlite_tools.py
mcp/terminal_tools.py
(remove mcp/ directory entirely)

validator/validator.py                   ← duplicates clip_pipeline/verify.py
validator/captions_subtitle_validation.py ← plain English, not code
validator/quality_scoring.py             ← syntax error
validator/retry_render.py                ← broken imports
validator/test_validation.py             ← broken imports
(remove validator/ directory entirely)

feedback_loop/feedback.py
feedback_loop/feedback_memory.py
feedback_loop/examine_feedback.py
(remove feedback_loop/ directory entirely)

tools/models.py                          ← Pydantic models, unused
tools/tools.py                           ← duplicates code in clip_pipeline
tools/test_tools.py                      ← broken imports
(remove tools/ directory entirely)
```

### Files to consolidate (keep editorial content)

Merge into a single file `references/_global/style.md`:

- `RAG/knowledge/captions_rules.md` — caption position, count, font size, safe margins
- `RAG/knowledge/clip_styles.md` — movement cadence, silence limit, hook timing, layout
- `RAG/knowledge/creator_styles.md` — pacing notes
- `RAG/knowledge/ffmpeg_patterns.md` — encoder defaults
- `RAG/knowledge/known_bugs.md` — known fixes
- `reference/notes.py/kai_fast_zoom_reference.md` — specific reference clip notes

Then delete `RAG/` and `reference/` directories entirely.

### Restructure clip_pipeline/

```
clip_pipeline/
├── __init__.py
├── config.py                          (unchanged)
├── source.py                          (unchanged)
├── caption.py                         (unchanged — used by render)
├── analyze/                           ← NEW grouping (move existing files)
│   ├── __init__.py
│   ├── whisper.py
│   ├── detect.py
│   └── framing.py
├── editorial/                         ← NEW package (placeholder in Phase 1; populated in Phase 2)
│   ├── __init__.py
│   ├── copy.py                        ← move clip_pipeline/copy.py here
│   ├── rules.py                       ← Phase 2
│   └── decisions.py                   ← Phase 2 (replaces intelligence.py)
├── render/                            ← NEW grouping (move existing files)
│   ├── __init__.py
│   ├── pool.py                        ← parallel render driver from render.py
│   ├── grade.py
│   ├── trim.py
│   ├── enhance.py
│   └── strategies/                    ← Phase 1 cosmetic split (later if it helps)
│       ├── simple.py
│       ├── composite.py
│       └── inset.py
├── validate/                          ← NEW grouping
│   ├── __init__.py
│   ├── technical.py                   ← renamed verify.py
│   └── editorial.py                   ← Phase 2
├── thumbnail.py                       (unchanged for now)
└── intelligence.py                    ← KEEP for one phase, deprecated in Phase 2
```

### Steps

- [ ] **Step 1: Tag current working state.** `git init` if not yet a repo; commit the current state as `before-cleanup` so we can revert if anything breaks.
- [ ] **Step 2: Delete the 23 dead files** (see list above).
- [ ] **Step 3: Consolidate markdown into `references/_global/style.md`.**
- [ ] **Step 4: Move existing modules into the new `analyze/`, `render/`, `validate/` subfolders.** Update imports in `cut.py`.
- [ ] **Step 5: Move `copy.py` → `editorial/copy.py`.** Update imports.
- [ ] **Step 6: Rename `verify.py` → `validate/technical.py`.** Update imports.
- [ ] **Step 7: Run `python cut.py --help`** to confirm imports still resolve.
- [ ] **Step 8: Smoke test on the cached IMG_2606 source** — should produce identical outputs to pre-cleanup.
- [ ] **Step 9: Commit `phase-1-cleanup-complete`.**

**Success criteria:** `cut.py jobs/phone_creator.json --source IMG_2606.mov --top-n 1 --keep-temp` produces the same MP4 count and same verify pass as before, and the repo file count drops by 23.

---

## Phase 2 — Editorial Rules Layer (1–2 days)

**Goal:** Replace the hardcoded `intelligence.py` heuristics with decisions grounded in the consolidated `references/_global/style.md` rulebook. **This is the first version where the pipeline has actual editorial taste, even without per-client reference clips.**

### New code

**`clip_pipeline/editorial/rules.py`** (~150 lines)

Loads `references/_global/style.md` and parses the rule statements into a typed `EditorialRules` dataclass:

```python
@dataclass
class EditorialRules:
    caption_position: str           # "center" | "lower_third" | "upper_third"
    caption_max_lines: int          # 2
    caption_min_font_px: int        # 70
    caption_highlight_color: str    # "yellow"
    safe_margins_pct: float         # 5.0 (% inside frame edge)

    max_silence_s: float            # 1.0
    movement_cadence_s: float       # 2.0 (something visual changes every N seconds)
    hook_first_s: float             # 0.5 (something happens in the first half-second)
    zoom_on_loud_moments: bool

    clip_min_s: float               # 5.0
    clip_max_s: float               # 30.0

    encoder_profile: dict           # 1080x1920, yuv420p, h264, fps=60
```

Rules are parsed from the markdown via simple regex/keyword matching, with sensible defaults. The dataclass is the single source of editorial truth for the rest of the pipeline.

**`clip_pipeline/editorial/decisions.py`** (~200 lines, replaces `intelligence.py`)

Takes the loaded `EditorialRules` + per-clip `framing` signals + the candidate's audio energy/transcript, and produces `Decision` per clip:

```python
@dataclass
class Decision:
    clip_id: str
    hero_aspect: str
    crop_tightness: float
    caption_y_fraction: float       # GROUNDED — derived from rules + framing.speaker_rect
    caption_font_px: int            # GROUNDED — derived from rules
    caption_highlight_color: str    # GROUNDED — from rules
    apply_zoom: bool                # GROUNDED — loud-moment detection on audio
    apply_filler_trim: bool         # always True if rules say max_silence_s
    grade_intensity: float
    use_esrgan: bool                # only when speaker_rect tiny vs frame
    rationale: str                  # human-readable explanation of choices
```

The function signature:

```python
def decide_clip(
    candidate: Candidate,
    framing: Framing,
    rules: EditorialRules,
    audio_energy: list[float] | None = None,
) -> Decision:
```

**Concrete rule-grounded decisions:**
- `caption_y_fraction` is computed so captions don't overlap `framing.speaker_rect` AND respect `rules.safe_margins_pct` AND obey `rules.caption_position`.
- `caption_font_px` = `max(rules.caption_min_font_px, output_height * 0.05)`.
- `apply_zoom` = True if the audio has loud-moment peaks AND `rules.zoom_on_loud_moments`.
- `apply_filler_trim` = True if `rules.max_silence_s` is set (already always-on in current trim.py).

### Editorial validator (with vision)

**`clip_pipeline/validate/editorial.py`** (~150 lines)

Two-layer validation: mechanical rules-check + vision-LLM divergence check.

```python
@dataclass
class EditorialReport:
    clip_id: str
    rules_obeyed: list[str]
    rules_violated: list[dict]      # {"rule": "caption_position", "expected": "center", "observed": "lower_third"}
    vision_findings: list[str]      # natural-language flags from Pegasus / Claude vision
    fingerprint: dict               # for later use by RAG
```

**Mechanical rules-check (always runs, free):**
- Read 5 frames from the rendered clip
- Detect caption position in each frame (parse .ass alignment + margin)
- Detect face position
- Confirm captions don't overlap face
- Confirm font size ≥ rules.caption_min_font_px (read from .ass directly)
- Confirm clip duration in `[rules.clip_min_s, rules.clip_max_s]`
- Confirm no silence > rules.max_silence_s (read from `trim_report.json`)

**Vision-LLM check (Pegasus first, optional Claude Sonnet vision):**
- Extract 3 frames from rendered clip (10%, 50%, 90% positions)
- Send to TwelveLabs Pegasus via `client.analyze` with a focused prompt:
  ```
  "Look at these frames from a short-form video clip. Check for these specific
  issues and return JSON {issues: [...], pass: bool}:
   1. Subject cropped at edge / partially out of frame
   2. Captions overlap face or main subject
   3. Captions cut off at edge
   4. Visible artifacts (compression, scaling, distortion)
   5. Color cast or banding"
  ```
- Findings appended to `vision_findings`. If anything is critical (subject cropped, captions cut off), it's added to `rules_violated` so the clip fails verification.
- Cost: free (Pegasus). Falls back gracefully if TwelveLabs is unavailable.

Failures recorded in `out/<job>/editorial_report.json`. If any clip has any `rules_violated`, `cut.py` exits non-zero.

### Optional pre-render sanity check

A `--verify-before-render` CLI flag in `cut.py` runs ONE vision call per clip BEFORE the expensive render. Given the decision JSON + 3 source frames + relevant rules, asks Pegasus or Claude vision:

> "Given these decision parameters and these source frames, will the resulting render meet the rules? Specifically check caption_y_fraction won't overlap visible elements at the times when captions appear. Return {predict_pass: bool, concerns: [...]}."

If `predict_pass=False`, the decision is adjusted (e.g., caption_y_fraction moved) or the clip is dropped from the render queue. Saves render time on clips that would have failed editorial validation anyway.

Cost: optional, Pegasus free / Claude Sonnet vision ~1¢ per clip when enabled.

Failures are recorded in `out/<job>/editorial_report.json`. If any clip has any rules_violated, `cut.py` exits non-zero.

### cut.py wiring

```python
# After framings, before render:
rules = editorial.rules.load_global()        # reads references/_global/style.md
decisions = [editorial.decisions.decide_clip(c, f, rules) for c, f in zip(selected, framings)]
cfg.cache_write(job.work_dir / "decisions.json", decisions)

# Render takes decisions instead of recs:
produced = render.render_all(job, src_path, words, selected, framings, variants, decisions)

# After render, before delivery:
editorial_report = validate.editorial.run(job, produced, decisions, rules)
```

### Steps

- [ ] **Step 1: Write `editorial/rules.py`** with the dataclass and markdown parser. Unit-test by loading `references/_global/style.md` and printing the parsed rules.
- [ ] **Step 2: Write `editorial/decisions.py`.** Port the useful bits of `intelligence.py` heuristics into rules-grounded form. Delete `intelligence.py` afterward.
- [ ] **Step 3: Update `render.py`** to accept the new `Decision` dataclass instead of `recs: dict`. Wire `Decision.caption_y_fraction` into the caption builder.
- [ ] **Step 4: Update `caption.py`** to honor `Decision.caption_y_fraction` and `caption_font_px` and `caption_highlight_color` per clip, rather than relying solely on `variants.json`.
- [ ] **Step 5: Write `validate/editorial.py` mechanical layer.** All the rules-compliance checks. Add to `cut.py` after render.
- [ ] **Step 6: Add vision-LLM layer to `validate/editorial.py`.** Pegasus `client.analyze` call against 3 sampled frames per output. Findings appended to the editorial report.
- [ ] **Step 7: (Optional) Add `--verify-before-render` flag to `cut.py`.** Vision sanity-check per clip BEFORE render. Saves time on doomed clips.
- [ ] **Step 8: Smoke test on IMG_2606.** Captions, font, position now driven by rulebook. Pegasus check reports any visible issues. `editorial_report.json` shows specific violations or `pass: True`.
- [ ] **Step 9: Commit `phase-2-editorial-rules`.**

**Success criteria:** the pipeline produces outputs where every editorial decision traces to a specific line in `references/_global/style.md`. The mechanical rules-check passes AND the Pegasus vision-check reports `pass: True` for every clip. When the smoke output has a problem (caption overlap, cropped subject), the editorial report flags it specifically rather than the pipeline silently passing a bad clip.

---

## Phase 3 — RAG Librarian (1–2 days)

**Goal:** Stand up the local RAG (the Librarian) so that the editorial decisions module can retrieve reference clips and ground decisions in real high-performing content, not just the global rulebook.

### Dependencies added

```
sentence-transformers>=2.7   # ~80MB local model, free
# (sqlite is stdlib; numpy + polars + cv2 already installed)
```

### New code

**`clip_pipeline/rag/fingerprint.py`** (~200 lines)

Extracts THREE signal types per reference video:

```python
def fingerprint_clip(video_path: Path, with_vision: bool = True) -> ClipFingerprint:
    # 1. Transcript (faster-whisper, same as analyze/whisper.py)
    # 2. Numeric/visual fingerprint (OpenCV + numpy):
    #    avg brightness, color temp, motion_score, cuts/sec, face_present_pct,
    #    face_position, zoom_cadence_s, max_silence_s, dominant_aspect, fps
    # 3. Vision-LLM fingerprint (TwelveLabs Pegasus, when with_vision=True):
    #    natural-language description of "what makes this clip work"
    #    — hook style, pacing energy, caption style, emotional arc
```

Shared between `rag/ingest.py`, `validate/editorial.py`, and the moment-detection upgrade in `analyze/detect.py`.

**`clip_pipeline/rag/ingest.py`** (~200 lines)

Watches `references/inbox/<client>/`. For each new MP4:

1. Read sidecar JSON (niche, source URL, engagement stats).
2. Upload video to TwelveLabs (cached — only first time per reference).
3. Run `fingerprint_clip(video, with_vision=True)`:
   - Numeric features via OpenCV
   - **Vision description via Pegasus** with prompt:
     ```
     "This clip got {views} views and {likes} likes in the {niche} niche.
      In one paragraph each, describe:
       1. THE HOOK — what happens in the first 2 seconds that grabs attention.
       2. PACING — how quickly cuts, zooms, or visual changes occur.
       3. CAPTION STYLE — position, size, color, animation, key-word treatment.
       4. PAYOFF — the moment of peak engagement and how it's delivered.
      Return strict JSON with keys: hook, pacing, captions, payoff."
     ```
4. Compute transcript embedding via `sentence-transformers/all-MiniLM-L6-v2`.
5. **Compute description embedding** — embed the Pegasus description so retrieval can match on style/vibe, not just transcript content.
6. Insert row into `data/rag.db` (SQLite). Embeddings stored as BLOB columns.
7. Move MP4 from `inbox/` to `references/<client>/<id>.mp4`.
8. Mark processed in a `references/inbox/<client>/.processed` lockfile.

Run via `python -m clip_pipeline.rag.ingest <client>` (cron / n8n trigger / manual).

**`clip_pipeline/rag/retrieve.py`** (~100 lines)

```python
def retrieve(
    client: str,
    niche: str,
    k: int = 5,
    query_transcript: str | None = None,
    query_visual: VisualSignature | None = None,
) -> list[ClipFingerprint]:
    # SQL filter: WHERE client=? AND niche=?
    # Cosine similarity over BOTH the transcript embedding AND the Pegasus
    # description embedding. Top-K by combined score.
    # Each returned fingerprint includes the Pegasus description so the
    # decisions module has both numbers AND natural-language guidance.
```

### Moment detection upgrade (Pegasus replaces keyword queries)

**Replace `analyze/detect.py`'s 7 keyword queries with ONE Pegasus analyze call:**

```python
# Before (Phase 2 and earlier — keyword search):
for query in ["high energy moment", "emotional reaction", ...]:
    page = client.search.query(query_text=query, ...)

# After (Phase 3 — content-aware moment detection):
result = client.analyze(
    video_id=video_id,
    model_name="pegasus1.5",
    prompt="""Identify the top 10 best moments for short-form clips. Judge by:
    emotional peaks, quotable spoken lines, visual events, narrative payoff,
    surprise or contrast. For each return:
    {start_s, end_s, why_it_works (one sentence), confidence (0-1),
     suggested_clip_length_s, dominant_signal (verbal|visual|both)}
    Return strict JSON: {moments: [...]}""",
)
```

Why this is better:
- One API call instead of 7 (faster, less quota).
- Pegasus reads the FULL video and chooses what HE thinks is best, with a rationale.
- The output already includes `suggested_clip_length_s` — no more guessing at clip length.
- `why_it_works` becomes the candidate's `rationale` field downstream.

### Hooking into editorial decisions (three grounding layers)

`editorial/decisions.py` blends three sources:

```python
def decide_clip(candidate, framing, rules, audio_energy=None, references=None) -> Decision:
    # Layer 1: rules (always loaded from style.md)
    target_caption_y = rules.compute_caption_y(framing.speaker_rect)
    target_font_px = rules.caption_min_font_px

    # Layer 2: retrieved reference fingerprints (when --client passes)
    if references:
        # NUMERIC grounding: median values from retrieved references
        ref_caption_y = median([r.editorial.caption_y_frac for r in references])
        ref_font_px = median([r.editorial.caption_font_px for r in references])
        ref_zoom_cadence = median([r.editorial.zoom_cadence_s for r in references])
        # Blend: 70% reference, 30% rules (references override rules where they conflict)
        target_caption_y = 0.7 * ref_caption_y + 0.3 * target_caption_y
        target_font_px = max(target_font_px, ref_font_px)

        # SEMANTIC grounding: Pegasus descriptions become part of the rationale
        rationale = (
            f"Matched 5 reference clips in {client}/{niche}. "
            f"Their pacing notes: {[r.vision.pacing for r in references]}. "
            f"Applying: caption_y={target_caption_y:.2f}, font={target_font_px}px"
        )

    return Decision(..., rationale=rationale)
```

Three layers of grounding:
1. `references/_global/style.md` (rulebook) — always loaded
2. `references/<client>/*.mp4` (retrieved fingerprints + Pegasus descriptions) — loaded when client has reference clips
3. Heuristics — only used as last resort when neither rules nor references cover a decision

### Steps

- [ ] **Step 1: Add `sentence-transformers` to requirements.txt.** `pip install` it.
- [ ] **Step 2: Write `rag/fingerprint.py`** — numeric (OpenCV) + Pegasus description.
- [ ] **Step 3: Write the SQLite schema.** Single migration file `data/schema.sql`. Two embedding BLOB columns (transcript + Pegasus description).
- [ ] **Step 4: Write `rag/ingest.py`.** Run it against a hand-curated reference clip; confirm a row in SQLite with both numeric features AND a Pegasus description.
- [ ] **Step 5: Write `rag/retrieve.py`.** Cosine-similarity search across both embedding columns.
- [ ] **Step 6: Upgrade `analyze/detect.py`.** Replace 7-keyword search with a single Pegasus `analyze` call. Fallback to the old keyword search if the analyze call fails (resilience).
- [ ] **Step 7: Wire retrieve into `editorial/decisions.py`.** When `--client <name>` is passed AND that client has indexed references, retrieve them and blend (numeric + Pegasus description) into the decision.
- [ ] **Step 8: Smoke test the same IMG_2606 source with --client phone_creator_v1.** Should now produce decisions explicitly citing reference clips with Pegasus descriptions in the rationale.
- [ ] **Step 9: Commit `phase-3-rag-librarian`.**

**Success criteria:**
1. Dropping a new MP4 in `references/inbox/<client>/` causes the next `cut.py` run for that client to retrieve it and quote its Pegasus description in `decisions.json[N].rationale`.
2. `analyze/detect.py` returns moments with `why_it_works` text from Pegasus, not just generic keyword scores.
3. End-to-end: from raw source → 3 finished clips, each clip's editorial choices traceable to (a) a rule in `style.md`, (b) a numeric median from references, or (c) a Pegasus description from references.

---

## Phase 4 — n8n Scout (your work, ½ day per niche)

**Goal:** One n8n workflow per client niche that scrapes trending content and drops references into `references/inbox/<client>/`.

### Workflow design

Trigger: `Schedule Trigger` node, daily at 6am.

Steps:

1. **HTTP / Apify node** — call your scraper of choice. Two cheap options:
   - **Apify TikTok Scraper** (apify.com/clockworks/free-tiktok-scraper) — free or $5/mo. Returns trending posts with engagement stats.
   - **YouTube Shorts Trending RSS** — free via search results, lighter weight.
2. **Filter node** — keep only posts where `likes / views > 0.06` AND `views > 50000` AND niche tag matches.
3. **Sort + Limit node** — top 10 by `(likes + saves * 3 + shares * 5)`.
4. **Loop / SplitInBatches node** — for each post:
   a. **HTTP Download** the video file.
   b. **Move Binary Data → Write Binary File** node, target path: `references/inbox/<client>/<post_id>.mp4`.
   c. **Write JSON File** node, target path: `references/inbox/<client>/<post_id>.json` with fields `{niche, source_url, views, likes, saves, shares, posted_at}`.
5. **Webhook / Execute Command node (optional)** — POST to a local Python endpoint or run `python -m clip_pipeline.rag.ingest <client>` to trigger the Librarian.

### Output contract (what Python expects)

```
references/inbox/<client>/<post_id>.mp4         # the video
references/inbox/<client>/<post_id>.json        # sidecar with metadata
references/inbox/<client>/.processed            # lockfile written by Python ingest
```

Sidecar JSON schema:
```json
{
  "niche": "gaming_reactions",
  "source_url": "https://www.tiktok.com/@user/video/...",
  "client": "client_name",
  "views": 412000,
  "likes": 38000,
  "saves": 6200,
  "shares": 2100,
  "posted_at": "2026-05-13T18:24:00Z"
}
```

I'll spec this in finer detail when you're ready to build it. The Python side is fully ready for it once Phase 3 is done.

---

## Phase 5 — Score Keeper (deferred — needs real engagement data)

**Goal:** After you post a clip and accumulate engagement, the system learns from it and adds top-performing OWN clips to the reference library.

**Why deferred:** until you have actual posted clips with real engagement numbers, this is theoretical. We design it once we have data to design against.

**Expected shape (placeholder):**
- A `feedback.py` CLI: `python -m clip_pipeline.feedback record --clip <id> --views <n> --likes <n> --saves <n>`
- A SQLite table `clip_engagement` keyed on `clip_id`
- A migration that re-fingerprints and inserts top-performing OUR clips into the same RAG used by Phase 3
- Optional: a "Why did this perform well?" Claude API call (Sonnet) for retrospective analysis on the top performers, results appended to `references/_global/learned_patterns.md`

We add this in 2-3 weeks of operating the system.

---

## Total scope

| Phase | Effort | Risk | Touches running pipeline? |
|---|---|---|---|
| 1 — Cleanup | ½ day | None (no behavior change) | No, just file reorganization |
| 2 — Editorial rules | 1–2 days | Medium (replaces intelligence.py) | Yes — output decisions change |
| 3 — RAG Librarian | 1–2 days | Low (additive, opt-in via --client) | Only when --client passed |
| 4 — n8n Scout | ½ day per niche | Low | Doesn't touch Python at all |
| 5 — Score Keeper | Deferred | Low | Additive, doesn't change render |

**Python work to first client demo:** Phases 1+2+3 = **3-4 focused days**.

---

## Open decisions to confirm before we start

1. **Phase 1 cleanup is destructive (delete 23 files).** Confirm we git-init + tag before deleting, so it's reversible.
2. **Phase 2 deletes `intelligence.py`.** Confirm you're OK with the heuristic-based decisions module being replaced by rule-grounded decisions.
3. **`references/_global/style.md`** is the FIRST version of the editorial spec, consolidated from your existing markdown. We may want to add MORE rules to it as we discover gaps. Confirm: it's editable by you anytime, even after deployment.
4. **No `--client` flag exists today.** Phase 3 introduces it. For Phase 1 and 2 the pipeline still runs with just `--source` and uses global rules only. Confirm.
5. **n8n workflow design in Phase 4 assumes Apify (paid) OR YouTube Shorts RSS (free).** Confirm which path you want, or specify a different scraper.

---

## Out of scope (explicit, so we don't add later)

- Multi-tenant config inside the Python codebase (Phase 1-3 use a single `--client <name>` flag; if you scale to 50 clients we can introduce per-client config folders, but not before).
- Real-time / streaming pipelines.
- Auto-distribution to TikTok/IG/Shorts APIs. (n8n's job downstream, not ours.)
- Custom training / fine-tuning of GFPGAN or ESRGAN models.
- Replacing TwelveLabs detection with a custom local model (TwelveLabs free tier is generous, no reason to swap yet).
- Anything that requires a managed service (Pinecone, ChromaDB Cloud, OpenAI embeddings, etc.).
