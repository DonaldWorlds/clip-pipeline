# Clip Pipeline — reference-conditioned clip rendering for content creators

You ingest a client's best short-form clips once. The pipeline learns their look. You feed in long-form source, get back ready-to-post short-form in that client's style — no manual editing, no per-clip variant matrix.

## How it works

```
references/<client>/*.mp4   (the Scout / your ingest brings these)
        │
        ▼
   ingest fingerprints   ──►   ClientStyleProfile (rag.db + profile.json)
                                       │
   long-form source  ──►  picks top-N moments  ──►  applies the profile  ──►  finished MP4s
```

**Critical principle:** style is learned from references; it is *not* a property of the pipeline. The pipeline is a delivery mechanism. References + `references/_global/style.md` (the floor) are the brain.

## What you get per client

For every long-form video you feed in, **N short clips × M delivery aspects**, where N is `--top-n` and M is the aspects the client's references actually use. Each output is:

- **Reference-driven moment selection** — TwelveLabs Pegasus picks the candidates; future Phase 5b re-ranks by similarity to the client's reference fingerprints.
- **Verbatim captions** burned in, with position / font / highlight color derived from the client's references (not a fixed catalog of "styles").
- **Filler-word and dead-air trimmed.** No more "ums" and 2-second pauses.
- **Audio normalized** to -14 LUFS (TikTok/Reels/Shorts standard).
- **Color graded** from the client's `profile.color_grade` (brightness, saturation, contrast, temperature — all derived from reference fingerprints).
- **Aspect ratios** = `profile.delivery_aspects` (whatever the client's refs use; most clients = `["9:16"]`).
- **Auto-zoom on static talking-head shots** so sitting-still clips don't feel like screenshots.
- **Real-ESRGAN + GFPGAN** — gated on `profile.use_upscaling` (derived from whether the client's references look enhanced). When on, it runs full-frame or speaker-region depending on shot type.
- **Social post copy** — TikTok / Instagram / YouTube captions + hashtags, gated on `profile.generate_post_copy`.

No thumbnails by default (short-form platforms auto-pick frames). No inset-blur composite. No "Bloomberg/TikTok-bro/Documentary" variant matrix.

## Workflow

1. **Onboard a client once** — ingest 3+ reference clips into `references/<client>/inbox/`, run ingest, profile.json is derived.
2. **For each long-form video:** `python cut.py jobs/<j>.json --client <client> --source /path/to/video.mp4 --top-n 3`
3. **Output:** `out/<job>/01_9x16.mp4`, `02_9x16.mp4`, `03_9x16.mp4` (one per moment, in the client's delivery aspects), plus optional `*_post.json` for social copy.

## Running it yourself

```bash
# One-time setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
# Get a free TwelveLabs API key at twelvelabs.io and put it in .env:
#   TWELVELABS_API_KEY=tlk_...
```

**Production mode (per-client, strict):**

```bash
# 1) Ingest reference clips for the client
python -m clip_pipeline.rag.ingest <client_name>

# 2) Cut a long-form source in that client's style
python cut.py jobs/<j>.json --client <client_name> \
  --source /path/to/your-video.mp4 \
  --top-n 3
```

Fails fast if `--client` has no profile and < 3 ingested references — refusing to render mediocre output for an un-onboarded client.

**Dev / demo mode (no client onboarding, floor defaults from style.md):**

```bash
python cut.py jobs/<j>.json --quickstart \
  --source /path/to/source.mp4 --top-n 3
```

`--quickstart` and `--client` are mutually exclusive.

## What's in the box

- **Whisper** full-video transcription with word-level timestamps
- **TwelveLabs Pegasus** for moment detection (replaces a 7-query keyword search)
- **TwelveLabs Pegasus** also for reference fingerprinting (hook / pacing / captions / payoff descriptions)
- **OpenCV Haar + HOG** for shot classification
- **Real-ESRGAN + GFPGAN** (profile-gated, not auto-on)
- **Sentence-transformers (`all-MiniLM-L6-v2`)** for RAG retrieval embeddings
- **SQLite** for the reference index at `data/rag.db`
- **4-way parallel ffmpeg** render pool

## Architecture (current)

| Layer | Module | Status |
|---|---|---|
| 1. Reference Scout (n8n) | `docs/superpowers/n8n_templates/scout_template.json` | Phase 4 |
| 2. Reference fingerprinting + RAG | `clip_pipeline/rag/fingerprint.py`, `ingest.py`, `retrieve.py` | Phase 3 |
| 3. ClientStyleProfile derivation | `clip_pipeline/profile.py` | Phase 5a |
| 4. Moment selection | `clip_pipeline/analyze/detect.py` (Pegasus); RAG-driven re-rank | Phase 5b (next) |
| 5. Deterministic render | `clip_pipeline/render/pool.py` | Phase 5a |
| 6. Validation (technical + profile-conformance) | `clip_pipeline/validate/{technical,editorial}.py` | Phase 5a |

## Spec / design docs

- [docs/superpowers/specs/2026-05-15-phase5a-client-style-profile-design.md](docs/superpowers/specs/2026-05-15-phase5a-client-style-profile-design.md) — current
- [docs/superpowers/specs/2026-05-13-clip-pipeline-v2-design.md](docs/superpowers/specs/2026-05-13-clip-pipeline-v2-design.md) — superseded in part by 5a
