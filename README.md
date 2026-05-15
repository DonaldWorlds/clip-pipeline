# Clip Pipeline — AI editing for content creators

You record long-form. We send back short-form, ready to post.

## What you get

For every video you send, **3 polished short clips** with:

- **AI-picked moments** — the highlights worth posting, not random cuts.
- **Verbatim captions** burned in, three caption styles (clean broadcast / punchy social / restrained editorial).
- **Filler-word and dead-air trimmed.** No more "ums" and 2-second pauses.
- **Audio normalized** to social-media broadcast standard (-14 LUFS — sounds professional on every platform).
- **Color graded** per caption style (broadcast cool / TikTok pop / filmic flat).
- **Three aspect ratios** when needed (9:16 for TikTok/Reels/Shorts, 1:1 for Insta feed, 16:9 for YouTube long-form). For phone-vertical sources we deliver vertical only — and in **two looks**: edge-to-edge and "inset blur" (the polished look creators pay editors for).
- **Auto-zoom on static talking-head shots** so sitting-still clips don't feel like screenshots.
- **Real-ESRGAN + GFPGAN face restoration** when the source needs it (streamer-style recordings, low-res phone footage). The subject stays sharp without the AI-plastic look.
- **A thumbnail per clip** — best-face frame extracted automatically.
- **Ready-to-paste social copy** — TikTok caption, Instagram caption, YouTube description, and 5-7 hashtags per clip.

## Turnaround

Phone-vertical source, ~10 minutes long → **~5 minutes** per finished clip pack.
Landscape source with streamer-overlay treatment (face restoration + composites) → **~12-15 minutes** per finished clip pack.

First time a particular source is processed, add ~5-10 minutes for AI indexing — cached after.

## Workflow

1. You hand over the long-form (file or YouTube URL).
2. We process it locally on Apple Silicon (faster-whisper, OpenCV, ffmpeg, ESRGAN/GFPGAN) with TwelveLabs for moment detection.
3. You get a delivery folder per clip:
   - `01_bloomberg_9x16.mp4` — broadcast style
   - `01_tiktok_bro_9x16.mp4` — social style
   - `01_documentary_9x16.mp4` — editorial style
   - `01_bloomberg_9x16_inset.mp4` — same, inset-blur look
   - `01_thumb.jpg` — thumbnail
   - `01_post.json` — caption copy ready to paste, per platform

## Who it's for

- Creators recording on phone who want to ship daily without an editor.
- Podcasters who want clips from each episode without paying $50-200 per cut.
- Streamers / reaction-content creators (we have a special pipeline that face-restores the small webcam overlay so it stays sharp at vertical aspects).

## Running it yourself

```bash
# One-time setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
# Get a free TwelveLabs API key at twelvelabs.io and put it in .env:
#   TWELVELABS_API_KEY=tlk_...
```

**Phone-vertical source (most content creators):**

```bash
python cut.py jobs/phone_creator.json \
  --source /path/to/your-video.mp4 \
  --top-n 3 --keep-temp
```

Output lands in `out/phone_creator/` — 3 styled clips × `_full` + `_inset` looks + thumbnail + post copy per clip.

**Landscape source (streamer-style screen recordings, podcasts, interviews):**

```bash
python cut.py jobs/myjob.json --source /path/to/source.mp4 --top-n 3 --keep-temp
```

Output is 3 clips × 3 styles × multiple aspects + face-restored composites where appropriate.

## What's in the box

- Whisper full-video transcription with word-level timestamps
- TwelveLabs Marengo for moment detection (visual + audio semantic search)
- TwelveLabs Pegasus for social caption copy generation
- OpenCV Haar cascade + HOG person detection for shot classification
- Real-ESRGAN + GFPGAN for subject upscaling (used only when source is genuinely low-res)
- Per-variant ffmpeg color grading (no external LUT files needed)
- 4-way parallel ffmpeg render pool

## Pricing (template)

| Tier | Includes | Price |
|---|---|---|
| Per-clip | 1 finished clip pack from a single source moment | $ _____ |
| Per-source | 3 clips from one source, full deliverable | $ _____ |
| Monthly retainer | Unlimited sources / N clips per month / N-day turnaround | $ _____ / mo |

## Spec / design docs

[docs/superpowers/specs/2026-05-13-clip-pipeline-v2-design.md](docs/superpowers/specs/2026-05-13-clip-pipeline-v2-design.md)
