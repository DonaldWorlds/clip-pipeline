# Clip Pipeline v2 — Auto-Detect + Multi-Aspect Design

**Status:** Approved — ready for implementation plan
**Date:** 2026-05-13
**Supersedes:** [2026-05-13-clip-pipeline.md](./2026-05-13-clip-pipeline.md) (manual-timestamp v1)

## 1. What this is

You hand the pipeline one recording. It scans the whole thing, picks the best moments, grades them, decides what aspect ratio fits each one based on shot composition, and renders each surviving clip in 3 caption-styling variants × 1–2 aspect ratios. No manual timestamp picking. Captions are still verbatim from whisper.

## 2. Non-goals

- Hand-picking timestamps. v1 had that; v2 does not.
- LLM-written overlay copy (hook / chyron / CTA). Removed entirely — those rendered badly in v1.
- Cloud APIs. Everything is local: whisper, Ollama, opencv, ffmpeg, yt-dlp.
- Auto-publishing.
- Music, b-roll, motion graphics, ken-burns.

## 3. End-to-end flow

```
source (file or YouTube URL)
  ↓
[1] Full-video whisper transcription            → words.json
[2] Audio energy analysis (local backend only)  → energy.json
[3] Candidate detection + grading (pluggable)   → graded.json
[4] Top-N selection                             → selected.json
[4.5] Framing + motion analysis (OpenCV)        → framing.json
[5] Render: each clip × 3 styles × 1–2 aspects  → out/<id>_<style>_<aspect>.mp4
[6] Verify each output                          → report.json
```

Every stage writes its output JSON to `out/<dir>/.work/`. Reruns skip cached stages — `cut.py` is idempotent.

**Stage 3 is pluggable.** It has two backends (`twelvelabs` and `local`) that produce the same `graded.json` schema. Downstream stages don't care which backend ran.

## 4. Inputs

### 4.1 Job file (`jobs/*.json`) — collapsed

```json
{
  "source": "https://www.youtube.com/watch?v=...",
  "source_type": "youtube",
  "output_dir": "out/myjob",
  "top_n": 3,
  "detection_backend": "auto",
  "llm_model": "llama3.1:8b",
  "whisper_model": "small",
  "variants": ["bloomberg", "tiktok_bro", "documentary"]
}
```

`detection_backend`: `"twelvelabs"` | `"local"` | `"auto"` (default).
- `auto`: use `twelvelabs` if `TWELVELABS_API_KEY` env var is set, else `local`.

CLI override: `--source PATH_OR_URL` (already implemented in v1).

No `clips` array. Timestamps are auto-detected.

### 4.2 Variants file — kept

`variants.json` stays as-is. The following fields are **unused but preserved** (in case overlays come back later, correctly positioned):
- `llm_persona`
- `chyron_uppercase`
- `overlay_style.*`

The fields that drive rendering: `caption_mode`, `subtitle_style.*`.

### 4.3 Optional keyword list (`keywords.json`)

Used in detection Stage 3. Plain JSON list of hype words and profanity to score. Pre-populated with reasonable defaults; user-editable.

```json
{
  "hype": ["wild", "crazy", "damn", "no way", "actually", "literally"],
  "profanity": ["shit", "fuck", "bitch", "ass"]
}
```

## 5. Pipeline stages — detail

### 5.1 Stage 1: Full-video whisper transcription

- One pass over the entire source with `faster-whisper` (word timestamps, vad, beam_size=5).
- Output: `words.json` — list of `{text, start, end}` covering the whole recording.
- Cached by source-file mtime + whisper-model name.

### 5.2 Stage 2: Audio energy analysis (`local` backend only)

Skipped entirely when `detection_backend == twelvelabs`. When backend is `local`:
- `ffmpeg -i source -af "loudnorm=print_format=json" -f null -` for global stats.
- Then `ffmpeg -i source -af astats=metadata=1:reset=0.1 -f null -` (or equivalent rolling RMS) to get a per-100ms RMS value series.
- Output: `energy.json` — list of `{t, rms_db}` pairs.

### 5.3 Stage 3: Candidate detection + grading (pluggable backend)

Both backends produce the same `graded.json` schema:

```json
[
  { "id": "01", "start": 123.4, "end": 148.7, "score": 0.87,
    "rationale": "high-energy reaction with quotable line", "topic": "..." }
]
```

Downstream stages only read `graded.json` — they don't know or care which backend produced it.

#### 5.3a Backend: `twelvelabs` (default when API key present)

1. Compute SHA-256 of the source file (or canonical URL); use as cache key.
2. If not cached, upload source to TwelveLabs and create an index. Store the returned `video_id` in `.work/twelvelabs.json` keyed by source hash.
3. Run a small set of search/classify queries against the indexed video. Initial queries (tunable in code):
   - `"high energy moment"`, `"emotional reaction"`, `"key insight"`, `"punchline or joke"`, `"intense delivery"`
4. Collect ranked moments with their confidence scores. Deduplicate overlapping moments (keep highest-scoring of any pair with >50% overlap).
5. Snap boundaries to whisper sentence boundaries (Stage 1 still runs locally — required for verbatim captions regardless of detection backend).
6. Cap clip length at 15s minimum, 45s maximum. Drop moments below confidence threshold (default 0.55).
7. Map TwelveLabs response → `graded.json` schema. `score` = TwelveLabs confidence; `rationale` = the query that surfaced it.

#### 5.3b Backend: `local` (fallback)

Algorithmic candidate scoring + Ollama grading, as previously specced:

Slide a 5-second window with 1-second stride. For each window:

| Signal | How |
|---|---|
| `energy_z` | mean RMS in window, z-score over full-recording baseline |
| `excl_density` | `!` + `?` count per 100 words |
| `hype_hits` | matches against `keywords.json` hype list, normalized by word count |
| `profanity_hits` | same but profanity list |
| `wps_z` | words-per-second z-score |

`window_score = 0.4*energy_z + 0.2*excl_density + 0.2*hype_hits + 0.1*profanity_hits + 0.1*wps_z`.

Cluster adjacent windows above `median + 1.5σ`. Boundary-refine to whisper sentence edges (±3s lookaround). Cap 15–45s.

Then for each candidate, send Ollama the transcript snippet + signals; Ollama returns `{grade: 1-10, rationale, topic}`. Final `score` written to `graded.json` is `(grade / 10) * window_score_normalized`.

**Ollama robustness:** `llama3.1:8b` default (not `gemma:2b`). Timeout 300s. Retry once on timeout. Pre-warm with a dummy call before the loop.

### 5.4 Stage 4: Top-N selection

Sort by `score` from `graded.json`. Pick top `top_n` (default 3). Output: `selected.json`.

### 5.5 Stage 4.5: Framing + motion analysis (OpenCV)

OpenCV runs ONCE per selected clip and produces both **aspect selection** and **motion compensation** decisions. Designed assuming the **common case is full-body** content (creators sitting/standing in front of the camera at some distance, not webcam close-ups).

For each selected clip:
- Sample 12 evenly-spaced frames from the source clip range (full resolution, before any aspect crop).
- Run TWO detectors on each frame:
  - **Haar face cascade** (`haarcascade_frontalface_default.xml`) with lenient parameters: `scaleFactor=1.1, minNeighbors=3, minSize=(24, 24)`. The reduced `minSize` is critical — default `(30, 30)` misses small faces in full-body framing.
  - **HOG person detector** (`cv2.HOGDescriptor_getDefaultPeopleDetector()`). Required because full-body shots routinely fail face detection (face is too small / off-angle / partly occluded by a desk/computer/mic). The HOG detector finds the whole person.
- Compute signals:
  - **Subject signals:**
    - `face_present_pct` — % of frames with ≥1 face
    - `mean_face_area_pct` — average face-bbox / frame area, only over frames where face was present
    - `max_face_count` — max faces in any frame
    - `person_present_pct` — % of frames with ≥1 HOG person bbox
    - `mean_person_height_pct` — average person-bbox height / frame height (proxy for shot tightness; full-body ≈ 70–95%, mid-shot ≈ 50–70%, close-up ≈ 30–50%)
  - **Motion signals:**
    - `frame_diff_mean` — mean absolute pixel difference between adjacent sampled frames (downsampled to 320px wide for speed)
    - `subject_position_variance` — variance of subject bbox center (x, y) across frames; uses face bbox if face detected, else person bbox
    - `subject_center_xy` — mean subject bbox center; **face bbox preferred** (more stable), **person bbox fallback** (when face not detected), **image center fallback** (when neither)

**Aspect classification (full-body-dominant defaults):**

| Class | Rule (checked top-down, first match) | Aspects rendered |
|---|---|---|
| `multi_subject` | `max_face_count ≥ 2` AND `face_present_pct ≥ 60` | `16:9` + `1:1` (square fits two faces; vertical would crop one) |
| `talking_head` | `face_present_pct ≥ 60` AND `mean_face_area_pct ≥ 12` | `9:16` + `1:1` (close-up — wastes width in 16:9) |
| `full_body` | `person_present_pct ≥ 70` AND (`face_present_pct < 60` OR `mean_face_area_pct < 12`) | `9:16` + `16:9` (the creator-content sweet spot — covers TikTok/Reels/Shorts + YouTube long-form) |
| `mid_shot` | `face_present_pct ≥ 60` AND `mean_face_area_pct < 12` AND `person_present_pct ≥ 70` | `9:16` + `1:1` (treat as a "tighter full-body"; vertical works, square works, 16:9 has too much side air) |
| `no_subject` | else | `16:9` only (preserve original framing — safest default) |

**Important:** `full_body` is the expected common case for client work. The detector should reach it whenever a person is reliably present and either their face is too small for `talking_head` OR face detection failed entirely (very likely on full-body shots).

**Motion compensation:**

A clip is `static` if:
- `frame_diff_mean < 6.0` (mean abs pixel diff, 0–255 scale), AND
- `subject_position_variance < 200` (pixels² at source resolution; uses whichever subject signal is available)

If `static` AND class is in `{talking_head, mid_shot, full_body}`:
- Apply **subject-centered Ken Burns zoom** during render (Stage 5): zoom from 100% to 108% over the clip duration, centered on `subject_center_xy` (face-preferred, person-fallback). ffmpeg `zoompan` filter with the center expressed as a fraction of the input frame.
- For `multi_subject` and `no_subject`, no zoom — multi-face zoom would crop someone out; no-subject zoom has no reliable anchor.

If NOT `static`, no zoom — the source already has visible motion.

**Frozen-frame escape hatch:** if `frame_diff_mean < 0.5` across the entire clip's sampled frames, the source itself is a still image (encoding bug, dead stream). The clip is **dropped** from the render queue and logged in `report.json`. Stage 6 verification is the safety net; this is the early detector.

Output: `framing.json` — per clip:
```json
{
  "id": "01",
  "class": "full_body",
  "static": true,
  "zoom": { "enabled": true, "start": 1.00, "end": 1.08, "center_x": 0.51, "center_y": 0.42 },
  "aspects": ["9:16", "16:9"],
  "signals": {
    "face_present_pct": 25, "mean_face_area_pct": 3.1,
    "person_present_pct": 92, "mean_person_height_pct": 78,
    "frame_diff_mean": 4.2, "subject_position_variance": 130
  }
}
```

**User may edit this file** to force aspects, toggle zoom, or change the zoom range before re-running the render stage. Common override: bump `aspects` to `["9:16", "1:1", "16:9"]` for a clip you want delivered to every platform.

### 5.6 Stage 5: Render

For each selected clip × each style variant × each aspect for that clip:
1. Extract the clip from source (re-encode, frame-accurate). Cached.
2. Build the `.ass` caption file from the whisper words, in the variant's style. Cached.
3. Build the `-vf` filter chain in this order:
   - If `framing.zoom.enabled`, prepend `zoompan=...` zooming `zoom.start` → `zoom.end` over clip duration, centered on `(zoom.center_x, zoom.center_y)` as fractions of the input frame. Set `fps=source_fps` and a duration in frames matching the clip to avoid the default 25fps stutter.
   - Then aspect crop:
     - `9:16` → `scale=-2:1920,crop=1080:1920`
     - `1:1` → `scale=-2:1080,crop=1080:1080`
     - `16:9` → `scale=1920:-2,crop=1920:1080`
   - Then `ass=<caption_file>`. No drawtext.
4. Encoder: `libx264 -preset medium -crf 20 -c:a aac -b:a 192k -movflags +faststart`.

Output: `out/<dir>/<clip_id>_<style>_<aspect>.mp4`.

### 5.7 Stage 6: Verification

Before declaring the job done, for every rendered MP4:
- `ffprobe` shows: duration ≥ clip span – 0.5s, ≥1 video stream, ≥1 audio stream.
- Extract 3 frames (10%, 50%, 90% of duration). Pixel-diff each adjacent pair using mean absolute difference. If MAD < threshold across both pairs → **flag as still-image bug**.
- Caption-verbatim check (existing `tools/verify_captions.py`).

Failures listed in `report.json` and the job exits non-zero.

## 6. Outputs

```
out/<dir>/
  <id>_bloomberg_9x16.mp4
  <id>_bloomberg_1x1.mp4
  <id>_tiktok_bro_9x16.mp4
  <id>_tiktok_bro_1x1.mp4
  <id>_documentary_9x16.mp4
  <id>_documentary_1x1.mp4
  ... (etc per selected clip, only aspects the framing detector picked)
  report.json
  .work/                       (cached intermediates; kept with --keep-temp)
```

Per-clip count: 3 styles × (1 or 2 aspects) = **3 or 6 MP4s per clip**.
Default `top_n: 3` → 9–18 MP4s per job.

## 7. Tooling / dependencies

- Python 3.11+ (in venv)
- `faster-whisper`, `yt-dlp`, `opencv-python`, `requests`, `twelvelabs` (pinned in `requirements.txt`)
- System: `ffmpeg`
- Optional service: `ollama serve` running, with `llama3.1:8b` pulled. Only required when `detection_backend` resolves to `local`.
- Optional env var: `TWELVELABS_API_KEY`. When set, `auto` backend resolves to `twelvelabs`.
- Fonts: macOS-default Helvetica/Impact/Georgia, override via `variants.json` per role.

## 8. Constraints / invariants

- **Captions are verbatim**, sourced from whisper Word objects. No backend (local or cloud) is allowed to influence caption text.
- **Detection is pluggable but its output schema is fixed.** Backends produce `graded.json` in the same shape; downstream stages are backend-agnostic.
- **Each stage is idempotent and cached.** Killing the pipeline mid-render and re-running resumes from the failed step. TwelveLabs uploads are cached by source-file SHA-256 so a re-run never re-uploads.
- **Hard-fail over silent fallback.** Missing source, no speech transcribed, no candidates above threshold, no `TWELVELABS_API_KEY` when backend is explicit `twelvelabs`, no Ollama running when backend is explicit `local`, framing detector returns nothing → exit non-zero with a clear message.
- **Verification gates completion.** Stage 6 must pass or the job exits non-zero, even if Stage 5 produced files.

## 9. CLI

```
python cut.py <job.json> [--source PATH_OR_URL] [--variants <variants.json>]
              [--top-n N] [--detection-backend twelvelabs|local|auto]
              [--keep-temp] [--resume-from STAGE]
```

- `--top-n` overrides `top_n` in the job file.
- `--detection-backend` overrides `detection_backend` in the job file. Useful for forcing `local` to compare quality side-by-side.
- `--resume-from` skips earlier stages and uses cached `.work/` outputs (defaults to detecting the first uncached stage).

## 10. Acceptance criteria

1. **Auto-detect produces ≥ top_n candidates** scored above baseline, no manual timestamps required, on both backends.
2. **Backends are interchangeable.** Running the same source with `--detection-backend twelvelabs` vs `--detection-backend local` produces different `graded.json` content but the downstream pipeline is identical and both reach Stage 6 verification.
3. **Framing classification matches a human's eye** on a 5-clip eyeball test that includes at least 3 full-body shots. A full-body shot must not be classified as `no_subject`.
4. **All rendered MP4s play as moving video.** Static clips (any class except `multi_subject`/`no_subject`) get subject-centered Ken Burns zoom applied in Stage 5. Stage 6 verification exits non-zero if any output is still a frozen frame.
5. **All captions are verbatim.** `tools/verify_captions.py` returns 0 for every `<id>.<style>.ass` against its transcript.
6. **Job is idempotent.** Re-running the same `cut.py jobs/myjob.json` after partial failure skips completed stages. TwelveLabs uploads never repeat on a cached source.
7. **Default `top_n=3` job on a ~10-minute source completes** in under 30 min on Apple Silicon with `whisper_model: small` (excluding first-run TwelveLabs upload + index time on `twelvelabs` backend).

## 11. Out of scope (explicit)

- Drawtext overlays (hook/chyron/CTA). Removed in v2.
- Cloud LLMs for grading copy. (Different from the TwelveLabs detection backend, which is in scope.)
- YOLO / DETR / vision-transformer subject detection. (Haar face + HOG person is the v2 ceiling. Revisit if HOG fallback misclassifies frequently.)
- Music, b-roll, transitions, multi-source comp, animated lower-thirds.
- Motion beyond the static-clip Ken Burns zoom. (No pans, no auto-cuts mid-clip, no speaker-tracked reframing.)
- Per-clip manual review UI.
- TwelveLabs's other features (semantic search, custom classifiers, embedding export). We only call the moment-detection endpoints.

## 12. Migration from v1

v1's `cut.py` becomes a near-total rewrite. Salvaged code:
- `parse_time`, `Word`, `ClipSpec` dataclasses (still useful)
- ASS caption builder (`build_ass`, `group_phrases`, `ass_escape`, `fmt_ass_time`)
- Source acquisition (`ensure_source`)
- Clip extraction (`extract_clip`)
- Aspect filter map (`aspect_to_filter`)

Removed code:
- `llm_overlays`, `drawtext_filter`, `_filter_quote`, `font_path`, `FONT_FALLBACK`
- `escape_ass_filter_path` still used by ass-filter call
- The drawtext lines inside `render_variant`

Existing `tools/check_font_path.py` and `tools/check_drawtext_escape.py` become obsolete; remove.

## 13. Open risks

- **TwelveLabs free-tier ceiling unknown at spec time.** Their free quota historically covered a few hundred hours of indexing per year. For sustained client work, a paid tier may be required; the math (likely <$10/source video) is still trivial relative to client invoice. Build the backend so a quota error fails gracefully with "fall back to `--detection-backend local`" guidance in the error message.
- **TwelveLabs upload time** counts against total wall-clock. A 10-minute source takes ~1–3 minutes to upload + index on first run. Subsequent runs hit the cache.
- **Local backend quality ceiling.** Algorithmic + Ollama detection is realistically 60–70% as good as TwelveLabs for "exciting moment" detection. For client work the user should prefer `twelvelabs`; `local` is the fallback / offline mode.
- **Ollama timeouts (local backend only):** v1 hit a 180s timeout. v2 uses `llama3.1:8b` (bigger but more reliable on Mac with Metal), 300s timeout, 1 retry, and pre-warm.
- **Face cascade misses on full-body shots.** The HOG person detector is the planned backstop, but HOG itself misses on partial occlusion (person behind a desk/computer — exactly the Akademiks case). If both detectors miss reliably, the clip falls to `no_subject` and renders only `16:9`. Mitigation: `framing.json` is user-editable to force an aspect set.
- **Sentence-boundary refinement may extend clips past the energy peak:** initial heuristic caps lookahead at ±3s and clip length at 45s max.
- **Zoom thresholds are calibrated guesses.** `frame_diff_mean < 6.0` and `subject_position_variance < 200` may misfire on noisy webcam footage (graininess registering as motion) or unusually still studio shots. Mitigation: thresholds are constants in code, and `framing.json` is editable to flip `zoom.enabled` per clip before re-render.
