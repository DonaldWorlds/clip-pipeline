# Finance Creator Clip Pipeline — Spec

**Status:** Draft — awaiting approval
**Date:** 2026-05-13
**Entry point:** `cut.py`

## 1. Goal

Take one source video (local file or YouTube URL) plus a manually-authored list of 5 timestamp ranges, and emit 15 finished short-form videos: each of the 5 clips rendered in 3 stylistic variants (Bloomberg-clean, TikTok-finance-bro, Documentary-sober).

Every variant of a given clip uses the **same verbatim transcript** for captions. Only the visual styling and the editorial overlay text (hook / chyron / CTA) change between variants.

## 2. Non-goals

- Automatic clip selection / "find the best moments." Timestamps are human-picked.
- Cloud transcription, cloud LLMs, or any paid API. Local-only (`faster-whisper` + Ollama).
- Music, b-roll, transitions, multi-track editing. Single source clip in, single rendered MP4 out.
- Paraphrased / "punched up" captions. Captions are verbatim by construction.
- Auto-publishing to social platforms.

## 3. User flow

1. User picks a source (local file path or YouTube URL) and writes a job JSON describing it.
2. User scrubs the source themselves and writes 5 clip entries (`id`, `start`, `end`, `topic`).
3. User runs `python cut.py jobs/<job>.json`.
4. Pipeline produces `out/<clip_id>_<variant>.mp4` × 15 in the output dir.

The user is the editor; the pipeline is the renderer.

## 4. Inputs

### 4.1 Job file (`jobs/*.json`)

```json
{
  "source": "https://youtube.com/watch?v=...",        // or "/path/to/file.mp4"
  "source_type": "youtube",                            // or "local"
  "output_dir": "out/2026-05-13-fed-presser",          // optional, defaults to "out"
  "aspect": "9:16",                                    // 9:16 | 1:1 | 16:9
  "llm_model": "llama3.1:8b",                          // optional, Ollama model tag
  "whisper_model": "large-v3",                         // optional
  "variants": ["bloomberg", "tiktok_bro", "documentary"], // optional; default = all
  "clips": [
    { "id": "01_powell_pivot", "start": "00:12:34", "end": "00:13:07", "topic": "Powell signals slower hikes" },
    { "id": "02_yields_blowout", "start": 845.0,   "end": 878.5,    "topic": "10Y yield curve reaction" }
    // ... 5 total
  ]
}
```

- `start` / `end` accept either seconds (number) or `HH:MM:SS` / `MM:SS` strings.
- If `clips` count ≠ 5, emit a warning but proceed (the user is in control).
- `topic` is editorial framing for the LLM only — it is never injected into captions.

### 4.2 Variants file (`variants.json`)

Locked. Three keys: `bloomberg`, `tiktok_bro`, `documentary`. Each variant defines:
- `llm_persona` — system-prompt voice for hook/chyron/CTA
- `caption_mode` — `lower_third` | `karaoke_word` | `subtitles`
- `chyron_uppercase` — bool
- `subtitle_style` — ASS V4+ style fields (font, size, colors, alignment, margins, border)
- `overlay_style` — drawtext params for hook / chyron / CTA (font, size, color, box)

The exact contents are the file already at the project root and are out of scope to change in this spec.

## 5. Pipeline stages

For each clip × for each variant:

### 5.1 Source acquisition
- `source_type: local` → use path as-is.
- `source_type: youtube` → `yt-dlp -f "bestvideo[height<=1080]+bestaudio/best" --merge-output-format mp4` into `<output_dir>/.work/source.mp4`. Cached: skip if file exists.

### 5.2 Clip extraction
- `ffmpeg -ss <start> -to <end> -i <source> -c:v libx264 -preset veryfast -crf 18 -c:a aac -b:a 192k` → `<work>/<clip_id>.mp4`.
- Re-encode (not stream-copy) so cuts are frame-accurate.
- Cached per clip id: skip if file exists.

### 5.3 Transcription (once per clip, shared across variants)
- `faster-whisper` with `word_timestamps=True`, `vad_filter=True`, `condition_on_previous_text=False`, `beam_size=5`.
- Default model `large-v3`; overridable per job.
- Output: list of `Word(text, start, end)`. The transcript file is written to `<work>/<clip_id>.transcript.txt` for debugging.
- Empty result is a hard error.

### 5.4 LLM overlays (per variant)
- Local Ollama at `http://localhost:11434/api/generate`, `format=json`.
- System prompt forbids paraphrasing the transcript and constrains lengths: hook 3–7 words, chyron 2–4 words, CTA 3–6 words.
- User prompt provides persona, editorial topic, and the verbatim transcript labeled "FOR CONTEXT ONLY — never paraphrase."
- Validation: all three keys must be non-empty strings. Otherwise hard error (so the user notices, not a silent fallback).
- Output cached at `<work>/<clip_id>.<variant>.overlays.json`.

### 5.5 Caption track (per variant)
- Build a `.ass` file directly from whisper Word objects. The LLM never touches caption text.
- Caption mode driven by variant's `caption_mode`:
  - `karaoke_word` → one Dialogue line per word at its `[start, end]`, full-screen via the Highlight style (alignment 5).
  - `lower_third` → phrases of ≤ 7 words or ≤ 3.0s, Base style, alignment 2 (bottom-center) with `margin_v` from style.
  - `subtitles` → phrases of ≤ 12 words or ≤ 4.5s, Base style.
- ASS file path is escaped for use in ffmpeg's `ass=` filter (colons and single quotes).

### 5.6 Final render (per variant)
- Single `ffmpeg` pass:
  1. Aspect/crop filter (`scale=...,crop=...`) for `9:16` / `1:1` / `16:9`.
  2. `ass=<caption_file>` for verbatim captions.
  3. Three `drawtext` overlays — hook (top, y ≈ 8%), chyron (above CTA, `y=h-th-200`), CTA (bottom, `y=h-th-60`). Each in a filled box with variant-specific colors.
- Encoder: `libx264 -preset medium -crf 20 -c:a aac -b:a 192k -movflags +faststart`.
- Output: `<output_dir>/<clip_id>_<variant>.mp4`.

## 6. Outputs

- `<output_dir>/<clip_id>_<variant>.mp4` — the deliverable, 5 × 3 = 15 files per job.
- `<output_dir>/.work/` — intermediates (source.mp4 cache, extracted clips, transcripts, overlay JSON, ass files). Deleted at end unless `--keep-temp`.

## 7. Tooling / dependencies

- Python 3.11+
- System binaries on `$PATH`: `ffmpeg`, `yt-dlp`
- Python pkgs: `faster-whisper`
- Local services: Ollama running, with the configured model already pulled (e.g. `ollama pull llama3.1:8b`)
- Fonts: macOS system locations for Helvetica / Impact / Georgia. Per-variant override via `overlay_style.<role>_fontfile` if needed on Linux.

## 8. Constraints / invariants

- **Captions are verbatim.** They are built from `Word` objects directly; the LLM is structurally unable to influence caption text.
- **Each clip is transcribed exactly once**, then the same Word list is reused across all variants.
- **Hard-fail over silent fallback.** Missing source, no speech transcribed, malformed LLM JSON, unknown variant, unsupported aspect → exit non-zero with a clear message. We are running locally with a human watching.
- Intermediate artifacts are cached by filename so reruns after a single-variant tweak are cheap.

## 9. CLI

```
python cut.py <job.json> [--variants <variants.json>] [--keep-temp]
```

- `--variants` defaults to `./variants.json`.
- `--keep-temp` retains `.work/` for inspection.

## 10. Acceptance criteria

1. Given a YouTube URL job with 5 valid clips and Ollama running, the pipeline produces 15 MP4s without manual intervention.
2. Each variant's captions contain ONLY words that appear in the whisper transcript (byte-for-byte, modulo whitespace) — verified by inspecting the `.ass` for that clip.
3. The same clip across 3 variants has visibly different fonts/colors/overlay text but identical spoken content.
4. Re-running the same job is idempotent: clip extraction and transcription do not repeat unless `.work/` is cleared.
5. A malformed clip range, missing source, or Ollama-down condition exits non-zero before any video is written.

## 11. Out of scope (explicit)

- Burning a watermark, logo, or progress bar.
- Auto-thumbnails or social-platform metadata.
- Multi-source compilations (one source per job).
- A GUI. The job file is the UI.
- Tests beyond the acceptance criteria above (this is a single-operator tool).
