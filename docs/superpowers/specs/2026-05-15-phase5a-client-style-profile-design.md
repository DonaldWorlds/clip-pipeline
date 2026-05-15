# Phase 5a — ClientStyleProfile + Variants Matrix Collapse

**Status:** Draft — awaiting user review
**Date:** 2026-05-15
**Supersedes (partially):** [2026-05-13-clip-pipeline-v2-design.md](./2026-05-13-clip-pipeline-v2-design.md) §4.2, §5.6, §6
**Sibling spec (next):** Phase 5b — Moment Intelligence Layer (separate doc)

## 1. What this is

The pipeline stops having taste of its own. Every rendering decision — caption style, aspect, color grade, upscale, frame style, post-copy — is driven by a single `ClientStyleProfile` derived from that client's reference clips at ingest time. The pipeline becomes a delivery mechanism; the references are the brain.

Concretely: `cut.py --client fitness_creator_v1 source.mp4 --top-n 3` produces **N clips × M delivery_aspects MP4s**, where N is `top_n` and M is the number of aspects the client's references actually use. No variants, no frame-style matrix, no thumbnails by default.

## 2. Non-goals (for 5a)

- Improving moment selection / picking better moments. That's Phase 5b.
- Removing TwelveLabs or Pegasus. They stay as the candidate generator.
- Building a client-onboarding UI. Profile derivation happens at ingest in code.
- Migrating Phase 1–4 outputs. New clients get profiles; old jobs keep working through a legacy `variants.json` codepath until removed.

## 3. The ClientStyleProfile

Stored alongside the per-client RAG database (`references/<client>/profile.json`, also embedded in `rag.db` for retrieval-time use).

```json
{
  "client": "fitness_creator_v1",
  "derived_at": "2026-05-15T14:30:00Z",
  "n_references": 12,

  "delivery_aspects": ["9:16"],
  "typical_clip_length_s": 14.0,

  "caption": {
    "mode": "karaoke_word",
    "position": "lower_third",
    "font": "Impact",
    "font_px_at_1080": 96,
    "primary_color": "&H00FFFFFF",
    "highlight_color": "&H0000FF00",
    "outline_color": "&H00000000",
    "outline_width": 6,
    "y_frac": 0.78,
    "derivation_confidence": 0.72
  },

  "color_grade": {
    "brightness_target": 132.0,
    "saturation_target": 1.08,
    "contrast_target": 1.05,
    "color_temp": "warm"
  },

  "frame_style": "edge_to_edge",
  "use_upscaling": false,
  "generate_thumbnail": false,
  "generate_post_copy": true,

  "source_fingerprint_ids": ["fit01", "fit02", "fit03", "..."]
}
```

### 3.1 Derivation rules — where each field comes from

| Field | Source | Rule |
|---|---|---|
| `delivery_aspects` | reference video dimensions | unique aspects across all refs (rounded to {9:16, 1:1, 16:9}); fallback `["9:16"]` if no refs ingested |
| `typical_clip_length_s` | reference `duration_s` | median across refs |
| `caption.mode` | reference Pegasus `captions` description | text-classified against {`subtitles`, `lower_third`, `karaoke_word`}; tie → `karaoke_word` (the short-form default per `references/_global/style.md`) |
| `caption.position` | reference `EditorialSignature.caption_y_frac` | `lower_third` if median `y_frac` ∈ [0.65, 0.90], `center` if ∈ [0.40, 0.65], `top` if < 0.40 |
| `caption.font` | reference Pegasus `captions` description | text-classified against {Impact, Helvetica, Inter, Montserrat}; fallback Impact |
| `caption.font_px_at_1080` | reference `caption_font_px` | median, scaled to 1080-tall reference height |
| `caption.primary_color`, `highlight_color`, `outline_color`, `outline_width` | reference Pegasus `captions` description + OCR (if available) | mode value across refs; defaults from `style.md` if uncertain |
| `color_grade.brightness_target` | reference `VisualSignature.avg_brightness` | median across refs |
| `color_grade.saturation_target` | derived | computed from reference HSV mean S; default 1.0 if absent |
| `color_grade.contrast_target` | derived | derived from reference luminance histogram spread; default 1.0 |
| `color_grade.color_temp` | reference `VisualSignature.color_temp` | mode value ({warm, neutral, cool}) |
| `frame_style` | reference visual analysis | `inset` if ≥ 50% of refs show a centered subject region < 80% of frame height with blurred edges; else `edge_to_edge` |
| `use_upscaling` | reference visual sharpness | `true` if median reference Laplacian variance > threshold AND median resolution ≥ 1080p (proxy for "looks enhanced") |
| `generate_thumbnail` | hardcoded default | `false`. Override via `--with-thumbnail` CLI flag or profile edit |
| `generate_post_copy` | hardcoded default | `true` (cheap, still useful) |

Two derivation principles:
1. **No invention.** If references can't support a decision (n < 3 refs, or signal is too noisy), fall back to `references/_global/style.md` defaults — never to a pipeline-baked constant.
2. **The profile is user-editable.** `references/<client>/profile.json` can be hand-tuned. Re-running ingest regenerates it unless `--no-overwrite-profile` is passed.

## 4. End-to-end flow (delta from current)

```
ingest (rag/ingest.py)
  for each new reference under references/<client>/inbox/:
    fingerprint → rag.db (existing)
    + UPDATE references/<client>/profile.json   ← NEW
  ↓
cut.py --client <client> <source>
  load profile.json                              ← NEW
  Stage 1–4.5 unchanged
  Stage 5 render: ONE pass per (clip × profile.delivery_aspects[i])  ← was 3 styles × 2 aspects
  no thumbnail unless profile.generate_thumbnail
  no inset-blur unless profile.frame_style == "inset"
  upscale only if profile.use_upscaling
  ↓
Stage 6 verify
  technical checks unchanged
  editorial check now compares output fingerprint to client profile,
  not just to style.md
```

### 4.1 Render-stage changes (Stage 5)

- `clip_pipeline/render/grade.py`: `GRADE_FILTERS` dict (hardcoded per-variant filters) **deleted**. Replaced by `build_grade_filter(profile.color_grade) -> str` which constructs the ffmpeg expression from the profile's numeric targets.
- `clip_pipeline/render/enhance.py`: callsite no longer keyed on `framing.cls == "overlay"`. New gate: `if profile.use_upscaling: run(esrgan, gfpgan)`.
- Frame-style branch: remove the inset-blur composite from the default render path. Only emitted when `profile.frame_style == "inset"`.
- Aspect loop: render `for aspect in profile.delivery_aspects`, not for the hardcoded `[9:16, 1:1]` pair.

### 4.2 Thumbnail (Stage 5.5)

- Removed from default pipeline. `clip_pipeline/thumbnail.py` stays in the repo (Phase 1 already deleted enough; no need to over-prune).
- Invoked only when `profile.generate_thumbnail` is true OR `--with-thumbnail` CLI flag passed.

### 4.3 Validation (Stage 6)

- `clip_pipeline/validate/editorial.py`: extended to take a `ClientStyleProfile` argument. Existing checks against `references/_global/style.md` (mechanical rules-check, Pegasus vision-check skeleton) **stay**. NEW checks added:
  - rendered output's `caption_y_frac` within ±0.05 of `profile.caption.y_frac`
  - rendered output's `avg_brightness` within ±15 of `profile.color_grade.brightness_target`
  - rendered output's `color_temp` matches `profile.color_grade.color_temp`
  - rendered output's aspect ∈ `profile.delivery_aspects`
- Failures listed in `report.json` with which axis failed.

## 5. variants.json — what happens to it

- `variants.json` at the repo root **stops being read** by the new code path.
- Kept on disk for one release as a fossil; flagged with `DEPRECATED` comment at the top. Removed in Phase 6.
- The three named variants (bloomberg, tiktok_bro, documentary) no longer correspond to render passes. If someone wants a "Bloomberg-style" clip, they ingest Bloomberg-style references under a client name and the profile reflects them.

## 6. CLI

```
python cut.py --client <client> <source> [--source PATH_OR_URL]
              [--top-n N] [--detection-backend twelvelabs|local|auto]
              [--with-thumbnail] [--no-overwrite-profile]
              [--resume-from STAGE]
```

- `--client` becomes the primary axis. Profile is loaded from `references/<client>/profile.json`.
- `--with-thumbnail` overrides `profile.generate_thumbnail = false`.
- Missing profile (`--client` points to a name with no `profile.json`) → hard fail with message "no profile for client X; run `python -m clip_pipeline.rag.ingest --client X` first."

## 7. Outputs

```
out/<dir>/
  <id>_9x16.mp4                ← was <id>_bloomberg_9x16.mp4
  <id>_1x1.mp4                 ← only if 1:1 ∈ delivery_aspects
  report.json
  .work/
```

Default `top_n=3` with `delivery_aspects=["9:16"]` → **3 MP4s** (was 18).
`top_n=3` with `delivery_aspects=["9:16","1:1"]` → **6 MP4s**.

## 8. Acceptance criteria

1. A new client ingest (≥ 3 references) produces a valid `profile.json` covering all fields in §3.
2. `cut.py --client X source.mp4 --top-n 3` on a profile with `delivery_aspects=["9:16"]` produces exactly 3 MP4s — no variants suffix in filename, no thumbnails, no inset-blur duplicate.
3. Editing `profile.json` (e.g., bumping `delivery_aspects` to `["9:16","1:1"]`) and re-running produces 6 MP4s without re-deriving the profile.
4. `clip_pipeline/render/grade.py` no longer contains the `GRADE_FILTERS` dict; `enhance.py` no longer branches on `framing.cls`.
5. `validate/editorial.py` flags an output whose `caption_y_frac` is > 0.05 off-profile.
6. Removing all of `references/<client>/` for an unknown client and running `cut.py --client X` fails with a clear message (not a silent fallback).
7. A profile manually edited to `use_upscaling: false` produces no ESRGAN/GFPGAN invocation in the render logs, even on a clip whose framing class is `overlay`.

## 9. Out of scope (explicit)

- **Moment Intelligence layer.** Phase 5b. Stage 3 still runs Pegasus.analyze → keyword-search fallback as today. This spec does not change *which* moments get picked, only *how they're rendered*.
- New caption modes beyond the existing three.
- Removing TwelveLabs / Pegasus dependence.
- Removing `variants.json` (deferred to Phase 6).
- Per-platform delivery automation (TikTok/IG/YT upload).
- LUT-based color grading (`.cube` files).

## 10. Open risks

- **Reference-derivation thin-data ceiling.** With < 5 references the profile may be noisy (one outlier ref skews the median). Mitigation: hard floor of 3 references before profile generation; under 3 → fall back to `style.md` defaults across the board.
- **Caption-style classification from Pegasus text is fuzzy.** Pegasus describes captions in prose; the classifier maps that prose to {subtitles, lower_third, karaoke_word}. Misclassifications are likely on edge cases. Mitigation: log the classifier's confidence, dump to `profile.json` as `caption.derivation_confidence` so it's visible; profile is user-editable as the final escape hatch.
- **OCR-free caption-color extraction is unreliable.** Color comes from Pegasus's prose ("yellow highlights") which may be wrong. Mitigation: provide sane defaults from `style.md` and accept that color is the most-likely-to-need-manual-edit field.
- **Color-grade targets may not match perceptual look.** A brightness target of "132" doesn't capture what makes a reference "look right." This spec gets the macro right (warm vs cool, bright vs dim); fine-tuning is a Phase 6+ concern.
- **Editorial validation against profile + style.md may double-flag.** If style.md says captions live in `lower_third` and the profile derives `karaoke_word`, the validator could fire on both. Mitigation: profile overrides style.md when both apply; style.md is the floor (minimum rules), profile is the ceiling (per-client preferences).

## 11. Deliverables

- New: `clip_pipeline/profile.py` — dataclass + load/save/derive functions.
- Modified: `clip_pipeline/rag/ingest.py` — calls profile derivation after each new fingerprint.
- Modified: `clip_pipeline/render/grade.py`, `enhance.py`, top-level render loop in `cut.py`.
- Modified: `clip_pipeline/validate/editorial.py` — accepts profile, adds the 4 new checks.
- Modified: `cut.py` — `--client` flag promoted, `--with-thumbnail` added.
- Deprecated: `variants.json` (banner added, behavior removed).
