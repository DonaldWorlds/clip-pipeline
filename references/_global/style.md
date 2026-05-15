# Global Style Spec — Short-Form Clip Editorial Rules

This file is the source of truth for editorial decisions in the clip pipeline.
The editorial/rules.py module parses this file at startup; every render
decision is grounded in a rule from here.

You can edit this file at any time. The next pipeline run picks up changes
automatically. No code edits required.

---

## Captions

- captions centered horizontally
- avoid facecam overlap (captions must not cover detected faces)
- max 2 lines
- minimum font size 70 px
- highlight keywords yellow
- subtitles must stay inside safe margins (5% inset from frame edges)

## Pacing

- something visual changes every 2 seconds (zoom, cut, motion, caption update)
- no silence longer than 1 second
- hook in the first 0.5 seconds — something happens immediately to grab attention
- zoom on loud moments (audio peaks)
- aggressive zooms acceptable for reaction-style content
- rapid cuts encouraged

## Clip Length

- minimum 5 seconds
- maximum 30 seconds
- short-form vertical content is the default

## Encoder Profile

- output 1080×1920 for vertical (9:16)
- pixel format yuv420p
- codec h264
- fps 60 when source supports it, otherwise match source fps
- always shortest=1 when combining video streams

## Creator Style Conventions

### Reaction / commentary content
- aggressive zooms
- loud captions (high contrast, bold weight)
- high motion
- rapid cuts
- reaction-focused — speaker reaction is the focal point

### Talking-head content (phone-vertical, single speaker)
- facecam-as-content layout (the whole frame is the speaker)
- captions in lower third
- minimal cuts, rely on zoom and caption animation for movement

### Streamer / screen-capture content (overlay class)
- preserve the small webcam region as the subject
- composite with blurred background to fill aspect ratio
- ESRGAN upscale the webcam region

## Known Issues to Avoid

### OBS / streamer black-frame bug
- Trim the first 0.5 seconds of OBS recordings before processing.
- ffmpeg pattern: `-ss 0.5 -i input.mp4 ...`

### Caption safe-zone violations
- Reduce subtitle width to 80% of frame width on vertical renders.

## Specific Reference: kai_fast_zoom style
- zoom every 1-2 seconds
- captions centered
- hard cuts (no fades)
- loud emotional pacing
- hook immediately (first frame establishes the action)
