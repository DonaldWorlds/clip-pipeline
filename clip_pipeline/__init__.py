"""clip_pipeline — auto-detect, multi-aspect clip rendering for short-form video.

Stages, in order:
  1. whisper.transcribe          — full-video word-level transcription
  2. detect.energy               — audio energy stats (local backend only)
  3. detect.find_candidates      — pluggable backend produces graded.json
  4. select.top_n                — pick the N highest-scoring candidates
  4.5. framing.analyze           — face + HOG person + motion → aspect & zoom decisions
  5. render.render_all           — ffmpeg renders each clip × style × aspect
  6. verify.run                  — ffprobe + still-frame + caption checks
"""
