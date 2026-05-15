# Talking Head Vertical Render

Use:
- 1080x1920
- gameplay bottom
- facecam top
- h264
- yuv420p

Command Pattern:

ffmpeg -i gameplay.mp4 -i facecam.mp4 \
-filter_complex ...

Important:
- always use shortest=1
- set fps=60