"""Stage 5.7: social-media post copy generation per clip.

For each selected clip, ask TwelveLabs Pegasus (the multimodal LLM already
indexing the video for detection) to write platform-specific caption copy:

  - tiktok_caption     (1-2 punchy sentences, ≤100 chars)
  - instagram_caption  (3-4 sentences, more polished)
  - youtube_description (2-3 sentences, descriptive for SEO)
  - hashtags           (5-7 strings starting with '#')

Output: <out_dir>/<clip_id>_post.json — one file per clip.

This turns the deliverable from "here are 3 rendered MP4s" into "here are 3
MP4s, captions ready to paste, hashtags ready to use" — what content creators
actually want.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from .config import Candidate, Job, cache_read, cache_write


PROMPT = (
    "Write social-media post copy for this video clip. "
    "Return STRICT JSON ONLY, no markdown, no commentary, with exactly these keys:\n"
    '  "tiktok_caption": "1-2 punchy sentences, hook-first, 100 characters max"\n'
    '  "instagram_caption": "3-4 sentences, polished, includes a soft CTA"\n'
    '  "youtube_description": "2-3 sentences, descriptive for SEO and search"\n'
    '  "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"]\n'
    "No emojis. Hashtags must start with '#' and contain no spaces."
)


def _parse_response(text: str) -> dict | None:
    """Pegasus sometimes wraps JSON in markdown fences. Strip them, then load."""
    text = text.strip()
    # Strip ```json ... ``` fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first balanced JSON object in the text.
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _twelvelabs_video_id(job: Job) -> str | None:
    """Look up the cached TwelveLabs video_id for this source."""
    cache_path = job.work_dir / "twelvelabs_videos.json"
    if not cache_path.exists():
        return None
    try:
        cached = cache_read(cache_path)
    except Exception:
        return None
    return cached.get(job.source_hash)


def generate_for_clip(client, video_id: str, cand: Candidate) -> dict | None:
    """One Pegasus call per clip."""
    try:
        resp = client.analyze(
            video_id=video_id,
            model_name="pegasus1.5",
            prompt=PROMPT,
            start_time=cand.start,
            end_time=cand.end,
            temperature=0.4,
        )
    except Exception as e:
        print(f"[copy] {cand.id}: analyze failed: {e}", file=sys.stderr)
        return None

    text = getattr(resp, "data", None) or getattr(resp, "text", None) or str(resp)
    parsed = _parse_response(text)
    if not parsed:
        print(f"[copy] {cand.id}: could not parse response: {text[:200]!r}",
              file=sys.stderr)
        return None
    return parsed


def generate_all(job: Job, selected: list[Candidate]) -> dict:
    """Stage 5.7: produce per-clip post copy files. Returns a dict keyed by
    cand.id with the parsed JSON for inspection. Skips clips that already have
    a saved post.json."""
    video_id = _twelvelabs_video_id(job)
    if not video_id:
        print("[copy] no TwelveLabs video_id cached; skipping caption-copy generation",
              file=sys.stderr)
        return {}

    if not os.environ.get("TWELVELABS_API_KEY"):
        print("[copy] TWELVELABS_API_KEY not set; skipping caption-copy generation",
              file=sys.stderr)
        return {}

    from twelvelabs import TwelveLabs
    client = TwelveLabs(api_key=os.environ["TWELVELABS_API_KEY"])

    out: dict[str, dict] = {}
    for cand in selected:
        out_path = job.output_dir / f"{cand.id}_post.json"
        if out_path.exists():
            try:
                out[cand.id] = cache_read(out_path)
                continue
            except Exception:
                pass  # regenerate on parse error

        print(f"[copy] {cand.id}: generating post copy via Pegasus…",
              file=sys.stderr)
        parsed = generate_for_clip(client, video_id, cand)
        if parsed is None:
            continue
        cache_write(out_path, parsed)
        out[cand.id] = parsed
        # One-line summary so the operator sees the result.
        tiktok = parsed.get("tiktok_caption", "")
        tags = " ".join(parsed.get("hashtags", []))
        print(f"[copy]   {cand.id}: {tiktok!r} {tags}", file=sys.stderr)
    return out
