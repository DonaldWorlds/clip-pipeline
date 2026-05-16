"""Stage 1: full-video word-level transcription with faster-whisper."""
from __future__ import annotations

import sys
from pathlib import Path

from ..config import Job, Word, cache_read, cache_write


def transcribe(job: Job, source_path: Path) -> list[Word]:
    """Returns word list for the entire source. Cached at
    <work>/words.<model>.json so re-runs skip whisper entirely."""
    cache = job.work_dir / f"words.{job.whisper_model}.json"
    if cache.exists():
        return [Word(**w) for w in cache_read(cache)]

    from faster_whisper import WhisperModel  # lazy: heavy import

    print(f"[whisper] transcribing full source with {job.whisper_model}", file=sys.stderr)
    model = WhisperModel(job.whisper_model, device="auto", compute_type="auto")
    segments, _ = model.transcribe(
        str(source_path),
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
        beam_size=5,
    )
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            text = w.word.strip()
            if not text:
                continue
            words.append(Word(text=text, start=float(w.start), end=float(w.end)))
    if not words:
        sys.exit("whisper produced no words from source")

    cache_write(cache, words)
    # Plain-text dump for human inspection / verify_captions.py.
    (job.work_dir / f"transcript.{job.whisper_model}.txt").write_text(
        " ".join(w.text for w in words)
    )
    return words


def words_in_range(words: list[Word], start: float, end: float) -> list[Word]:
    """Filter words whose midpoint falls inside [start, end]."""
    return [w for w in words if start <= (w.start + w.end) / 2 <= end]
