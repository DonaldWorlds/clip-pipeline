# Clip Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the draft `cut.py` from "works on my whiteboard" to "runs end-to-end on a real job and meets every acceptance criterion in the spec."

**Architecture:** The pipeline structure in `cut.py` is correct. This plan does NOT redesign it. It adds the missing pieces around it (deps, example job, README), fixes two latent bugs that violate the spec's hard-fail invariant, adds a verbatim-caption verifier (acceptance criterion #2), and ends with a manual smoke run that exercises the full pipeline.

**Tech Stack:** Python 3.11+, `faster-whisper`, `ffmpeg`, `yt-dlp`, local Ollama. No pytest — per the spec, this is a single-operator tool. Each task ships a self-verifying script or a manual check.

**Reference:** Spec at [docs/superpowers/specs/2026-05-13-clip-pipeline.md](../specs/2026-05-13-clip-pipeline.md).

---

## File map

- Create: `requirements.txt` — pinned Python deps
- Create: `README.md` — setup + run instructions
- Create: `jobs/example.json` — copy-pasteable starter job
- Create: `tools/verify_captions.py` — checks `.ass` words ⊆ transcript words (AC2 enforcement)
- Create: `tools/check_drawtext_escape.py` — self-test for drawtext escaping (Task 4)
- Create: `tools/check_font_path.py` — self-test for font_path hard-fail (Task 3)
- Modify: `cut.py:306-311` — `font_path()` hard-fails on missing font
- Modify: `cut.py:314-326` — `drawtext_filter()` escapes commas and brackets, fixes backslash order
- Modify: `.gitignore` — ignore `out/`, `__pycache__`, `.venv`

---

## Task 1: Project bootstrap — requirements, README, .gitignore

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `.gitignore`

- [ ] **Step 1: Create `requirements.txt`**

```
faster-whisper==1.0.3
```

(That is the only Python dep — `ffmpeg`, `yt-dlp`, and Ollama are system-level. `urllib` and `json` are stdlib.)

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
out/
.work/
.DS_Store
```

- [ ] **Step 3: Create `README.md`**

```markdown
# clip-pipeline

Finance creator clip pipeline. Takes one source video + 5 hand-picked timestamps and emits 15 finished short-form videos (5 clips × 3 visual variants: Bloomberg-clean, TikTok-finance-bro, Documentary-sober).

Captions are verbatim from `faster-whisper`. Overlays (hook / chyron / CTA) are generated per-variant by a local Ollama model. The LLM never touches caption text.

See [docs/superpowers/specs/2026-05-13-clip-pipeline.md](docs/superpowers/specs/2026-05-13-clip-pipeline.md) for the full spec.

## Setup

System binaries:

```bash
brew install ffmpeg yt-dlp ollama
ollama serve &              # leave running
ollama pull llama3.1:8b
```

Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On first run, `faster-whisper` will download the chosen model (default `large-v3`, ~3 GB).

## Run

1. Copy `jobs/example.json` to `jobs/<your-job>.json` and edit: `source`, `source_type`, 5 clip entries.
2. Run:

```bash
python cut.py jobs/<your-job>.json
```

Outputs land in `out/` (or `output_dir` if set). Intermediates are cached under `out/.work/` and deleted on success unless `--keep-temp` is passed.

## Verify captions are verbatim

```bash
python tools/verify_captions.py out/.work/<clip_id>.<variant>.ass out/.work/<clip_id>.transcript.txt
```
```

- [ ] **Step 4: Commit**

```bash
git init -q  # if not already a repo
git add requirements.txt README.md .gitignore
git commit -m "chore: project bootstrap (deps, readme, gitignore)"
```

---

## Task 2: Example job file

**Files:**
- Create: `jobs/example.json`

- [ ] **Step 1: Write `jobs/example.json`**

```json
{
  "source": "https://www.youtube.com/watch?v=REPLACE_ME",
  "source_type": "youtube",
  "output_dir": "out/example",
  "aspect": "9:16",
  "llm_model": "llama3.1:8b",
  "whisper_model": "large-v3",
  "variants": ["bloomberg", "tiktok_bro", "documentary"],
  "clips": [
    { "id": "01", "start": "00:01:00", "end": "00:01:25", "topic": "REPLACE: editorial framing (LLM context only)" },
    { "id": "02", "start": "00:03:14", "end": "00:03:42", "topic": "REPLACE" },
    { "id": "03", "start": "00:05:00", "end": "00:05:28", "topic": "REPLACE" },
    { "id": "04", "start": "00:08:30", "end": "00:08:55", "topic": "REPLACE" },
    { "id": "05", "start": "00:11:05", "end": "00:11:33", "topic": "REPLACE" }
  ]
}
```

- [ ] **Step 2: Sanity-check it parses**

```bash
python -c "import json, sys; json.load(open('jobs/example.json')); print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add jobs/example.json
git commit -m "docs: add example job file"
```

---

## Task 3: Harden `font_path()` — hard-fail on missing font

**Why:** Spec §8 says "Hard-fail over silent fallback." Current `font_path()` silently falls back to Helvetica when the resolved font file doesn't exist — meaning a Linux user running the TikTok variant gets Helvetica instead of Impact and never notices. Convert to a hard error.

**Files:**
- Modify: `cut.py:306-311`
- Create: `tools/check_font_path.py`

- [ ] **Step 1: Write the check script that will fail today**

Create `tools/check_font_path.py`:

```python
"""Self-test for font_path hard-fail behavior."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cut import font_path  # noqa: E402

# Case 1: explicit path that does not exist -> must raise.
try:
    font_path({"hook_fontfile": "/nonexistent/font.ttf"}, "hook")
except FileNotFoundError as e:
    print(f"OK case 1: raised FileNotFoundError ({e})")
else:
    sys.exit("FAIL case 1: expected FileNotFoundError for missing explicit path")

# Case 2: named font with no system file -> must raise (not fall back).
try:
    font_path({"hook_font": "DefinitelyNotInstalled-XYZ"}, "hook")
except FileNotFoundError as e:
    print(f"OK case 2: raised FileNotFoundError ({e})")
else:
    sys.exit("FAIL case 2: expected FileNotFoundError for unknown font name")

print("all font_path checks passed")
```

- [ ] **Step 2: Run it — expect failure**

```bash
python tools/check_font_path.py
```

Expected: exits non-zero on case 1 (current code returns the string without checking existence).

- [ ] **Step 3: Update `font_path()` in `cut.py`**

Replace the body of `font_path` (currently lines 306-311) with:

```python
def font_path(overlay_style: dict, role: str) -> str:
    explicit = overlay_style.get(f"{role}_fontfile")
    if explicit:
        if not Path(explicit).is_file():
            raise FileNotFoundError(
                f"font file not found for {role}: {explicit}"
            )
        return explicit
    name = overlay_style.get(f"{role}_font", "Helvetica")
    candidate = FONT_FALLBACK.get(name)
    if not candidate or not Path(candidate).is_file():
        raise FileNotFoundError(
            f"no system font file resolved for {role}={name!r}; "
            f"set overlay_style.{role}_fontfile in variants.json"
        )
    return candidate
```

- [ ] **Step 4: Run the check — expect pass**

```bash
python tools/check_font_path.py
```

Expected: `all font_path checks passed`.

- [ ] **Step 5: Commit**

```bash
git add cut.py tools/check_font_path.py
git commit -m "fix: font_path hard-fails on missing font (spec §8)"
```

---

## Task 4: Harden `drawtext_filter()` escaping

**Why:** ffmpeg's filter syntax uses `,` to separate filters and `[` `]` for stream labels. The current escape function (cut.py:315-320) handles `\`, `'`, `:`, `%` but not `,`, `[`, `]`. A chyron like "FED, MAY 2026" or a CTA containing `[` will break the filter graph. Also: backslash escape must run first or it double-escapes the slashes added later.

**Files:**
- Modify: `cut.py:314-326`
- Create: `tools/check_drawtext_escape.py`

- [ ] **Step 1: Write the check script**

Create `tools/check_drawtext_escape.py`:

```python
"""Self-test for drawtext escaping."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cut import drawtext_filter  # noqa: E402

def check(text, must_contain_escaped, label):
    out = drawtext_filter(
        text, fontfile="/x.ttf", size=40, color="white",
        box_color="black@0.5", y="10",
    )
    for needle in must_contain_escaped:
        if needle not in out:
            sys.exit(f"FAIL {label}: expected {needle!r} in:\n{out}")
    print(f"OK {label}")

# Comma must be escaped so ffmpeg doesn't read it as a filter separator.
check("FED, MAY 2026", [r"\,"], "comma")
# Square brackets must be escaped so ffmpeg doesn't read them as stream labels.
check("BUY [NOW]", [r"\[", r"\]"], "brackets")
# Colon, single quote, percent still escaped.
check("up 5% — 9:30 'open'", [r"\%", r"\:", r"\'"], "colon/quote/percent")
# Backslashes in input should result in escaped backslashes, not eaten.
check(r"path\to\thing", [r"\\"], "backslash")

print("all drawtext_filter checks passed")
```

- [ ] **Step 2: Run — expect failure (comma + brackets not escaped today)**

```bash
python tools/check_drawtext_escape.py
```

Expected: `FAIL comma` (current code does not escape `,`).

- [ ] **Step 3: Update `drawtext_filter()` in `cut.py`**

Replace `drawtext_filter` (currently lines 314-326) with:

```python
def drawtext_filter(text, *, fontfile, size, color, box_color, y) -> str:
    # Order matters: escape backslashes FIRST so we don't double-escape the
    # slashes we add below. ffmpeg's drawtext consumes its own backslashes,
    # so each metachar must be preceded by one.
    escaped = text.replace("\\", "\\\\")
    for ch in (":", "'", "%", ",", "[", "]"):
        escaped = escaped.replace(ch, "\\" + ch)
    return (
        f"drawtext=fontfile='{fontfile}':text='{escaped}'"
        f":fontsize={size}:fontcolor={color}"
        f":box=1:boxcolor={box_color}:boxborderw=20"
        f":x=(w-text_w)/2:y={y}"
    )
```

- [ ] **Step 4: Run — expect pass**

```bash
python tools/check_drawtext_escape.py
```

Expected: `all drawtext_filter checks passed`.

- [ ] **Step 5: Commit**

```bash
git add cut.py tools/check_drawtext_escape.py
git commit -m "fix: drawtext escapes comma/brackets and orders backslash first"
```

---

## Task 5: Caption verbatim verifier (acceptance criterion #2)

**Why:** Spec acceptance criterion #2 requires that captions contain ONLY words present in the whisper transcript. The code enforces this structurally (captions are built from `Word` objects), but we want a runnable verifier so the user can audit any `.ass` after the fact.

**Files:**
- Create: `tools/verify_captions.py`

- [ ] **Step 1: Create the verifier**

```python
"""Verify that every word in a .ass caption file is present in the whisper
transcript (verbatim, modulo punctuation and case).

Usage:
    python tools/verify_captions.py <clip_id.variant.ass> <clip_id.transcript.txt>

Exits 0 if every caption word maps to a transcript word, 1 otherwise.
"""
import re
import sys
from pathlib import Path


def normalize(token: str) -> str:
    return re.sub(r"[^\w']+", "", token).lower()


def ass_dialogue_words(ass_text: str) -> list[str]:
    words: list[str] = []
    for line in ass_text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        # Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
        # The Text field is the 10th column. Splits 9 times to preserve commas in Text.
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        text = parts[9]
        # Strip ASS override blocks like {\\an5} just in case.
        text = re.sub(r"\{[^}]*\}", "", text)
        text = text.replace("\\N", " ")
        for raw in text.split():
            n = normalize(raw)
            if n:
                words.append(n)
    return words


def transcript_words(text: str) -> set[str]:
    return {normalize(t) for t in text.split() if normalize(t)}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    ass_path, transcript_path = Path(sys.argv[1]), Path(sys.argv[2])
    cap_words = ass_dialogue_words(ass_path.read_text())
    trans = transcript_words(transcript_path.read_text())
    missing = [w for w in cap_words if w not in trans]
    if missing:
        print(f"FAIL: {len(missing)} caption word(s) not in transcript:")
        for w in missing[:20]:
            print(f"  {w!r}")
        return 1
    print(f"OK: all {len(cap_words)} caption words present in transcript")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Synthesize a tiny fixture to self-test the verifier**

```bash
mkdir -p /tmp/cv
cat > /tmp/cv/t.txt <<'EOF'
The Fed signaled slower hikes today.
EOF
cat > /tmp/cv/ok.ass <<'EOF'
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Base,,0,0,0,,Fed signaled slower
EOF
cat > /tmp/cv/bad.ass <<'EOF'
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Base,,0,0,0,,Fed signaled FASTER hikes today
EOF
python tools/verify_captions.py /tmp/cv/ok.ass /tmp/cv/t.txt
python tools/verify_captions.py /tmp/cv/bad.ass /tmp/cv/t.txt; echo "exit=$?"
```

Expected:
- First run: `OK: all 3 caption words present in transcript`, exit 0.
- Second run: `FAIL: 1 caption word(s) not in transcript: 'faster'`, exit 1.

- [ ] **Step 3: Commit**

```bash
git add tools/verify_captions.py
git commit -m "feat: caption verbatim verifier (spec AC #2)"
```

---

## Task 6: End-to-end smoke run + acceptance check

**Why:** This is the spec's acceptance gate. Everything above is preparation; this task is where we run the pipeline on real input and confirm criteria 1, 3, 4, 5 from the spec.

This task is **operator-driven**. The agent should run the commands, capture output, and stop at any failure for human inspection rather than papering over it.

**Files:** None modified; this is an acceptance test.

**Prereqs (verify before starting):**
- `ollama serve` is running and `ollama list` shows the model in `jobs/example.json` (default `llama3.1:8b`)
- `ffmpeg -version` works
- `yt-dlp --version` works
- A short test source exists. If the user hasn't provided one, ask them for either (a) a local mp4 path or (b) a public YouTube URL ≤ 3 minutes long. Do not pick one yourself.

- [ ] **Step 1: Build the smoke job file**

Copy `jobs/example.json` to `jobs/smoke.json` and edit:
- `source` / `source_type` per the user's input above
- Reduce clips to 2 short ranges (≤ 10s each) for fast turnaround during smoke
- `output_dir`: `out/smoke`

Show the resulting file to the user and ask them to confirm the ranges before running.

- [ ] **Step 2: Run the pipeline with `--keep-temp` so we can inspect intermediates**

```bash
python cut.py jobs/smoke.json --keep-temp
```

Expected: 2 clips × 3 variants = 6 mp4s in `out/smoke/`, intermediates in `out/smoke/.work/`.

If it fails, do NOT retry blindly. Read the error, fix root cause, re-run.

- [ ] **Step 3: Acceptance criterion #1 — 15 (here 6) MP4s exist**

```bash
ls -1 out/smoke/*.mp4 | wc -l
```

Expected: `6` (we used 2 clips for smoke; the spec's "15" assumes the full 5).

- [ ] **Step 4: Acceptance criterion #2 — captions are verbatim**

For each clip × variant:

```bash
for clip in 01 02; do
  for v in bloomberg tiktok_bro documentary; do
    python tools/verify_captions.py \
      out/smoke/.work/${clip}.${v}.ass \
      out/smoke/.work/${clip}.transcript.txt || exit 1
  done
done
```

Expected: every invocation prints OK and exits 0.

- [ ] **Step 5: Acceptance criterion #3 — visual variant diff**

Manual: open `out/smoke/01_bloomberg.mp4`, `out/smoke/01_tiktok_bro.mp4`, `out/smoke/01_documentary.mp4` and confirm visually:
- Same spoken words appear in captions across all three
- Fonts/colors clearly differ (Helvetica white lower-third vs Impact green karaoke vs Georgia subtitles)
- Hook/chyron/CTA text differs (LLM produced distinct copy per persona)

Ask the user to confirm before checking the box.

- [ ] **Step 6: Acceptance criterion #4 — idempotent re-run**

```bash
# Note timestamp on existing intermediates
ls -la out/smoke/.work/01.mp4 out/smoke/.work/01.transcript.txt
# Re-run
python cut.py jobs/smoke.json --keep-temp
# Confirm clip extraction did not re-run (mtime unchanged on 01.mp4)
ls -la out/smoke/.work/01.mp4
```

Expected: `01.mp4` mtime is unchanged. (Note: transcript and overlay JSON ARE regenerated currently — that's per the spec, only clip extraction and source download are cached. If this is wrong, flag it but don't fix without approval.)

- [ ] **Step 7: Acceptance criterion #5 — hard-fail paths**

Three induced failures, each should exit non-zero with a clear message:

```bash
# 5a. Unknown variant
python -c "import json; j=json.load(open('jobs/smoke.json')); j['variants']=['bogus']; json.dump(j, open('/tmp/bad1.json','w'))"
python cut.py /tmp/bad1.json; echo "exit=$?"

# 5b. Missing local source
python -c "import json; j=json.load(open('jobs/smoke.json')); j['source']='/no/such/file.mp4'; j['source_type']='local'; json.dump(j, open('/tmp/bad2.json','w'))"
python cut.py /tmp/bad2.json; echo "exit=$?"

# 5c. Ollama down — stop ollama serve, re-run smoke.json
#   (Operator step: run `pkill -f 'ollama serve'`, then re-run, then restart.)
```

Expected: each `exit` line shows a non-zero code, and stderr explains the cause (`unknown variants:`, `source file not found:`, connection error).

- [ ] **Step 8: Write a one-line acceptance note in the spec**

Append to `docs/superpowers/specs/2026-05-13-clip-pipeline.md`:

```markdown

---

## Acceptance log

- 2026-05-13: Smoke run on `jobs/smoke.json` passed AC1–AC5. Operator: <user initials>.
```

- [ ] **Step 9: Final commit**

```bash
git add docs/superpowers/specs/2026-05-13-clip-pipeline.md jobs/smoke.json
git commit -m "test: smoke run passed all acceptance criteria"
```

---

## Self-review

Spec coverage:
- §4 inputs — Task 2 (example job) + Task 1 (README)
- §5 stages — already in `cut.py`; Tasks 3 & 4 fix latent bugs in §5.6
- §6 outputs — Task 6 verifies
- §7 tooling — Task 1 README documents
- §8 hard-fail invariant — Task 3 (font), Task 6 (induced failures)
- §10 acceptance criteria — Task 5 (AC2), Task 6 (AC1, AC3, AC4, AC5)

No placeholders. No "TBD." Every step has exact code or exact commands.

Type consistency: `font_path` returns `str` everywhere; `drawtext_filter` keyword args unchanged; `verify_captions.py` uses only stdlib.
