"""RAG Librarian — ingest module.

Watches `references/inbox/<client>/`. For each new .mp4 with a sidecar .json:
  1. Read sidecar metadata (niche, engagement, source_url, client)
  2. Upload to TwelveLabs (cached per-source-hash) — needed for Pegasus
  3. Fingerprint (transcript + visual + editorial + Pegasus)
  4. Compute transcript + Pegasus-description embeddings via sentence-transformers
  5. Insert row into data/rag.db (SQLite)
  6. Move MP4 from inbox/ to references/<client>/<id>.mp4
  7. Mark processed (no re-ingest on next run)

Run:  python -m clip_pipeline.rag.ingest <client>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import fingerprint as fp


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCES_INBOX = REPO_ROOT / "references" / "inbox"
REFERENCES_STORE = REPO_ROOT / "references"
DATA_DIR = REPO_ROOT / "data"
DB_PATH = DATA_DIR / "rag.db"
SCHEMA_PATH = DATA_DIR / "schema.sql"


# ---------- DB ----------

def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fresh = not DB_PATH.exists()
    conn = sqlite3.connect(str(DB_PATH))
    if fresh and SCHEMA_PATH.exists():
        conn.executescript(SCHEMA_PATH.read_text())
        conn.commit()
    return conn


def _file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------- TwelveLabs upload (cached) ----------

def _twelvelabs_upload(video_path: Path) -> str | None:
    """Returns a TwelveLabs video_id for the file, or None if upload fails.

    Per-machine cache file maps file_hash -> video_id so we never re-upload
    the same content. Cache is INDEX-AWARE: the cache file embeds the index_id
    it was populated against; if the current account's first index_id no
    longer matches (user deleted + recreated the index), the cache is
    invalidated wholesale on next ingest. This prevents Pegasus 404s from
    cached video_ids pointing at a deleted index.
    """
    if not os.environ.get("TWELVELABS_API_KEY"):
        print("[ingest] TWELVELABS_API_KEY not set; skipping Pegasus description",
              file=sys.stderr)
        return None

    # Resolve current index up front so we can validate cache against it.
    try:
        from twelvelabs import TwelveLabs
        client = TwelveLabs(api_key=os.environ["TWELVELABS_API_KEY"])
        indexes = list(client.indexes.list())
        if not indexes:
            print("[ingest] no TwelveLabs index found", file=sys.stderr)
            return None
        index_id = indexes[0].id
    except Exception as e:
        print(f"[ingest] TwelveLabs index lookup failed: {e}", file=sys.stderr)
        return None

    cache_path = DATA_DIR / "twelvelabs_references.json"
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    # Index-aware invalidation: if the cache was populated against a
    # different index_id, throw it out. Deleting + recreating the index
    # on the dashboard implicitly resets the cache here.
    cached_index = cache.get("_index_id")
    if cached_index != index_id:
        if cache:
            print(f"[ingest] TwelveLabs index changed "
                  f"(cached={cached_index!r}, now={index_id!r}); "
                  f"invalidating {len(cache) - 1} cached video_id(s)",
                  file=sys.stderr)
        cache = {"_index_id": index_id}

    h = _file_hash(video_path)
    if h in cache and h != "_index_id":
        return cache[h]

    try:
        with open(video_path, "rb") as f:
            task = client.tasks.create(index_id=index_id, video_file=f)
        task_id = getattr(task, "id", None) or getattr(task, "_id", None)
        if not task_id:
            return None
        print(f"[ingest] TwelveLabs task {task_id}; waiting...", file=sys.stderr)
        done = client.tasks.wait_for_done(task_id, sleep_interval=6.0)
        if getattr(done, "status", "") != "ready":
            return None
        video_id = getattr(done, "video_id", None)
        if video_id:
            cache[h] = video_id
            cache_path.write_text(json.dumps(cache, indent=2))
        return video_id
    except Exception as e:
        print(f"[ingest] TwelveLabs upload failed: {e}", file=sys.stderr)
        return None


# ---------- embedding ----------

_EMBED_MODEL = None


def _embed(texts: list[str]) -> list[np.ndarray]:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = _EMBED_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return [v.astype("float32") for v in vecs]


# ---------- per-clip ingest ----------

def ingest_one(
    mp4_path: Path,
    sidecar_path: Path,
    client: str,
    skip_twelvelabs: bool = False,
) -> bool:
    """Ingest a single reference clip. Returns True on success."""
    if not mp4_path.exists() or not sidecar_path.exists():
        print(f"[ingest] missing files for {mp4_path.name}", file=sys.stderr)
        return False
    try:
        meta = json.loads(sidecar_path.read_text())
    except Exception as e:
        print(f"[ingest] sidecar parse failed for {sidecar_path}: {e}", file=sys.stderr)
        return False

    clip_id = mp4_path.stem
    niche = meta.get("niche", "unknown")
    engagement = {k: meta.get(k) for k in ("views", "likes", "saves", "shares")}

    # Already in DB?
    conn = _connect()
    cur = conn.execute("SELECT id FROM reference_clips WHERE id=?", (clip_id,))
    if cur.fetchone():
        print(f"[ingest] {clip_id}: already in DB, skipping", file=sys.stderr)
        conn.close()
        return True

    # TwelveLabs upload (for Pegasus description). Skipped entirely when
    # `--no-pegasus` is passed — saves 30-60s per clip when the account
    # doesn't have a Pegasus-enabled index. The fingerprinter handles
    # video_id=None gracefully (blank PegasusDescription).
    if skip_twelvelabs:
        twelvelabs_video_id = None
    else:
        print(f"[ingest] {clip_id}: uploading to TwelveLabs...", file=sys.stderr)
        twelvelabs_video_id = _twelvelabs_upload(mp4_path)

    # Fingerprint
    print(f"[ingest] {clip_id}: fingerprinting...", file=sys.stderr)
    t0 = time.time()
    fingerprint = fp.fingerprint_clip(
        mp4_path,
        clip_id=clip_id,
        niche=niche,
        engagement=engagement,
        twelvelabs_video_id=twelvelabs_video_id,
        transcribe=True,
    )
    print(f"[ingest] {clip_id}: fingerprint done in {time.time()-t0:.1f}s",
          file=sys.stderr)

    # Embeddings
    description = fingerprint.pegasus.joined() or "no description"
    transcript_text = fingerprint.transcript or "no transcript"
    transcript_vec, description_vec = _embed([transcript_text, description])

    # Move MP4 to references/<client>/
    target_dir = REFERENCES_STORE / client
    target_dir.mkdir(parents=True, exist_ok=True)
    target_mp4 = target_dir / mp4_path.name
    if not target_mp4.exists():
        mp4_path.rename(target_mp4)
    # Move sidecar too
    target_sidecar = target_dir / sidecar_path.name
    if sidecar_path.exists() and not target_sidecar.exists():
        sidecar_path.rename(target_sidecar)

    # Insert
    conn.execute(
        """
        INSERT INTO reference_clips (
            id, client, niche, source_url, posted_at, duration_s,
            views, likes, saves, shares,
            transcript, pegasus_hook, pegasus_pacing, pegasus_captions, pegasus_payoff,
            visual_signature, editorial_signature,
            transcript_embedding, description_embedding
        )
        VALUES (?, ?, ?, ?, ?, ?,  ?, ?, ?, ?,  ?, ?, ?, ?, ?,  ?, ?,  ?, ?)
        """,
        (
            clip_id, client, niche, meta.get("source_url"), meta.get("posted_at"),
            fingerprint.duration_s,
            engagement.get("views"), engagement.get("likes"),
            engagement.get("saves"), engagement.get("shares"),
            fingerprint.transcript,
            fingerprint.pegasus.hook, fingerprint.pegasus.pacing,
            fingerprint.pegasus.captions, fingerprint.pegasus.payoff,
            json.dumps(asdict(fingerprint.visual)),
            json.dumps(asdict(fingerprint.editorial)),
            transcript_vec.tobytes(),
            description_vec.tobytes(),
        ),
    )
    conn.commit()
    conn.close()
    print(f"[ingest] {clip_id}: inserted into rag.db (niche={niche})", file=sys.stderr)
    return True


# ---------- inbox driver ----------

def ingest_inbox(client: str, skip_twelvelabs: bool = False) -> int:
    """Walk references/inbox/<client>/, ingest every .mp4 with a matching
    sidecar .json. Returns the count successfully ingested."""
    inbox = REFERENCES_INBOX / client
    if not inbox.is_dir():
        print(f"[ingest] no inbox at {inbox}", file=sys.stderr)
        return 0

    mp4s = sorted(inbox.glob("*.mp4"))
    if not mp4s:
        print(f"[ingest] {client}: inbox empty", file=sys.stderr)
        return 0

    n = 0
    for mp4 in mp4s:
        sidecar = mp4.with_suffix(".json")
        if not sidecar.exists():
            print(f"[ingest] {mp4.name}: no sidecar, skipping", file=sys.stderr)
            continue
        if ingest_one(mp4, sidecar, client, skip_twelvelabs=skip_twelvelabs):
            n += 1
    print(f"[ingest] {client}: ingested {n} new reference(s)", file=sys.stderr)

    # Update the ClientStyleProfile at ingest time (acceptance criterion #1).
    # If the client now has ≥ MIN_REFS_FOR_DERIVATION fingerprints, derive a
    # fresh profile.json so it's inspectable / editable BEFORE the first cut.py run.
    from .. import profile as profile_mod
    derived = profile_mod.derive_from_references(client)
    if derived is not None:
        profile_mod.save(derived)
    else:
        print(f"[ingest] {client}: profile not yet derivable "
              f"(need ≥{profile_mod.MIN_REFS_FOR_DERIVATION} refs)", file=sys.stderr)

    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("client", help="Client name (folder under references/inbox/)")
    ap.add_argument(
        "--no-pegasus",
        action="store_true",
        help="Skip TwelveLabs upload entirely. Use when your TL account "
             "doesn't have a Pegasus-enabled index — saves 30-60s per clip. "
             "Fingerprinting still runs locally (transcript + visual + "
             "editorial signatures + transcript embeddings). Pegasus "
             "description fields are left blank; moment-intelligence "
             "automatically picks the Marengo-mode scoring weights.",
    )
    args = ap.parse_args()

    # Load .env so TWELVELABS_API_KEY is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    n = ingest_inbox(args.client, skip_twelvelabs=args.no_pegasus)
    return 0 if n >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
