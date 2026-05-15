"""RAG Librarian — retrieval module.

Given a client + niche + (optional) query text, return the top-K most relevant
reference clips from data/rag.db. Used by editorial/decisions.py to ground
per-clip decisions in real high-performing references.

Cosine similarity is computed over BOTH the transcript embedding AND the
Pegasus-description embedding. The two scores are blended (60% description,
40% transcript) so style/vibe matters more than topic match.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "rag.db"


@dataclass
class RetrievedReference:
    id: str
    client: str
    niche: str
    duration_s: float
    views: int | None
    likes: int | None
    transcript: str
    pegasus: dict                 # {hook, pacing, captions, payoff}
    visual: dict                  # parsed JSON visual_signature
    editorial: dict               # parsed JSON editorial_signature
    score: float                  # combined similarity
    transcript_score: float
    description_score: float


_EMBED_MODEL = None


def _embed(texts: list[str]) -> np.ndarray:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")


def retrieve(
    client: str,
    niche: str | None = None,
    k: int = 5,
    query_transcript: str | None = None,
    query_description: str | None = None,
) -> list[RetrievedReference]:
    """Return top-K reference clips for the given client (+ optional niche)
    ranked by combined cosine similarity with the query embeddings.

    If no query is provided, returns the top-K by total engagement (likes+views).
    Returns [] if the DB doesn't exist or has no matches."""
    if not DB_PATH.exists():
        return []

    conn = sqlite3.connect(str(DB_PATH))
    if niche:
        cur = conn.execute(
            "SELECT * FROM reference_clips WHERE client=? AND niche=?",
            (client, niche),
        )
    else:
        cur = conn.execute(
            "SELECT * FROM reference_clips WHERE client=?", (client,),
        )
    col_names = [d[0] for d in cur.description]
    rows = [dict(zip(col_names, r)) for r in cur.fetchall()]
    conn.close()

    if not rows:
        return []

    # If no query provided, sort by engagement
    if not query_transcript and not query_description:
        rows.sort(
            key=lambda r: ((r.get("likes") or 0) + (r.get("views") or 0)),
            reverse=True,
        )
        return [_to_ref(r, score=1.0, t_score=0.0, d_score=0.0) for r in rows[:k]]

    # Compute query embeddings
    query_texts = [
        (query_transcript or "no transcript"),
        (query_description or "no description"),
    ]
    query_vecs = _embed(query_texts)
    q_t, q_d = query_vecs[0], query_vecs[1]

    # Score each row
    scored: list[tuple[float, float, float, dict]] = []
    for row in rows:
        t_blob = row.get("transcript_embedding") or b""
        d_blob = row.get("description_embedding") or b""
        if not t_blob and not d_blob:
            continue
        t_vec = np.frombuffer(t_blob, dtype="float32") if t_blob else None
        d_vec = np.frombuffer(d_blob, dtype="float32") if d_blob else None
        # Cosine similarity = dot product (vectors are normalized at ingest)
        t_score = float(np.dot(q_t, t_vec)) if t_vec is not None and t_vec.size == q_t.size else 0.0
        d_score = float(np.dot(q_d, d_vec)) if d_vec is not None and d_vec.size == q_d.size else 0.0
        combined = 0.4 * t_score + 0.6 * d_score
        scored.append((combined, t_score, d_score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[RetrievedReference] = []
    for combined, t_s, d_s, row in scored[:k]:
        out.append(_to_ref(row, score=combined, t_score=t_s, d_score=d_s))
    return out


def _to_ref(row: dict, score: float, t_score: float, d_score: float) -> RetrievedReference:
    def _parse(s):
        try:
            return json.loads(s) if s else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return RetrievedReference(
        id=row["id"],
        client=row["client"],
        niche=row["niche"],
        duration_s=row.get("duration_s") or 0.0,
        views=row.get("views"),
        likes=row.get("likes"),
        transcript=row.get("transcript") or "",
        pegasus={
            "hook": row.get("pegasus_hook") or "",
            "pacing": row.get("pegasus_pacing") or "",
            "captions": row.get("pegasus_captions") or "",
            "payoff": row.get("pegasus_payoff") or "",
        },
        visual=_parse(row.get("visual_signature")),
        editorial=_parse(row.get("editorial_signature")),
        score=round(score, 4),
        transcript_score=round(t_score, 4),
        description_score=round(d_score, 4),
    )
