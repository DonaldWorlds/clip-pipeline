-- RAG schema for the Librarian. One row per indexed reference clip.
-- Created lazily by clip_pipeline/rag/ingest.py on first run.

CREATE TABLE IF NOT EXISTS reference_clips (
    id                    TEXT PRIMARY KEY,
    client                TEXT NOT NULL,
    niche                 TEXT NOT NULL,
    source_url            TEXT,
    posted_at             TEXT,
    duration_s            REAL,

    -- engagement (from n8n sidecar)
    views                 INTEGER,
    likes                 INTEGER,
    saves                 INTEGER,
    shares                INTEGER,

    -- transcript + Pegasus description (natural language)
    transcript            TEXT,
    pegasus_hook          TEXT,
    pegasus_pacing        TEXT,
    pegasus_captions      TEXT,
    pegasus_payoff        TEXT,

    -- numeric fingerprint as JSON (cv2-measured features)
    visual_signature      TEXT,   -- JSON
    editorial_signature   TEXT,   -- JSON

    -- embeddings as BLOBs (numpy.tobytes())
    transcript_embedding  BLOB,
    description_embedding BLOB,

    ingested_at           TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_refs_client_niche ON reference_clips(client, niche);
CREATE INDEX IF NOT EXISTS idx_refs_client ON reference_clips(client);
