CREATE TABLE feedback_memory (
    id INTEGER PRIMARY KEY,
    clip_id INTEGER,
    outcome TEXT,
    pattern_type TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);