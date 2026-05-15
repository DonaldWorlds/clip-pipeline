import sqlite3


def save_metrics(
    clip_id,
    retention,
    watch_time,
    ctr,
    motion_score
):

    conn = sqlite3.connect("clips.db")

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO clip_metrics
        (
            clip_id,
            retention,
            watch_time,
            ctr,
            motion_score
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            clip_id,
            retention,
            watch_time,
            ctr,
            motion_score
        )
    )

    conn.commit()

    conn.close()


'''
STEP 3 — USE METRICS FOR FEEDBACK

THIS is the important part.

Eventually:

high retention clips
↓
extract patterns
↓
store in RAG
↓
future clips prioritize similar edits

THAT is how systems evolve.

EXAMPLE FUTURE FLOW
thumbnail style performs well
↓
system increases confidence score
↓
future generations bias toward that style

Now your system becomes:

adaptive

instead of static.

THIS Is The Real Long-Term Vision

Eventually your pipeline becomes:

GENERATE
↓
VALIDATE
↓
UPLOAD
↓
TRACK PERFORMANCE
↓
LEARN PATTERNS
↓
IMPROVE FUTURE OUTPUTS
'''