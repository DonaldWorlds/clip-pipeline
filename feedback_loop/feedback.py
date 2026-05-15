import sqlite3


def store_feedback(
    clip_id,
    outcome,
    pattern_type,
    notes
):

    conn = sqlite3.connect("clips.db")

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO feedback_memory
        (
            clip_id,
            outcome,
            pattern_type,
            notes
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            clip_id,
            outcome,
            pattern_type,
            notes
        )
    )

    conn.commit()

    conn.close()

