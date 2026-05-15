import sqlite3


def get_pending_jobs():
    conn = sqlite3.connect("clips.db")

    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM clips WHERE status='pending'"
    )

    rows = cursor.fetchall()

    conn.close()

    return rows