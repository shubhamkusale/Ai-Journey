def get_note(note_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, text FROM notes WHERE id = ?", (note_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {"id": row[0], "text": row[1]}

def list_notes(limit):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, text FROM notes ORDER BY id LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    notes = []
    for row in rows:
        notes.append({"id": row[0], "text": row[1]})
    return notes