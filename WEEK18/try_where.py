import sqlite3

def get_connection():
    return sqlite3.connect("notes.db")

def create_table():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL)")
    conn.commit()
    conn.close()

def add_note(text):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO notes (text) VALUES (?)", (text,))
    conn.commit()
    note_id = cur.lastrowid
    conn.close()
    return note_id