import sqlite3

conn = sqlite3.connect("practice.db")
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS notes")
cur.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL)")

cur.execute("INSERT INTO notes (text) VALUES (?)", ("buy milk",))
cur.execute("INSERT INTO notes (text) VALUES (?)", ("call dad",))
cur.execute("INSERT INTO notes (text) VALUES (?)", ("pay bill",))
conn.commit()

cur.execute("UPDATE notes SET text = ? WHERE id = ?", ("buy oat milk", 1))
cur.execute("DELETE FROM notes WHERE id = ?", (2,))
conn.commit()

cur.execute("SELECT id, text FROM notes")
print(cur.fetchall())
conn.close()