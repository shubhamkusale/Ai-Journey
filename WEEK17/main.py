import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

class NoteIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)

class Note(BaseModel):
    id: int
    text: str

def load_notes():
    if os.path.exists("notes.json"):
        with open("notes.json", "r") as f:
            return json.load(f)
    return []

def save_notes():
    with open("notes.json", "w") as f:
        json.dump(notes, f)

notes = load_notes()

def next_id():
    return max([n["id"] for n in notes], default=0) + 1

@app.get("/notes", response_model=list[Note])
def list_notes(limit: int = 10):
    return notes[:limit]

@app.get("/notes/search", response_model=list[Note])
def search_notes(word: str):
    result = []
    for n in notes:
        if word in n["text"]:
            result.append(n)
    return result

@app.get("/notes/{note_id}", response_model=Note)
def get_note(note_id: int):
    for n in notes:
        if n["id"] == note_id:
            return n
    raise HTTPException(status_code=404, detail="Note not found")

@app.post("/notes", response_model=Note, status_code=201)
def create_note(body: NoteIn):
    note = {"id": next_id(), "text": body.text}
    notes.append(note)
    save_notes()
    return note

@app.put("/notes/{note_id}", response_model=Note)
def replace_note(note_id: int, body: NoteIn):
    for n in notes:
        if n["id"] == note_id:
            n["text"] = body.text
            save_notes()
            return n
    raise HTTPException(status_code=404, detail="Note not found")

@app.delete("/notes/{note_id}", status_code=204)
def delete_note(note_id: int):
    for n in notes:
        if n["id"] == note_id:
            notes.remove(n)
            save_notes()
            return
    raise HTTPException(status_code=404, detail="Note not found")