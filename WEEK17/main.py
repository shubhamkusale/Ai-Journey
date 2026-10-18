from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
notes = []

class Note(BaseModel):
    test: str


@app.get("/notes")
def get_notes():
    return notes

@app.post("/notes")
def add_note(note: Note):
    notes.append({"text ": note.text})
    return {"added":note.text}

@app.get("/notes/search")
def find_notes(word:str):
    result = []
    for note in notes:
        if word in note["text"]:
            result.append(note)
    return result
