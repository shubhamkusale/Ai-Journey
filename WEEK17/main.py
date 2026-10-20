from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import json
import os

app = FastAPI()

class Note(BaseModel):
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

@app.get("/notes")
def get_notes():
    return notes

@app.post("/notes")
def add_note(note: Note):
    notes.append({"text": note.text})
    save_notes()
    return {"added": note.text}

@app.get("/notes/search")
def find_notes(word: str):
    result = []
    for note in notes:
        if word in note["text"]:
            result.append(note)
    return result

@app.get("/files/hello.txt")
def get_hello():
    return FileResponse("hello.txt")