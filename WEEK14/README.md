# Week 14 — LangChain Concepts + Real AI APIs

## Overview
This week moved from building models from scratch (Weeks 1–13) into using a **hosted, pre-trained LLM via API** — the foundation for building agentic systems like Jarvis. Covered the core building blocks that every LLM-powered application uses under the hood: making API calls, giving the model memory, chaining multi-step calls together, and letting the model use external tools.

---

## What's Covered

### 1. Basic API Calls (Groq)
- Used [Groq](https://console.groq.com) as the LLM provider — hosts Llama/GPT-OSS models, free tier, fast inference.
- Learned why renting a hosted model via API is the correct approach over trying to train/host a 70B+ parameter model locally (cost, hardware, and scale are out of reach for an individual).
- Hit and resolved a real `model_not_found` (404) error — `llama-3.3-70b-versatile` was deprecated by Groq; migrated to `openai/gpt-oss-120b`.

### 2. Secure API Key Handling
- Learned why hardcoding API keys in source files is dangerous (permanent exposure in Git history once committed, even if later removed).
- Implemented the standard pattern: `.env` file for the real key + `python-dotenv` to load it + `.gitignore` to keep `.env` out of version control.
- Verified via `git log -p` / `git check-ignore` that no real key was ever committed to the repo history.

### 3. Memory (Multi-Turn Conversation)
- Core concept: **an LLM API call has zero memory between calls.** Every request is treated as brand new.
- The "memory" effect in every chatbot is created by **resending the entire conversation history** with every new request.
- Implemented manually in plain Python — a growing `conversation` list, appended to after every user message and every model reply, then resent in full on each call.
- Understood the tradeoff: longer conversations = more tokens resent = higher cost and slower responses each turn.

### 4. Chains
- Core concept: a chain is simply **the output of one API call becoming part of the input text for the next API call.**
- No special library or object required — just standard Python (variables + f-strings) connecting two `chat()` calls in sequence.
- Built a working summarize → translate chain as the example.

### 5. Tools
- Core concept: LLMs generate text by predicting likely-looking patterns — they don't calculate, fetch live data, or execute logic. Tools connect the model to real functions that *can*.
- Learned the full round-trip: model receives a tool "menu" → decides whether a tool is needed → if yes, requests a specific tool + arguments (not an answer yet) → your code runs the real function → the real result is sent back to the model in a second API call → model produces the final, human-readable answer.
- Built a working `calculate` tool as the example (real arithmetic via Python, not model guessing).

---

## Key Files
- `Lern_api.py` — Groq API setup, `.env`-based key loading

## Still To Cover (Week 14 continued)
- RAG (Retrieval-Augmented Generation)
- Prompt Engineering
- Vector databases / embeddings (ChromaDB, tied into Jarvis's planned memory layer)

## Why This Matters
These four blocks — API calls, memory, chains, tools — are the actual building blocks of any real agentic system, including Jarvis. Memory and Tools from this week are planned to be carried directly into the Jarvis repo as its first working "text-only brain" (V1), rather than staying as standalone practice scripts.