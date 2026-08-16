Today i learned that we cant learn something by forcing aur brain has a limit after that we have to stop 
20 min learning 5 mins rest for deepest focus learning 
90 mins deep learning max taking small 5min gaps 
active recall 
before sleep rewise what u do all the day and try to remember smallest things ku did
Today i guess no code but htis readme might help get u learn fast like 1% peoples 

i got this now i wested 3 years to learn this now i know 

# Week 13 — Transformers

## What I learned

### The Problem with LSTMs
- Reads words sequentially (slow — can't parallelize)
- Still struggles with very long sequences
- Must wait for word 1 before processing word 2

### The Transformer Solution (2017 — "Attention Is All You Need")
- Reads ALL words simultaneously (parallel = fast)
- Every word looks at every other word (self-attention)
- Powers: GPT, Claude, Gemini — all modern AI

### Self-Attention
- For each word: "which other words should I pay attention to?"
- Example: "it" in "The animal didn't cross because it was tired"
  → "it" pays most attention to "animal" (not "street")
- Q (Query)  = "what am I looking for?"
- K (Key)    = "what do I contain?"
- V (Value)  = "what do I actually give?"

### Positional Encoding
- Transformer reads all words at once → loses word ORDER
- Solution: add position signal to each word BEFORE attention
- Uses sine/cosine waves → every position gets unique pattern
- Word vector = meaning (embedding) + position (encoding)

### Multi-Head Attention
- Run 4 attention heads simultaneously
- Head 1 → subject-verb relationships
- Head 2 → sentiment words
- Head 3 → negation ("not good")
- Head 4 → contrast ("but", "however")
- Combined = richer understanding than single attention

### Stacked Layers
- Layer 1 → finds basic connections
- Layer 2 → finds connections between connections
- GPT-4 has 96 layers

## What I Built
Same sentiment classifier as Week 12 — but using Transformer instead of LSTM.
Same data. Different architecture. Better results.

## Results Comparison

| Model | Accuracy | Architecture |
|-------|----------|--------------|
| LSTM (Week 12) | 93.8% | Sequential, hidden state |
| Transformer (Week 13) | 100.0% | Parallel, self-attention |

**Transformer wins. This is why all modern AI uses Transformers.**

## How It Works