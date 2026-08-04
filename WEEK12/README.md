# Week 12 — RNNs and LSTMs

## What I learned

### RNN (Recurrent Neural Network)
- Reads sequences ONE word at a time
- Carries a hidden state (memory) forward
- Problem: forgets early words in long sentences
- Called vanishing gradient problem

### LSTM (Long Short-Term Memory)
- Fixes RNN's forgetting problem
- Has TWO memory lanes:
  - Hidden state = short term memory
  - Cell state = long term memory (protected)
- Has THREE gates:
  - Forget gate: erases old useless info
  - Input gate: writes new important info
  - Output gate: decides what to output now
- Example: word "but" triggers forget gate
  → erases negative signal from "slow"
  → ready for positive "brilliantly"

## What I built
Sentiment classifier — reads movie reviews
predicts POSITIVE or NEGATIVE

## How it works
1. Build vocabulary (every word gets a number)
2. Convert reviews to number sequences
3. Pad all sequences to same length
4. Embedding: numbers → meaning vectors
5. LSTM reads word by word with memory
6. Final hidden state = summary of whole review
7. FC layer → positive or negative

## Files
- sentiment_lstm.py — full training code