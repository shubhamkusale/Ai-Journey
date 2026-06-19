# Pass/Fail Neural Network (PyTorch)

A neural network that predicts whether a student passes or fails based on hours studied.

## What it does
It takes an input (hours studied) and passes it through a network of neurons.
Each neuron multiplies the input by its own weights, adds them up, and passes
a number forward — layer by layer — until the final neuron outputs a value
between 0 and 1 (fail or pass). The network learns by guessing, measuring how
wrong it is, and nudging its weights thousands of times until the guesses are right. 
## What I learned
- Built a network with nn.Sequential (Linear → ReLU → Linear → Sigmoid)
- Trained it with a loss function (BCELoss) and an optimizer (Adam)
- Watched the loss drop from ~1.0 to ~0.002 over 1000 epochs
- Queried the trained model: 5 hours studied → 0.88 (likely pass)

## How to run
```
python pass_fail_net.py
```