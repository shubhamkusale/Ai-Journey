import torch
import torch.nn as nn

# PIECE 1 — the data
X = torch.tensor([[1.0], [2.0], [3.0], [6.0], [7.0], [8.0]])
y = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])

# PIECE 2 — the model (the machine)
model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)


loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(X)              # forward — guess
    loss = loss_fn(y_pred, y)      # how wrong
    optimizer.zero_grad()          # reset
    loss.backward()                # backward — find the blame
    optimizer.step()               # nudge the weights
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# Ask the trained network: 5 hours — pass or fail?
test = torch.tensor([[5.0]])     # put 5.0 in the blank
answer = model(test)              # pass your test tensor through the trained model
print(answer)                     # show the result