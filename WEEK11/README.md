we done this week 11 for deep learning of aur neural networks to understand more about pytorch, neaural networks and all 
EPOCH STARTS
     ↓
┌─────────────────────────────────┐
│  TRAINING PHASE                 │
│  model.train()                  │
│                                 │
│  for each batch (750 batches):  │
│  1. zero_grad()                 │
│  2. forward pass                │
│  3. calculate loss              │
│  4. backward()                  │
│  5. step() — update weights     │
└─────────────────────────────────┘
     ↓
┌─────────────────────────────────┐
│  VALIDATION PHASE               │
│  model.eval()                   │
│  torch.no_grad()                │
│                                 │
│  for each batch (no learning):  │
│  - forward pass only            │
│  - measure loss + accuracy      │
│  - NO backward, NO step         │
└─────────────────────────────────┘
     ↓
┌─────────────────────────────────┐
│  AFTER EPOCH                    │
│  - scheduler checks val_loss    │
│  - print results                │
│  - checkpoint if improved       │
│  - early stop if patience hit   │
└─────────────────────────────────┘
     ↓
NEXT EPOCH (or stop if early stopping)