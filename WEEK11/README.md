# Week 11 — Training Deep Networks Well

## What I learned
- Overfitting vs Underfitting
- Dropout (prevent memorization)
- Batch Normalization (stable training)
- Learning Rate + ReduceLROnPlateau scheduler
- AdamW optimizer
- Early Stopping + Checkpointing
- Data Augmentation
- Train/Val Split

## Results

| What I changed | Val Accuracy |
|----------------|-------------|
| Baseline (Week 10 MNIST) | ~90% easy dataset |
| Fashion-MNIST + all techniques | 88.4% harder dataset |

## Model
- 2 Conv blocks (16→32 filters)
- Batch Norm + Dropout after each block
- AdamW optimizer (lr=0.001)
- Early stopping patience=5
- Data augmentation (rotation + shift)

## Files
- `fashion_mnist_improved.py` — full training code
- `best_fashion_model.pth` — saved best model weights