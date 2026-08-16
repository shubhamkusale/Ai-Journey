import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import math

reviews = [
    "this movie was absolutely fantastic and wonderful",
    "great film loved every moment of it",
    "brilliant acting and amazing story",
    "one of the best movies i have ever seen",
    "incredible experience highly recommend this film",
    "wonderful performances and beautiful cinematography",
    "this was a masterpiece loved it completely",
    "outstanding movie will watch again",
    "terrible movie waste of time",
    "worst film i have ever seen horrible",
    "boring and pointless avoid this movie",
    "awful acting and terrible story",
    "complete disaster hated every moment",
    "disappointing and dull would not recommend",
    "poorly made film very bad experience",
    "dreadful movie absolutely hated it",
]
labels = [1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0]

def build_vocab(reviews):
    all_words = []
    for review in reviews:
        words = re.sub(r'[^a-z ]', '', review.lower()).split()
        all_words.extend(words)
    word_counts = Counter(all_words)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word in word_counts:
        vocab[word] = len(vocab)
    return vocab

def text_to_numbers(review, vocab, max_len=10):
    words   = re.sub(r'[^a-z ]', '', review.lower()).split()
    numbers = [vocab.get(word, 1) for word in words]
    if len(numbers) < max_len:
        numbers = [0] * (max_len - len(numbers)) + numbers
    else:
        numbers = numbers[:max_len]
    return numbers

vocab = build_vocab(reviews)

class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, vocab, max_len=10):
        self.data = []
        for review, label in zip(reviews, labels):
            numbers = text_to_numbers(review, vocab, max_len)
            self.data.append((
                torch.tensor(numbers, dtype=torch.long),
                torch.tensor(label,   dtype=torch.long)
            ))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                pe[pos, i]   = math.sin(pos / (10000 ** (i / embed_size)))
                if i + 1 < embed_size:
                    pe[pos, i+1] = math.cos(pos / (10000 ** (i / embed_size)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads,
                 num_layers, num_classes, max_len=10):
        super().__init__()
        self.embedding    = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_size, max_len)
        encoder_layer     = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout      = nn.Dropout(0.3)
        self.fc           = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

dataset    = ReviewDataset(reviews, labels, vocab)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = SentimentTransformer(
    vocab_size  = len(vocab),
    embed_size  = 32,
    num_heads   = 4,
    num_layers  = 2,
    num_classes = 2
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

print("Training Transformer...")
for epoch in range(20):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for texts, labels_batch in dataloader:
        optimizer.zero_grad()
        outputs  = model(texts)
        loss     = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total   += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()

    accuracy = 100 * correct / total
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}: loss={total_loss:.4f}, acc={accuracy:.1f}%")

def predict(review, model, vocab):
    model.eval()
    numbers = text_to_numbers(review, vocab)
    tensor  = torch.tensor(numbers).unsqueeze(0)
    with torch.no_grad():
        output  = model(tensor)
        _, pred = torch.max(output, 1)
    return "POSITIVE 😊" if pred.item() == 1 else "NEGATIVE 😞"

print("\nTesting:")
print(predict("this movie was amazing loved it",   model, vocab))
print(predict("terrible film hated every second",  model, vocab))
print(predict("fantastic story and great acting",  model, vocab))
print(predict("boring and awful waste of time",    model, vocab))