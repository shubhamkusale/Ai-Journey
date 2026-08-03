import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

# ─── STEP 1: RAW DATA ───
# Simple movie reviews dataset
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
# 1 = positive, 0 = negative

# ─── STEP 2: BUILD VOCABULARY ───
def build_vocab(reviews):
    all_words = []
    for review in reviews:
        words = re.sub(r'[^a-z ]', '', review.lower()).split()
        all_words.extend(words)
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Create word → number mapping
    # 0 = padding token (empty space)
    # 1 = unknown word
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.items():
        vocab[word] = len(vocab)
    
    return vocab

vocab = build_vocab(reviews)
print(f"Vocabulary size: {len(vocab)}")

# ─── STEP 3: CONVERT TEXT TO NUMBERS ───
def text_to_numbers(review, vocab, max_len=10):
    words = re.sub(r'[^a-z ]', '', review.lower()).split()
    
    # Convert words to numbers
    numbers = [vocab.get(word, 1) for word in words]
    # vocab.get(word, 1) = if word not in vocab, use 1 (<UNK>)
    
    # Pad or truncate to max_len
    if len(numbers) < max_len:
        numbers = [0] * (max_len - len(numbers)) + numbers
    else:
        numbers = numbers[:max_len]
    
    return numbers

# ─── STEP 4: DATASET CLASS ───
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

dataset    = ReviewDataset(reviews, labels, vocab)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ─── STEP 5: LSTM MODEL ───
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        
        # Word numbers → vectors
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # LSTM — the memory network
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # Dropout before final layer
        self.dropout = nn.Dropout(0.3)
        
        # Final classifier
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_len)
        
        x = self.embedding(x)
        # x shape: (batch, seq_len, embed_size)
        
        output, (hidden, cell) = self.lstm(x)
        # hidden shape: (1, batch, hidden_size)
        
        # Take final hidden state only
        x = hidden[-1]
        # x shape: (batch, hidden_size)
        
        x = self.dropout(x)
        x = self.fc(x)
        # x shape: (batch, num_classes)
        
        return x

# ─── STEP 6: SETUP ───
model     = SentimentLSTM(
    vocab_size   = len(vocab),
    embed_size   = 32,
    hidden_size  = 64,
    num_classes  = 2
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# ─── STEP 7: TRAINING LOOP ───
print("\nTraining...")
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

# ─── STEP 8: TEST ON NEW REVIEW ───
def predict(review, model, vocab):
    model.eval()
    numbers = text_to_numbers(review, vocab)
    tensor  = torch.tensor(numbers).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)
    
    return "POSITIVE 😊" if predicted.item() == 1 else "NEGATIVE 😞"

print("\nTesting on new reviews:")
print(predict("this movie was amazing loved it",  model, vocab))
print(predict("terrible film hated every second", model, vocab))
print(predict("fantastic story great acting",     model, vocab))