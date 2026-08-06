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
        words = re.sub(r'[^a-z]', '',review.lower()).split()
        all_words.extend(words)

    word_counts = Counter(all_words)

    vocab = {'<PAD>':0, 'UNK':1}
    for word,count in word_counts.items():
        vocab[word] = len(vocab)

    return vocab

vocab = build_vocab(reviews)
print(f"Vocabulary size: {len(vocab)}")

def text_to_numbers(review, vocab, max_len = 10):
    words  = re.sub(r'[^a-z ]', '', review.lower()).split()
    numbers =[vocab.get(word, 1)for word in words]

    if len(numbers) < max_len:
        numers = [0]* (max_len - len(numbers)) + numbers
    else:
        numbers =numbers[:max_len]

    return numbers

class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, vocab, max_len= 10):
        self.data = []
        for review, label in zip(reviews, labels):
            numbers = text_to_numbers(review, labels)
            self.data.append((
                torch.tensor(numbers, dtype=torch.long),
                torch.tensor(label,   dtype=torch.long)
            ))
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class SentimentLSTM(nn.module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        x = hidden[-1]
        x = self.dropout(x)
        x = self.fc(x)
        return x