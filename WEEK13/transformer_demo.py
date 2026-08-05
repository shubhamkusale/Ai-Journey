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
