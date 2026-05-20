"""
Weighted Pooling para NLP
Exemplo: Classificação de risco em textos clínicos simplificados

Autor: Prof. Mozart Hasse (adaptado)
"""

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import LABELS, SENTENCES


# =========================
# 2. Vocabulário simples
# =========================
def build_vocab(sentences):
    vocab = {"<pad>": 0}
    idx = 1

    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

vocab = build_vocab(SENTENCES)

def encode(sentence, vocab):
    return [vocab[word] for word in sentence.split()]

encoded_sentences = [encode(s, vocab) for s in SENTENCES]

# padding
max_len = max(len(s) for s in encoded_sentences)

def pad(seq, max_len):
    return seq + [0] * (max_len - len(seq))

inputs = torch.tensor([pad(s, max_len) for s in encoded_sentences])

# =========================
# 3. Weighted Pooling Layer
# =========================
class WeightedPooling(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.scorer = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings, mask=None):
        """
        embeddings: (batch, seq_len, emb_dim)
        """

        scores = self.scorer(embeddings).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=1)  # atenção simplificada

        pooled = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)

        return pooled, weights


# =========================
# 4. Modelo completo
# =========================
class RiskClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pooling = WeightedPooling(emb_dim)
        self.classifier = nn.Linear(emb_dim, 2)

    def forward(self, x):
        mask = (x != 0).float()

        emb = self.embedding(x)
        pooled, weights = self.pooling(emb, mask)

        logits = self.classifier(pooled)
        return logits, weights


# =========================
# 5. Treinamento
# =========================
model = RiskClassifier(len(vocab), emb_dim=64)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()

    logits, weights = model(inputs)

    loss = criterion(logits, LABELS)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# =========================
# 6. Inspeção de atenção
# =========================
model.eval()

logits, weights = model(inputs)
preds = torch.argmax(logits, dim=1)

for i, sentence in enumerate(SENTENCES):
    print("\n======================")
    print("Frase:", sentence)
    print("Predição:", preds[i].item())

    tokens = sentence.split()
    token_weights = weights[i][:len(tokens)]

    for token, w in zip(tokens, token_weights):
        print(f"{token:10s} -> {w.item():.4f}")