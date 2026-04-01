import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


data = {
    "sequence": ["ACDEFGHIK", "LMNPQRSTV", "ACDLMNPQR"],
    "structure": ["012012012", "120120120", "012120120"]
}

df = pd.DataFrame(data)


amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {aa: i+1 for i, aa in enumerate(amino_acids)}

def encode_sequence(seq):
    return [aa_to_idx.get(aa, 0) for aa in seq]

def pad_sequences(sequences, max_len):
    padded = []
    for seq in sequences:
        seq = seq[:max_len]
        seq += [0] * (max_len - len(seq))
        padded.append(seq)
    return torch.tensor(padded)

sequences = [encode_sequence(seq) for seq in df['sequence']]
labels = [[int(c) for c in s] for s in df['structure']]

max_len = 20
X = pad_sequences(sequences, max_len)
y = pad_sequences(labels, max_len)


class ProteinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(amino_acids)+1, 128)

        self.lstm = nn.LSTM(
            128, 256,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Linear(512, 1)
        self.fc = nn.Linear(512, 3)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)

        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = attn_weights * lstm_out

        output = self.fc(context)
        return output

model = ProteinModel()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training started...\n")

for epoch in range(15):
    optimizer.zero_grad()

    outputs = model(X)
    loss = criterion(outputs.view(-1, 3), y.view(-1))

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

def q3_accuracy(preds, labels):
    correct = 0
    total = 0

    for p, l in zip(preds, labels):
        for pi, li in zip(p, l):
            if li != 0:
                total += 1
                if pi == li:
                    correct += 1

    return correct / total

with torch.no_grad():
    outputs = model(X)
    preds = torch.argmax(outputs, dim=-1)

acc = q3_accuracy(preds, y)
print(f"\nQ3 Accuracy: {acc:.4f}")


PREDICTION FUNCTION

def predict(sequence):
    seq = encode_sequence(sequence)
    seq = pad_sequences([seq], max_len)

    with torch.no_grad():
        output = model(seq)
        pred = torch.argmax(output, dim=-1)

    return pred[0].tolist()

test_seq = "ACDEFGHIK"
prediction = predict(test_seq)

print(f"\nSequence: {test_seq}")
print(f"Predicted Structure: {prediction}")
