import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 1. Load dataset
print("Loading dataset...")
data = np.load("CB513.npy", allow_pickle=True).item()  # CB513.npy should be a dict with 'X' and 'Y'
X = data['X']
Y = data['Y']
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# 2. Dataset class
class ProteinDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
        if self.Y.ndim == 3:  # If one-hot, convert to class index
            self.Y = torch.argmax(self.Y, dim=2)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = ProteinDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 3. Model definition
class ProteinPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ProteinPredictor, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=2, 
                              batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x = self.relu(self.cnn(x))
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        x, _ = self.bilstm(x)
        x = self.fc(x)
        return x

input_dim = X.shape[2]
hidden_dim = 128
num_classes = Y.shape[2] if Y.ndim == 3 else len(np.unique(Y))
model = ProteinPredictor(input_dim, hidden_dim, num_classes)

# 4. Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_Y in dataloader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.view(-1, num_classes), batch_Y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

# 5. Prediction
model.eval()
with torch.no_grad():
    sample_X = torch.tensor(X[:5], dtype=torch.float32).to(device)
    preds = model(sample_X)
    preds_classes = torch.argmax(preds, dim=2)
    print("Sample predictions:\n", preds_classes.cpu().numpy())
