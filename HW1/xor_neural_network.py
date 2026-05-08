import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define a simple Neural Network for XOR problem
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 2. Prepare Data
# XOR inputs: [0,0], [0,1], [1,0], [1,1]
# XOR outputs: 0, 1, 1, 0
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# 3. Initialize Model, Loss, and Optimizer
model = XORNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Training Loop
for epoch in range(2000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch+1}/2000], Loss: {loss.item():.4f}')

# 5. Testing
with torch.no_grad():
    predicted = model(X)
    print("\nPredictions:")
    print(predicted.round())
