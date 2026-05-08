import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Define the Neural Network Architecture
# This network has an input layer of 2 features, a hidden layer of 5 neurons, and 1 output neuron.
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # First layer: input (2) -> hidden (5)
        self.fc1 = nn.Linear(2, 5)
        # Activation function
        self.relu = nn.ReLU()
        # Second layer: hidden (5) -> output (1)
        self.fc2 = nn.Linear(5, 1)
        # Sigmoid to get a value between 0 and 1 (useful for binary classification)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 2. Prepare Dummy Data
# 4 samples with 2 features each
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
# Targets (labels)
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

# 3. Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Training Loop
print("Training the model...")
loss_history = []
for epoch in range(1000):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad() # Clear previous gradients
    loss.backward()       # Compute gradients
    optimizer.step()      # Update weights

    loss_history.append(loss.item())

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# Plotting the loss
plt.plot(loss_history)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 5. Test the model
print("\nPredictions after training:")
with torch.no_grad():
    predictions = model(X)
    print(predictions)

