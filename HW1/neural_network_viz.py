import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 1. NEURAL NETWORK DEFINITION
# ============================================================================
class VisualizationNN(nn.Module):
    """
    A neural network: 2 inputs → 8 hidden neurons → 8 hidden neurons → 1 output
    Two hidden layers allow learning more complex non-linear patterns.
    """
    def __init__(self):
        super(VisualizationNN, self).__init__()
        # First hidden layer: 2 inputs → 8 hidden neurons
        self.fc1 = nn.Linear(2, 8)
        # Activation function (introduces non-linearity)
        self.relu = nn.ReLU()
        # Second hidden layer: 8 hidden → 8 hidden
        self.fc2 = nn.Linear(8, 8)
        # Output layer: 8 hidden → 1 output
        self.fc3 = nn.Linear(8, 1)
        # Sigmoid for binary classification (output between 0 and 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass through the network"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# ============================================================================
# 2. DATA GENERATION
# ============================================================================
print("Generating synthetic dataset (Two Moons - non-linear separable)...")
X_raw, y_raw = make_moons(n_samples=300, noise=0.1, random_state=42)
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Convert to PyTorch tensors
X = torch.tensor(X_scaled, dtype=torch.float32)
y = torch.tensor(y_raw, dtype=torch.float32).reshape(-1, 1)

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Class distribution: {np.bincount(y_raw)}")

# ============================================================================
# 3. MODEL INITIALIZATION & TRAINING
# ============================================================================
model = VisualizationNN()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer for faster convergence

print("\nTraining the neural network...")
epochs = 1000
loss_history = []

for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

print("Training completed!")

# ============================================================================
# 4. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_network_architecture(ax):
    """
    Visualize the neural network architecture as a diagram.
    Shows: Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer
    """
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Layer positions
    layers = {
        'Input': (0.7, [2, 3]),
        'Hidden 1\n(8 neurons)': (1.7, np.linspace(1, 9, 8)),
        'Hidden 2\n(8 neurons)': (2.7, np.linspace(1, 9, 8)),
        'Output': (3.3, [5.5])
    }
    
    node_radius = 0.15
    colors = ['#3498db', '#2ecc71', '#2ecc71', '#e74c3c']  # Blue, Green, Green, Red
    
    # Draw nodes
    for layer_idx, (layer_name, (x_pos, y_positions)) in enumerate(layers.items()):
        for y_pos in y_positions:
            circle = patches.Circle((x_pos, y_pos), node_radius, 
                                   color=colors[layer_idx], alpha=0.7, ec='black', linewidth=1.5)
            ax.add_patch(circle)
        
        # Layer label
        ax.text(x_pos, -0.7, layer_name, ha='center', fontsize=9, fontweight='bold')
    
    # Draw connections between all adjacent layers
    layer_items = list(layers.items())
    for i in range(len(layer_items) - 1):
        _, (x1, y1_positions) = layer_items[i]
        _, (x2, y2_positions) = layer_items[i + 1]
        
        for y1 in y1_positions:
            for y2 in y2_positions:
                ax.plot([x1 + node_radius, x2 - node_radius], 
                       [y1, y2], 'gray', alpha=0.2, linewidth=0.6)
    
    ax.set_title('Neural Network Architecture\n(2 → 8 → 8 → 1)', fontsize=12, fontweight='bold', pad=10)

def plot_decision_boundary(ax, model, X_data, y_data):
    """
    Plot the decision boundary learned by the neural network.
    The heatmap shows how the network classifies different regions.
    """
    # Create a mesh grid to evaluate the model on
    h = 0.02  # Step size in mesh
    x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
    y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh grid
    mesh_input = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        mesh_output = model(mesh_input).numpy()
    
    mesh_output = mesh_output.reshape(xx.shape)
    
    # Plot decision boundary as contour
    contour = ax.contourf(xx, yy, mesh_output, levels=20, cmap='RdYlBu_r', alpha=0.8)
    ax.contour(xx, yy, mesh_output, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    
    # Plot training data points
    scatter = ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data.ravel(), 
                        cmap='RdYlBu_r', edgecolors='black', s=30, linewidth=0.5, alpha=0.8)
    
    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.set_title('Decision Boundary Learned by Network', fontsize=12, fontweight='bold')
    plt.colorbar(contour, ax=ax, label='Network Output Probability')

def plot_loss_curve(ax, loss_history):
    """
    Plot training loss over epochs.
    Shows how the network's prediction error decreases over time.
    """
    ax.plot(loss_history, linewidth=2, color='#2c3e50')
    ax.fill_between(range(len(loss_history)), loss_history, alpha=0.3, color='#3498db')
    
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Binary Cross-Entropy Loss', fontsize=10)
    ax.set_title('Training Loss over Epochs', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add final loss annotation
    final_loss = loss_history[-1]
    ax.text(len(loss_history) * 0.7, final_loss * 1.5, 
           f'Final Loss: {final_loss:.6f}', fontsize=10, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# 5. CREATE COMPREHENSIVE VISUALIZATION
# ============================================================================
print("\nGenerating visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Neural Network with Deeper Architecture: How It Learns to Classify Data', 
             fontsize=14, fontweight='bold', y=1.00)

# Plot 1: Network Architecture
plot_network_architecture(axes[0])

# Plot 2: Decision Boundary
plot_decision_boundary(axes[1], model, X.numpy(), y.numpy())

# Plot 3: Training Loss
plot_loss_curve(axes[2], loss_history)

plt.tight_layout()

# Save and display
output_path = 'neural_network_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {output_path}")

plt.show()

# ============================================================================
# 6. SUMMARY & INSIGHTS
# ============================================================================
print("\n" + "="*70)
print("NEURAL NETWORK VISUALIZATION SUMMARY")
print("="*70)
print(f"\nNetwork Architecture: 2 inputs → 8 hidden neurons → 8 hidden neurons → 1 output")
print(f"Training Data: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Training Epochs: {epochs}")
print(f"Final Loss: {loss_history[-1]:.6f}")
print(f"Initial Loss: {loss_history[0]:.6f}")

# Evaluate accuracy
with torch.no_grad():
    predictions = model(X) > 0.5
    accuracy = (predictions.squeeze() == y.squeeze()).float().mean().item()
print(f"Training Accuracy: {accuracy * 100:.2f}%")

print("\nWhat you're seeing in the visualizations:")
print("─" * 70)
print("1. NETWORK ARCHITECTURE:")
print("   Shows the structure: Input layer (2 neurons) → Hidden layer 1 (8 neurons)")
print("   → Hidden layer 2 (8 neurons) → Output layer (1 neuron).")
print("   Gray lines show connections between neurons.")
print()
print("2. DECISION BOUNDARY:")
print("   The heatmap shows how the network classifies different regions.")
print("   Black dashed line = 0.5 probability threshold (decision boundary).")
print("   Points = training data colored by actual class.")
print()
print("3. TRAINING LOSS:")
print("   Loss decreases over epochs, showing the network is learning.")
print("   Lower loss = better predictions on training data.")
print("="*70)
