"""
NBA Optimal Team Selection - Neural Network Model
This script builds and trains a deep neural network to predict team quality scores.
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

print("Using scikit-learn MLPRegressor for neural network")

# Load preprocessed data
print("\n" + "="*80)
print("Loading preprocessed data...")
print("="*80)

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

print(f"Training samples: {X_train.shape}")
print(f"Testing samples: {X_test.shape}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Feature columns: {feature_columns}")

# Define Neural Network Architecture
print("\n" + "="*80)
print("Building Neural Network Architecture")
print("="*80)

print("""
ARCHITECTURE DESIGN:
-------------------
This is a Multi-Layer Perceptron (MLP) designed for regression to predict team quality.

Layer Structure:
  1. Input Layer: 10 neurons (one per feature - automatically handled)
  2. Hidden Layer 1: 128 neurons with ReLU activation
  3. Hidden Layer 2: 64 neurons with ReLU activation
  4. Hidden Layer 3: 32 neurons with ReLU activation
  5. Hidden Layer 4: 16 neurons with ReLU activation
  6. Output Layer: 1 neuron for quality score (0-1)

Rationale:
- Deep architecture (4 hidden layers) allows learning complex patterns
- Decreasing neuron count (128→64→32→16) creates a funnel architecture
- Alpha (L2 regularization) prevents overfitting
- ReLU activation prevents vanishing gradients and learns non-linear patterns
- Adam solver uses adaptive learning rates for faster convergence
- Early stopping prevents overfitting by monitoring validation performance
""")

# Build the Multi-Layer Perceptron model
print("\n" + "="*80)
print("Building MLP Model...")
print("="*80)

model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32, 16),  # 4 hidden layers
    activation='relu',                      # ReLU activation function
    solver='adam',                          # Adam optimizer
    alpha=0.001,                           # L2 regularization
    batch_size=32,                         # Mini-batch size
    learning_rate='adaptive',              # Adaptive learning rate
    learning_rate_init=0.001,              # Initial learning rate
    max_iter=500,                          # Maximum iterations
    shuffle=True,                          # Shuffle training data
    random_state=42,                       # For reproducibility
    early_stopping=True,                   # Enable early stopping
    validation_fraction=0.1,               # Use 10% of training for validation
    n_iter_no_change=15,                   # Patience for early stopping
    verbose=True                           # Show progress
)

print("✓ MLP Model configured")
print("\nModel Architecture:")
print(f"  Input features: {X_train.shape[1]}")
print(f"  Hidden layers: {model.hidden_layer_sizes}")
print(f"  Activation: {model.activation}")
print(f"  Solver: {model.solver}")
print(f"  Learning rate: {model.learning_rate}")
print(f"  Max iterations: {model.max_iter}")
print(f"  Early stopping: {model.early_stopping}")

# Train the model
print("\n" + "="*80)
print("Training the Neural Network...")
print("="*80)

print("""
TRAINING PROCESS:
----------------
The model learns through these steps (repeated for each iteration):

1. FORWARD PROPAGATION:
   - Input features → Hidden Layer 1 → Hidden Layer 2 → Hidden Layer 3 → Hidden Layer 4 → Output
   - Each layer applies: output = ReLU(weights × input + bias)
   - ReLU(x) = max(0, x) - introduces non-linearity

2. LOSS CALCULATION:
   - Compare predicted quality scores with actual quality scores
   - Calculate MSE: loss = mean((predicted - actual)²)

3. BACKPROPAGATION:
   - Calculate gradient of loss with respect to each weight
   - Use chain rule to propagate error backwards through layers
   - Compute: ∂loss/∂weight for all weights

4. WEIGHT UPDATE (Adam Optimizer):
   - Update weights using adaptive learning rates
   - Update rule combines momentum and RMSprop
   - weight_new = weight_old - learning_rate × gradient

5. EARLY STOPPING:
   - Monitor validation loss after each iteration
   - Stop training if no improvement for 15 consecutive iterations
   - Prevents overfitting and saves training time

This process repeats until convergence or max_iter is reached.
""")

# Fit the model
model.fit(X_train, y_train)

print("\n✓ Training complete!")
print(f"  - Total iterations: {model.n_iter_}")
print(f"  - Final loss: {model.loss_:.6f}")

# Evaluate the model
print("\n" + "="*80)
print("Evaluating Model Performance...")
print("="*80)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTRAINING SET PERFORMANCE:")
print(f"  Mean Squared Error (MSE):  {train_mse:.6f}")
print(f"  Mean Absolute Error (MAE): {train_mae:.6f}")
print(f"  R² Score:                  {train_r2:.6f}")

print("\nTEST SET PERFORMANCE:")
print(f"  Mean Squared Error (MSE):  {test_mse:.6f}")
print(f"  Mean Absolute Error (MAE): {test_mae:.6f}")
print(f"  R² Score:                  {test_r2:.6f}")

print("\nINTERPRETATION:")
print(f"  - The model predicts team quality with ~{test_mae:.3f} average error")
print(f"  - R² of {test_r2:.3f} means the model explains {test_r2*100:.1f}% of variance")
if test_r2 > 0.8:
    print(f"  - Excellent performance! (R² > 0.8)")
elif test_r2 > 0.6:
    print(f"  - Good performance (R² > 0.6)")
else:
    print(f"  - Moderate performance (more training data may help)")

# Save the final model
print("\n" + "="*80)
print("Saving the model...")
print("="*80)

with open('team_quality_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Saved: team_quality_model.pkl")

# Save model metrics
metrics = {
    'train_mse': train_mse,
    'test_mse': test_mse,
    'train_mae': train_mae,
    'test_mae': test_mae,
    'train_r2': train_r2,
    'test_r2': test_r2,
    'n_iterations': model.n_iter_,
    'final_loss': model.loss_
}

with open('model_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
print("✓ Saved: model_metrics.pkl")

# Create and save visualization plots
print("\n" + "="*80)
print("Creating visualization plots...")
print("="*80)

# Plot 1: Loss curve (from model's loss_curve_)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

if hasattr(model, 'loss_curve_'):
    ax.plot(model.loss_curve_, linewidth=2, color='blue')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Over Iterations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: training_loss.png")
    plt.close()

# Plot 2: Predictions vs Actual
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Training set
axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=20)
axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Quality Score', fontsize=12)
axes[0].set_ylabel('Predicted Quality Score', fontsize=12)
axes[0].set_title(f'Training Set (R²={train_r2:.3f})', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Test set
axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=20, color='green')
axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Quality Score', fontsize=12)
axes[1].set_ylabel('Predicted Quality Score', fontsize=12)
axes[1].set_title(f'Test Set (R²={test_r2:.3f})', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print("✓ Saved: predictions_vs_actual.png")
plt.close()

# Plot 3: Error distribution
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

errors = y_test.flatten() - y_test_pred.flatten()
ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: error_distribution.png")
plt.close()

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE!")
print("="*80)
print("\nFiles created:")
print("  ✓ team_quality_model.pkl (trained MLP model)")
print("  ✓ model_metrics.pkl (performance metrics)")
print("  ✓ training_loss.png (loss curve)")
print("  ✓ predictions_vs_actual.png (prediction accuracy)")
print("  ✓ error_distribution.png (error analysis)")
print("\nNext step: Create Streamlit app for team selection!")