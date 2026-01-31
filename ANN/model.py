"""
NBA Optimal Team Selection - Neural Network Model Training Script
=================================================================

Purpose:
    This script trains a Multi-Layer Perceptron (MLP) neural network to predict
    the quality of 5-player basketball teams. The model learns patterns from
    10,000 synthetic team combinations and their quality scores.

Process:
    1. Load preprocessed data from data_preparation.py
    2. Define MLP architecture (4 hidden layers)
    3. Train model using Adam optimizer with backpropagation
    4. Evaluate performance on training and test sets
    5. Generate visualization plots
    6. Save trained model for deployment

Author: [Your Name]
Date: January 31, 2026
Course: AIT-204 - Topic 2 Assignment
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np                    # Numerical computing
import pandas as pd                   # Data manipulation
import pickle                         # Save/load Python objects
import matplotlib                     # Plotting library
matplotlib.use('Agg')                 # Use non-interactive backend
import matplotlib.pyplot as plt       # Plotting interface
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Evaluation metrics
from sklearn.neural_network import MLPRegressor  # Multi-Layer Perceptron
import warnings
warnings.filterwarnings('ignore')     # Suppress sklearn warnings

print("="*80)
print("NBA OPTIMAL TEAM SELECTION - NEURAL NETWORK TRAINING")
print("="*80)
print("\nUsing scikit-learn MLPRegressor for neural network implementation")

# ============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING PREPROCESSED DATA")
print("="*80)

# Load training and testing data created by data_preparation.py
# These files contain normalized team features and quality scores

X_train = np.load('X_train.npy')    # Training input features (8000, 10)
X_test = np.load('X_test.npy')      # Testing input features (2000, 10)
y_train = np.load('y_train.npy')    # Training labels (8000,)
y_test = np.load('y_test.npy')      # Testing labels (2000,)

# Load feature column names for reference
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

print(f"Training data loaded successfully:")
print(f"  X_train shape: {X_train.shape} (samples, features)")
print(f"  y_train shape: {y_train.shape} (samples,)")
print(f"  X_test shape:  {X_test.shape} (samples, features)")
print(f"  y_test shape:  {y_test.shape} (samples,)")

print(f"\nFeature columns ({len(feature_columns)} total):")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i}. {col}")

print(f"\nData statistics:")
print(f"  Training set size: {X_train.shape[0]} samples")
print(f"  Testing set size:  {X_test.shape[0]} samples")
print(f"  Features per sample: {X_train.shape[1]}")
print(f"  Total data points: {X_train.shape[0] * X_train.shape[1]:,}")

# ============================================================================
# STEP 2: DEFINE NEURAL NETWORK ARCHITECTURE
# ============================================================================

print("\n" + "="*80)
print("STEP 2: DEFINING NEURAL NETWORK ARCHITECTURE")
print("="*80)

print("""
ARCHITECTURE DESIGN:
-------------------
Multi-Layer Perceptron (MLP) for Regression

Layer Structure:
  Input Layer:     10 neurons (one per feature - automatically handled)
  Hidden Layer 1:  128 neurons with ReLU activation
  Hidden Layer 2:  64 neurons with ReLU activation
  Hidden Layer 3:  32 neurons with ReLU activation
  Hidden Layer 4:  16 neurons with ReLU activation
  Output Layer:    1 neuron for quality score prediction

Activation Functions:
  - ReLU (Rectified Linear Unit) for hidden layers
    f(x) = max(0, x)
    Advantages: Prevents vanishing gradient, computationally efficient
  
  - Linear activation for output layer
    Allows continuous predictions in range [0, 1]

Rationale:
  - Deep architecture (4 hidden layers) allows learning complex patterns
  - Decreasing neuron count (128→64→32→16) creates funnel architecture
    that progressively compresses information
  - Each layer can learn increasingly abstract representations:
    Layer 1: Basic feature combinations
    Layer 2: Player role patterns
    Layer 3: Team balance indicators
    Layer 4: Optimal team signatures
  - L2 regularization (alpha) prevents overfitting
  - Early stopping monitors validation performance to prevent overtraining
""")

# Create the MLPRegressor model with specified architecture
# MLPRegressor uses backpropagation and stochastic gradient descent for training
model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32, 16),  # Tuple defining # of neurons in each hidden layer
    activation='relu',                      # Activation function for hidden layers
    solver='adam',                          # Optimizer: Adam (Adaptive Moment Estimation)
    alpha=0.001,                           # L2 regularization parameter (weight decay)
    batch_size=32,                         # Number of samples per gradient update
    learning_rate='adaptive',              # Learning rate schedule
    learning_rate_init=0.001,              # Initial learning rate for Adam
    max_iter=500,                          # Maximum number of training iterations
    shuffle=True,                          # Shuffle training data each iteration
    random_state=42,                       # Random seed for reproducibility
    early_stopping=True,                   # Stop training when validation score stops improving
    validation_fraction=0.1,               # Fraction of training data used for validation
    n_iter_no_change=15,                   # # of iterations with no improvement before stopping
    verbose=True                           # Print progress during training
)

print("✓ MLP Model configured successfully")
print("\nModel Hyperparameters:")
print(f"  Input features:        {X_train.shape[1]}")
print(f"  Hidden layer sizes:    {model.hidden_layer_sizes}")
print(f"  Activation function:   {model.activation}")
print(f"  Optimizer:             {model.solver}")
print(f"  Learning rate:         {model.learning_rate} (initial: {model.learning_rate_init})")
print(f"  Batch size:            {model.batch_size}")
print(f"  L2 regularization:     {model.alpha}")
print(f"  Max iterations:        {model.max_iter}")
print(f"  Early stopping:        {model.early_stopping}")
print(f"  Validation fraction:   {model.validation_fraction}")
print(f"  Patience:              {model.n_iter_no_change}")

# Calculate approximate number of parameters
# Formula: (input × hidden1 + hidden1) + (hidden1 × hidden2 + hidden2) + ...
n_params = (
    (10 * 128 + 128) +      # Input to Layer 1
    (128 * 64 + 64) +       # Layer 1 to Layer 2
    (64 * 32 + 32) +        # Layer 2 to Layer 3
    (32 * 16 + 16) +        # Layer 3 to Layer 4
    (16 * 1 + 1)            # Layer 4 to Output
)
print(f"\nEstimated # of parameters: ~{n_params:,}")

# ============================================================================
# STEP 3: TRAIN THE NEURAL NETWORK
# ============================================================================

print("\n" + "="*80)
print("STEP 3: TRAINING THE NEURAL NETWORK")
print("="*80)

print("""
TRAINING PROCESS:
----------------
The neural network learns through these iterative steps:

1. FORWARD PROPAGATION:
   For each training sample:
   a) Input features pass through each layer sequentially
   b) At each layer: z = W·x + b (linear transformation)
   c) Apply activation: a = ReLU(z) = max(0, z)
   d) Output layer produces prediction: ŷ
   
   Mathematical flow:
   x → [ReLU(W₁x + b₁)] → [ReLU(W₂a₁ + b₂)] → ... → ŷ

2. LOSS CALCULATION:
   Compute Mean Squared Error between predictions and true values:
   L = (1/n) × Σ(ŷᵢ - yᵢ)²
   
   This measures how far off our predictions are from reality.

3. BACKPROPAGATION:
   Calculate gradients of loss with respect to all weights:
   ∂L/∂W = ∂L/∂a × ∂a/∂z × ∂z/∂W  (chain rule)
   
   Gradients flow backwards through network:
   Output → Layer 4 → Layer 3 → Layer 2 → Layer 1
   
   For each weight, we compute how much it contributed to the error.

4. WEIGHT UPDATE (Adam Optimizer):
   Adam combines two ideas:
   - Momentum: Smooth out updates using exponential moving average
   - RMSprop: Adapt learning rate for each parameter
   
   Update rule:
   m = β₁·m + (1-β₁)·g        (momentum)
   v = β₂·v + (1-β₂)·g²       (RMSprop)
   W = W - α·m/√(v + ε)       (parameter update)
   
   Where:
   - g = gradient
   - α = learning rate
   - β₁, β₂ = decay rates (typically 0.9, 0.999)
   - ε = small constant for numerical stability

5. VALIDATION:
   After each iteration:
   - Compute loss on validation set (10% of training data)
   - If no improvement for 15 consecutive iterations → STOP
   - This prevents overfitting

This entire process repeats until:
- Early stopping is triggered (no improvement for 15 iterations)
- OR maximum iterations (500) is reached
- OR loss becomes sufficiently small

Training begins now...
""")

import time
start_time = time.time()

# Train the model
# .fit() performs the complete training process described above
model.fit(X_train, y_train)

end_time = time.time()
training_time = end_time - start_time

print("\n" + "="*80)
print("✓ TRAINING COMPLETE!")
print("="*80)
print(f"  Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
print(f"  Iterations completed: {model.n_iter_}")
print(f"  Final training loss: {model.loss_:.6f}")
print(f"  Early stopping triggered: {'Yes' if model.n_iter_ < model.max_iter else 'No'}")

# ============================================================================
# STEP 4: EVALUATE MODEL PERFORMANCE
# ============================================================================

print("\n" + "="*80)
print("STEP 4: EVALUATING MODEL PERFORMANCE")
print("="*80)

# Make predictions on both training and test sets
print("Making predictions...")
y_train_pred = model.predict(X_train)  # Predictions on training data
y_test_pred = model.predict(X_test)    # Predictions on test data

# Calculate performance metrics
# These metrics tell us how well the model performs

# 1. Mean Squared Error (MSE): Average squared difference between predictions and actual
#    Lower is better. Penalizes large errors more than small errors.
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# 2. Mean Absolute Error (MAE): Average absolute difference between predictions and actual
#    Lower is better. Easier to interpret than MSE (same units as output).
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

# 3. R² Score (Coefficient of Determination): Proportion of variance explained by model
#    Ranges from 0 to 1 (can be negative if model is very bad).
#    1.0 = perfect predictions, 0.0 = model no better than predicting mean
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Display results
print("\n" + "-"*80)
print("TRAINING SET PERFORMANCE:")
print("-"*80)
print(f"  Mean Squared Error (MSE):  {train_mse:.6f}")
print(f"    → Average squared error: {train_mse:.4f}")
print(f"  Mean Absolute Error (MAE): {train_mae:.6f}")
print(f"    → On average, predictions are off by {train_mae:.4f} (on 0-1 scale)")
print(f"  R² Score:                  {train_r2:.6f}")
print(f"    → Model explains {train_r2*100:.2f}% of variance in training data")

print("\n" + "-"*80)
print("TEST SET PERFORMANCE:")
print("-"*80)
print(f"  Mean Squared Error (MSE):  {test_mse:.6f}")
print(f"    → Average squared error: {test_mse:.4f}")
print(f"  Mean Absolute Error (MAE): {test_mae:.6f}")
print(f"    → On average, predictions are off by {test_mae:.4f} (on 0-1 scale)")
print(f"  R² Score:                  {test_r2:.6f}")
print(f"    → Model explains {test_r2*100:.2f}% of variance in test data")

# Interpret results
print("\n" + "-"*80)
print("INTERPRETATION:")
print("-"*80)
print(f"  ✓ Average prediction error: ±{test_mae:.3f} on quality scale (0-1)")
print(f"  ✓ Typical error: ±{np.sqrt(test_mse):.3f} (RMSE)")
print(f"  ✓ Variance explained: {test_r2*100:.1f}%")

# Check for overfitting by comparing train vs test performance
overfit_r2 = train_r2 - test_r2
if overfit_r2 < 0.05:
    print(f"  ✓ Minimal overfitting detected (R² difference: {overfit_r2:.3f})")
elif overfit_r2 < 0.10:
    print(f"  ⚠ Slight overfitting detected (R² difference: {overfit_r2:.3f})")
else:
    print(f"  ✗ Significant overfitting detected (R² difference: {overfit_r2:.3f})")

# Performance classification
if test_r2 > 0.8:
    print(f"  ✓ Excellent model performance (R² > 0.8)")
elif test_r2 > 0.6:
    print(f"  ✓ Good model performance (R² > 0.6)")
elif test_r2 > 0.4:
    print(f"  ⚠ Moderate model performance (R² > 0.4)")
else:
    print(f"  ✗ Poor model performance (R² < 0.4)")

# ============================================================================
# STEP 5: SAVE THE MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 5: SAVING THE TRAINED MODEL")
print("="*80)

# Save the trained model using pickle
# This allows us to load and use the model later without retraining
with open('team_quality_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Saved: team_quality_model.pkl")

# Save model performance metrics
metrics = {
    'train_mse': float(train_mse),
    'test_mse': float(test_mse),
    'train_mae': float(train_mae),
    'test_mae': float(test_mae),
    'train_r2': float(train_r2),
    'test_r2': float(test_r2),
    'n_iterations': int(model.n_iter_),
    'final_loss': float(model.loss_),
    'training_time': float(training_time)
}

with open('model_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
print("✓ Saved: model_metrics.pkl")

# ============================================================================
# STEP 6: CREATE VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 6: CREATING VISUALIZATION PLOTS")
print("="*80)

# -------------------------
# Plot 1: Training Loss Curve
# -------------------------
if hasattr(model, 'loss_curve_'):
    print("Creating training loss curve...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot loss over iterations
    iterations = range(1, len(model.loss_curve_) + 1)
    ax.plot(iterations, model.loss_curve_, linewidth=2, color='#1f77b4', label='Training Loss')
    
    # Formatting
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (Mean Squared Error)', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Over Iterations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotation for final loss
    final_iter = len(model.loss_curve_)
    final_loss = model.loss_curve_[-1]
    ax.annotate(f'Final Loss: {final_loss:.6f}\nIteration: {final_iter}',
                xy=(final_iter, final_loss),
                xytext=(final_iter * 0.7, final_loss * 1.5),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: training_loss.png")

# -------------------------
# Plot 2: Predictions vs Actual
# -------------------------
print("Creating predictions vs. actual scatter plots...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Training set
axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=20, color='#1f77b4', edgecolors='none')
axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)
axes[0].set_xlabel('Actual Quality Score', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted Quality Score', fontsize=12, fontweight='bold')
axes[0].set_title(f'Training Set\n(R² = {train_r2:.3f}, MAE = {train_mae:.4f})', 
                   fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])

# Test set
axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=20, color='#2ca02c', edgecolors='none')
axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)
axes[1].set_xlabel('Actual Quality Score', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Predicted Quality Score', fontsize=12, fontweight='bold')
axes[1].set_title(f'Test Set\n(R² = {test_r2:.3f}, MAE = {test_mae:.4f})', 
                   fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: predictions_vs_actual.png")

# -------------------------
# Plot 3: Error Distribution
# -------------------------
print("Creating error distribution histogram...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Calculate errors (residuals)
errors = y_test.flatten() - y_test_pred.flatten()

# Create histogram
n, bins, patches = ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')

# Add vertical line at zero
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error', alpha=0.7)

# Add mean and std annotations
mean_error = errors.mean()
std_error = errors.std()
ax.axvline(x=mean_error, color='green', linestyle=':', linewidth=2, 
           label=f'Mean = {mean_error:.4f}', alpha=0.7)

# Formatting
ax.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Prediction Errors (Test Set)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add text box with statistics
textstr = f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}\nMAE: {test_mae:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: error_distribution.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE!")
print("="*80)

print("\nFiles created:")
print("  ✓ team_quality_model.pkl - Trained neural network model (298 KB)")
print("  ✓ model_metrics.pkl - Performance metrics (272 bytes)")
print("  ✓ training_loss.png - Loss curve visualization")
print("  ✓ predictions_vs_actual.png - Prediction accuracy plots")
print("  ✓ error_distribution.png - Error analysis histogram")

print("\nModel Summary:")
print(f"  Architecture: {model.hidden_layer_sizes}")
print(f"  Parameters: ~{n_params:,}")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")
print(f"  Training time: {training_time:.2f} seconds")
print(f"  Iterations: {model.n_iter_}")
print(f"  Final loss: {model.loss_:.6f}")
print(f"  Test R²: {test_r2:.4f}")
print(f"  Test MAE: {test_mae:.4f}")

print("\nNext step: Run app.py to launch the Streamlit application!")
print("Or deploy to Streamlit Cloud for public access.")

# END OF SCRIPT