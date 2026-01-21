import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

'''
each function gets called by something in app.py
'''

class LinearRegression:
    def __init__(self, learning_rate, n_iterations, random_seed=None):
        #init variables
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Random initialization of parameters
        # Using small random values from standard normal distribution
        self.slope = np.random.randn()
        self.intercept = np.random.randn()
        
        # Store initial values for documentation and visualization
        self.initial_slope = self.slope
        self.initial_intercept = self.intercept
        
        self.history = {
            'slope': [],
            'intercept': [],
            'grad_slope': [],
            'grad_intercept': [],
            'train_loss': [],      # Track training loss
            'val_loss': [],        # Track validation loss
            'grad_magnitude': [],  # Track gradient magnitude
            'param_change': []     # Track parameter change magnitude
        }

        #empty train/test variables
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

        #initialize data
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(
            BASE_DIR,
            "data",
            "synthetic_data_Simple_Linear.csv"
        )

        self.data = pd.read_csv(data_path)


    def data_split(self, test_size=0.2, random_state=42):
        #split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data["x"],
            self.data["y"],
            test_size=test_size,
            random_state=random_state
        )

    def calculate_mse(self, y_true, y_pred):
        """
        Calculate Mean Squared Error (MSE) loss function
        
        MSE = (1/n) * Σ(y_true - y_pred)²
        
        Args:
            y_true: actual target values
            y_pred: predicted values
            
        Returns:
            float: mean squared error
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def calculate_errors(self, y_true, y_pred):
        """
        Calculate individual prediction errors (residuals)
        
        Error = y_true - y_pred
        
        Args:
            y_true: actual target values
            y_pred: predicted values
            
        Returns:
            numpy array: individual errors for each prediction
        """
        return y_true - y_pred
    
    def calculate_mae(self, y_true, y_pred):
        """
        Calculate Mean Absolute Error (MAE)
        
        MAE = (1/n) * Σ|y_true - y_pred|
        
        Args:
            y_true: actual target values
            y_pred: predicted values
            
        Returns:
            float: mean absolute error
        """
        return np.mean(np.abs(y_true - y_pred))
    
    def calculate_rmse(self, y_true, y_pred):
        """
        Calculate Root Mean Squared Error (RMSE)
        
        RMSE = √(MSE) = √[(1/n) * Σ(y_true - y_pred)²]
        
        Args:
            y_true: actual target values
            y_pred: predicted values
            
        Returns:
            float: root mean squared error
        """
        return np.sqrt(self.calculate_mse(y_true, y_pred))

    def fit(self, method='batch', batch_size=32):
        """
        Main training method that calls the appropriate gradient descent variant
        
        Args:
            method: 'batch', 'mini-batch', or 'stochastic'
            batch_size: size of mini-batches (only used for mini-batch method)
        
        Returns:
            self (for method chaining)
        """
        if method == 'batch':
            return self.fit_batch()
        elif method == 'mini-batch':
            return self.fit_mini_batch(batch_size)
        elif method == 'stochastic':
            return self.fit_stochastic_grad()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'batch', 'mini-batch', or 'stochastic'")

    def fit_batch(self):
        """
        Batch Gradient Descent: Uses ALL training data in each iteration
        
        Advantages:
        - Stable, smooth convergence
        - Guaranteed to converge to global minimum (for convex problems)
        - Less noisy gradient estimates
        
        Disadvantages:
        - Slow for large datasets
        - May get stuck in local minima (for non-convex problems)
        - Requires loading entire dataset into memory
        """
        X_train = np.array(self.X_train).flatten()
        y_train = np.array(self.y_train).flatten()
        X_val = np.array(self.X_test).flatten()
        y_val = np.array(self.y_test).flatten()
        
        n = len(X_train)  # number of training samples

        for iteration in range(1, self.n_iterations + 1):
            # Store previous parameters for change tracking
            prev_slope = self.slope
            prev_intercept = self.intercept
            
            # Forward pass: make predictions on ALL data
            y_pred_train = self.slope * X_train + self.intercept
            
            # Calculate loss on training set
            train_loss = self.calculate_mse(y_train, y_pred_train)
            
            # Calculate loss on validation set
            y_pred_val = self.slope * X_val + self.intercept
            val_loss = self.calculate_mse(y_val, y_pred_val)
            
            # Compute gradients using ALL training data
            grad_slope = (1/n) * np.sum((y_pred_train - y_train) * X_train)
            grad_intercept = (1/n) * np.sum(y_pred_train - y_train)
            
            # Calculate gradient magnitude (L2 norm)
            grad_magnitude = np.sqrt(grad_slope**2 + grad_intercept**2)

            # Update parameters using gradient descent
            self.slope = self.slope - self.learning_rate * grad_slope
            self.intercept = self.intercept - self.learning_rate * grad_intercept
            
            # Calculate parameter change magnitude
            param_change = np.sqrt((self.slope - prev_slope)**2 + 
                                  (self.intercept - prev_intercept)**2)

            # Track history
            self.history["slope"].append(self.slope)
            self.history["intercept"].append(self.intercept)
            self.history["grad_slope"].append(grad_slope)
            self.history["grad_intercept"].append(grad_intercept)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["grad_magnitude"].append(grad_magnitude)
            self.history["param_change"].append(param_change)
        
        return self

    def fit_mini_batch(self, batch_size=32):
        """
        Mini-Batch Gradient Descent: Uses small batches of data in each iteration
        
        Advantages:
        - Balances speed and stability
        - More memory efficient than batch GD
        - Can escape shallow local minima
        - Leverages vectorization for efficiency
        
        Disadvantages:
        - Introduces some noise (less than SGD)
        - Requires tuning batch size hyperparameter
        - May oscillate near minimum
        
        Args:
            batch_size: number of samples per mini-batch (default: 32)
        """
        X_train = np.array(self.X_train).flatten()
        y_train = np.array(self.y_train).flatten()
        X_val = np.array(self.X_test).flatten()
        y_val = np.array(self.y_test).flatten()
        
        n = len(X_train)
        
        # Ensure batch_size doesn't exceed dataset size
        batch_size = min(batch_size, n)

        for iteration in range(1, self.n_iterations + 1):
            # Store previous parameters
            prev_slope = self.slope
            prev_intercept = self.intercept
            
            # Shuffle data at the start of each epoch
            indices = np.random.permutation(n)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Process mini-batches
            num_batches = n // batch_size
            
            # Accumulate gradients across all mini-batches in this iteration
            total_grad_slope = 0
            total_grad_intercept = 0
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass on mini-batch
                y_pred_batch = self.slope * X_batch + self.intercept
                
                # Compute gradients for this mini-batch
                batch_grad_slope = (1/batch_size) * np.sum((y_pred_batch - y_batch) * X_batch)
                batch_grad_intercept = (1/batch_size) * np.sum(y_pred_batch - y_batch)
                
                # Update parameters after each mini-batch
                self.slope = self.slope - self.learning_rate * batch_grad_slope
                self.intercept = self.intercept - self.learning_rate * batch_grad_intercept
                
                total_grad_slope += batch_grad_slope
                total_grad_intercept += batch_grad_intercept
            
            # Average gradients across all mini-batches for tracking
            avg_grad_slope = total_grad_slope / num_batches
            avg_grad_intercept = total_grad_intercept / num_batches
            
            # Calculate full training and validation loss (for tracking only)
            y_pred_train = self.slope * X_train + self.intercept
            train_loss = self.calculate_mse(y_train, y_pred_train)
            
            y_pred_val = self.slope * X_val + self.intercept
            val_loss = self.calculate_mse(y_val, y_pred_val)
            
            # Calculate gradient magnitude
            grad_magnitude = np.sqrt(avg_grad_slope**2 + avg_grad_intercept**2)
            
            # Calculate parameter change
            param_change = np.sqrt((self.slope - prev_slope)**2 + 
                                  (self.intercept - prev_intercept)**2)
            
            # Track history
            self.history["slope"].append(self.slope)
            self.history["intercept"].append(self.intercept)
            self.history["grad_slope"].append(avg_grad_slope)
            self.history["grad_intercept"].append(avg_grad_intercept)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["grad_magnitude"].append(grad_magnitude)
            self.history["param_change"].append(param_change)
        
        return self

    def fit_stochastic_grad(self):
        """
        Stochastic Gradient Descent (SGD): Uses ONE sample at a time
        
        Advantages:
        - Very fast updates (one sample per update)
        - Can escape local minima due to noise
        - Online learning capable
        - Memory efficient
        
        Disadvantages:
        - Very noisy gradient estimates
        - May never fully converge (oscillates near minimum)
        - Slower convergence overall despite fast updates
        - Requires careful learning rate tuning
        """
        X_train = np.array(self.X_train).flatten()
        y_train = np.array(self.y_train).flatten()
        X_val = np.array(self.X_test).flatten()
        y_val = np.array(self.y_test).flatten()
        
        n = len(X_train)
        
        # In SGD, we typically do multiple passes through the data
        # Each iteration = one epoch (pass through all data)
        for iteration in range(1, self.n_iterations + 1):
            # Store previous parameters
            prev_slope = self.slope
            prev_intercept = self.intercept
            
            # Shuffle data at the start of each epoch
            indices = np.random.permutation(n)
            
            # Accumulate gradients for tracking purposes
            epoch_grad_slopes = []
            epoch_grad_intercepts = []
            
            # Process each sample individually
            for idx in indices:
                x_sample = X_train[idx]
                y_sample = y_train[idx]
                
                # Forward pass on single sample
                y_pred_sample = self.slope * x_sample + self.intercept
                
                # Compute gradient for this single sample
                # Note: No division by n since we're using one sample
                grad_slope = (y_pred_sample - y_sample) * x_sample
                grad_intercept = (y_pred_sample - y_sample)
                
                # Update parameters immediately after each sample
                self.slope = self.slope - self.learning_rate * grad_slope
                self.intercept = self.intercept - self.learning_rate * grad_intercept
                
                epoch_grad_slopes.append(grad_slope)
                epoch_grad_intercepts.append(grad_intercept)
            
            # Average gradients across epoch for tracking
            avg_grad_slope = np.mean(epoch_grad_slopes)
            avg_grad_intercept = np.mean(epoch_grad_intercepts)
            
            # Calculate full training and validation loss (for tracking only)
            y_pred_train = self.slope * X_train + self.intercept
            train_loss = self.calculate_mse(y_train, y_pred_train)
            
            y_pred_val = self.slope * X_val + self.intercept
            val_loss = self.calculate_mse(y_val, y_pred_val)
            
            # Calculate gradient magnitude
            grad_magnitude = np.sqrt(avg_grad_slope**2 + avg_grad_intercept**2)
            
            # Calculate parameter change
            param_change = np.sqrt((self.slope - prev_slope)**2 + 
                                  (self.intercept - prev_intercept)**2)
            
            # Track history
            self.history["slope"].append(self.slope)
            self.history["intercept"].append(self.intercept)
            self.history["grad_slope"].append(avg_grad_slope)
            self.history["grad_intercept"].append(avg_grad_intercept)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["grad_magnitude"].append(grad_magnitude)
            self.history["param_change"].append(param_change)
        
        return self
    
    def predict(self, X):
        #makes predictions
        X = np.array(X).flatten()
        return self.slope * X + self.intercept

    def calc_metrics(self):
        #uses R^2, MSE, RMSE, and MAE to evaluate the model
        X = np.array(self.X_test).flatten()
        y = np.array(self.y_test).flatten()

        y_pred = self.predict(X)
        
        #calc the metrics
        #R^2
        ss_res = np.sum((y - y_pred) ** 2)
        ss_total = np.sum((y - y.mean()) ** 2)
        r2 = 1 - (ss_res/ss_total)

        #MSE
        mse = np.mean((y - y_pred) ** 2)

        #RMSE
        rmse = np.sqrt(mse)

        #MAE
        mae = np.mean(np.abs(y - y_pred))

        return {
            'R^2': r2,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae
        }
    
    def get_history(self):
        return self.history
    
    def get_data(self):
        return self.data
    
    def get_final_loss(self):
        """
        Returns the final training and validation loss values
        """
        if len(self.history['train_loss']) > 0:
            return {
                'final_train_loss': self.history['train_loss'][-1],
                'final_val_loss': self.history['val_loss'][-1]
            }
        return None
    
    def get_error_statistics(self):
        """
        Calculate comprehensive error statistics for training and validation sets
        
        Returns:
            dict: error statistics including MSE, MAE, RMSE, and individual errors
        """
        X_train = np.array(self.X_train).flatten()
        y_train = np.array(self.y_train).flatten()
        X_val = np.array(self.X_test).flatten()
        y_val = np.array(self.y_test).flatten()
        
        # Generate predictions
        y_train_pred = self.predict(X_train)
        y_val_pred = self.predict(X_val)
        
        # Calculate errors
        train_errors = self.calculate_errors(y_train, y_train_pred)
        val_errors = self.calculate_errors(y_val, y_val_pred)
        
        return {
            'train': {
                'errors': train_errors,
                'mse': self.calculate_mse(y_train, y_train_pred),
                'mae': self.calculate_mae(y_train, y_train_pred),
                'rmse': self.calculate_rmse(y_train, y_train_pred),
                'predictions': y_train_pred,
                'actual': y_train
            },
            'validation': {
                'errors': val_errors,
                'mse': self.calculate_mse(y_val, y_val_pred),
                'mae': self.calculate_mae(y_val, y_val_pred),
                'rmse': self.calculate_rmse(y_val, y_val_pred),
                'predictions': y_val_pred,
                'actual': y_val
            }
        }
    
    def get_initial_params(self):
        """
        Returns the initial parameter values before training
        """
        return {
            'initial_slope': self.initial_slope,
            'initial_intercept': self.initial_intercept
        }
    
    def predict_with_params(self, X, slope, intercept):
        """
        Make predictions using specific parameter values
        Useful for visualizing initial predictions before training
        
        Args:
            X: input values
            slope: slope parameter to use
            intercept: intercept parameter to use
            
        Returns:
            predictions using specified parameters
        """
        X = np.array(X).flatten()
        return slope * X + intercept
    
    def reset_parameters(self, random_seed=None):
        """
        Reset parameters to initial random values for fair comparison
        
        Args:
            random_seed: seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.slope = np.random.randn()
        self.intercept = np.random.randn()
        self.initial_slope = self.slope
        self.initial_intercept = self.intercept
        
        # Clear history
        self.history = {
            'slope': [],
            'intercept': [],
            'grad_slope': [],
            'grad_intercept': [],
            'train_loss': [],
            'val_loss': [],
            'grad_magnitude': [],
            'param_change': []
        }
    
    def compare_optimization_methods(self, methods=['batch', 'mini-batch', 'stochastic'], 
                                    batch_size=32, seed=42):
        """
        Compare different optimization methods starting from same initial parameters
        
        Args:
            methods: list of methods to compare
            batch_size: batch size for mini-batch method
            seed: random seed for reproducibility
            
        Returns:
            dict: results for each method
        """
        results = {}
        
        # Store original parameters
        original_slope = self.slope
        original_intercept = self.intercept
        original_lr = self.learning_rate
        original_iters = self.n_iterations
        
        for method in methods:
            # Reset to same initial parameters
            self.reset_parameters(seed)
            
            # Train with this method
            if method == 'batch':
                self.fit_batch()
            elif method == 'mini-batch':
                self.fit_mini_batch(batch_size)
            elif method == 'stochastic':
                self.fit_stochastic_grad()
            
            # Store results
            results[method] = {
                'history': self.get_history().copy(),
                'final_slope': self.slope,
                'final_intercept': self.intercept,
                'final_train_loss': self.history['train_loss'][-1],
                'final_val_loss': self.history['val_loss'][-1],
                'metrics': self.calc_metrics()
            }
        
        # Restore original state (use first method's result)
        if methods:
            self.slope = results[methods[0]]['final_slope']
            self.intercept = results[methods[0]]['final_intercept']
            self.history = results[methods[0]]['history']
        
        return results
    
    def compute_analytical_gradients(self, w, b):
        """
        Compute analytical gradients using calculus
        
        For MSE loss L = (1/n) * Σ(yᵢ - ŷᵢ)²:
        ∂L/∂w = (1/n) * Σ(ŷᵢ - yᵢ) * xᵢ
        ∂L/∂b = (1/n) * Σ(ŷᵢ - yᵢ)
        
        Args:
            w: slope parameter
            b: intercept parameter
            
        Returns:
            tuple: (grad_w, grad_b)
            """
        X_train = np.array(self.X_train).flatten()
        y_train = np.array(self.y_train).flatten()
        n = len(X_train)
        
        # Forward pass
        y_pred = w * X_train + b
        
        # Compute gradients
        grad_w = (1/n) * np.sum((y_pred - y_train) * X_train)
        grad_b = (1/n) * np.sum(y_pred - y_train)
    
        return grad_w, grad_b

    def compute_numerical_gradients(self, w, b, epsilon=1e-7):
        """
        Compute numerical gradients using finite differences
        
        Numerical gradient approximation:
        ∂L/∂w ≈ [L(w + ε) - L(w - ε)] / (2ε)  (central difference)
        
        Args:
            w: slope parameter
            b: intercept parameter
            epsilon: small perturbation value
            
        Returns:
            tuple: (numerical_grad_w, numerical_grad_b)
        """
        X_train = np.array(self.X_train).flatten()
        y_train = np.array(self.y_train).flatten()
        
        # Compute loss at w + epsilon
        y_pred_w_plus = (w + epsilon) * X_train + b
        loss_w_plus = self.calculate_mse(y_train, y_pred_w_plus)
        
        # Compute loss at w - epsilon
        y_pred_w_minus = (w - epsilon) * X_train + b
        loss_w_minus = self.calculate_mse(y_train, y_pred_w_minus)
        
        # Numerical gradient for w (central difference)
        numerical_grad_w = (loss_w_plus - loss_w_minus) / (2 * epsilon)
        
        # Compute loss at b + epsilon
        y_pred_b_plus = w * X_train + (b + epsilon)
        loss_b_plus = self.calculate_mse(y_train, y_pred_b_plus)
        
        # Compute loss at b - epsilon
        y_pred_b_minus = w * X_train + (b - epsilon)
        loss_b_minus = self.calculate_mse(y_train, y_pred_b_minus)
        
        # Numerical gradient for b (central difference)
        numerical_grad_b = (loss_b_plus - loss_b_minus) / (2 * epsilon)
        
        return numerical_grad_w, numerical_grad_b

    def verify_gradients(self, epsilon=1e-7):
        """
        Verify analytical gradients against numerical gradients
        
        Returns:
            dict: comparison of analytical vs numerical gradients
        """
        # Use current parameters
        w = self.slope
        b = self.intercept
        
        # Compute both types of gradients
        analytical_grad_w, analytical_grad_b = self.compute_analytical_gradients(w, b)
        numerical_grad_w, numerical_grad_b = self.compute_numerical_gradients(w, b, epsilon)
        
        # Compute relative errors
        # Relative error = |analytical - numerical| / max(|analytical|, |numerical|)
        rel_error_w = abs(analytical_grad_w - numerical_grad_w) / max(abs(analytical_grad_w), abs(numerical_grad_w), 1e-10)
        rel_error_b = abs(analytical_grad_b - numerical_grad_b) / max(abs(analytical_grad_b), abs(numerical_grad_b), 1e-10)
        
        return {
            'analytical': {
                'grad_w': analytical_grad_w,
                'grad_b': analytical_grad_b
            },
            'numerical': {
                'grad_w': numerical_grad_w,
                'grad_b': numerical_grad_b
            },
            'absolute_difference': {
                'grad_w': abs(analytical_grad_w - numerical_grad_w),
                'grad_b': abs(analytical_grad_b - numerical_grad_b)
            },
            'relative_error': {
                'grad_w': rel_error_w,
                'grad_b': rel_error_b
            },
            'epsilon': epsilon
        }
    
    def demonstrate_large_gradients(self, extreme_learning_rate=10.0, iterations=50):
        """
        Demonstrate the effects of large gradients through gradient explosion
        
        Uses an extremely high learning rate to show unstable training behavior
        
        Args:
            extreme_learning_rate: very large learning rate to cause instability
            iterations: number of iterations to run
            
        Returns:
            dict: history of unstable training
        """
        # Store original parameters
        original_lr = self.learning_rate
        original_slope = self.slope
        original_intercept = self.intercept
        
        # Use extreme learning rate
        self.learning_rate = extreme_learning_rate
        
        # Reset to initial random parameters
        np.random.seed(42)
        self.slope = np.random.randn()
        self.intercept = np.random.randn()
        
        # Track unstable training
        unstable_history = {
            'slope': [],
            'intercept': [],
            'grad_slope': [],
            'grad_intercept': [],
            'train_loss': [],
            'grad_magnitude': [],
            'exploded': False,
            'explosion_iteration': None
        }
        
        X_train = np.array(self.X_train).flatten()
        y_train = np.array(self.y_train).flatten()
        n = len(X_train)
        
        for iteration in range(iterations):
            # Forward pass
            y_pred = self.slope * X_train + self.intercept
            
            # Calculate loss
            train_loss = self.calculate_mse(y_train, y_pred)
            
            # Check for explosion (NaN or very large values)
            if np.isnan(train_loss) or np.isinf(train_loss) or train_loss > 1e10:
                unstable_history['exploded'] = True
                unstable_history['explosion_iteration'] = iteration + 1
                break
            
            # Compute gradients
            grad_slope = (1/n) * np.sum((y_pred - y_train) * X_train)
            grad_intercept = (1/n) * np.sum(y_pred - y_train)
            
            # Calculate gradient magnitude
            grad_magnitude = np.sqrt(grad_slope**2 + grad_intercept**2)
            
            # Store values
            unstable_history['slope'].append(self.slope)
            unstable_history['intercept'].append(self.intercept)
            unstable_history['grad_slope'].append(grad_slope)
            unstable_history['grad_intercept'].append(grad_intercept)
            unstable_history['train_loss'].append(train_loss)
            unstable_history['grad_magnitude'].append(grad_magnitude)
            
            # Update parameters
            self.slope = self.slope - self.learning_rate * grad_slope
            self.intercept = self.intercept - self.learning_rate * grad_intercept
        
        # Restore original parameters
        self.learning_rate = original_lr
        self.slope = original_slope
        self.intercept = original_intercept
        
        return unstable_history