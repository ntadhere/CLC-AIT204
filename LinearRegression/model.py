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

    #want to rename to fit_grad_descent
    def fit(self):
        """
        Implements gradient descent to train the model
        Tracks both training and validation loss at each iteration
        Also tracks gradient magnitudes and parameter changes
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
            
            # Forward pass: make predictions
            y_pred_train = self.slope * X_train + self.intercept
            
            # Calculate loss on training set
            train_loss = self.calculate_mse(y_train, y_pred_train)
            
            # Calculate loss on validation set
            y_pred_val = self.slope * X_val + self.intercept
            val_loss = self.calculate_mse(y_val, y_pred_val)
            
            # Compute gradients (corrected - using n instead of iteration)
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

    def fit_batch(self):
        pass

    def fit_mini_batch(self):
        pass

    def fit_stochastic_grad(self):
        pass
    
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
