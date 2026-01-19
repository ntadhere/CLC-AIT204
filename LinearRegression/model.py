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
            'val_loss': []         # Track validation loss
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

    def fit(self):
        """
        Implements gradient descent to train the model
        Tracks both training and validation loss at each iteration
        """
        X_train = np.array(self.X_train).flatten()
        y_train = np.array(self.y_train).flatten()
        X_val = np.array(self.X_test).flatten()
        y_val = np.array(self.y_test).flatten()
        
        n = len(X_train)  # number of training samples

        for iteration in range(1, self.n_iterations + 1):
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

            # Update parameters using gradient descent
            self.slope = self.slope - self.learning_rate * grad_slope
            self.intercept = self.intercept - self.learning_rate * grad_intercept

            # Track history
            self.history["slope"].append(self.slope)
            self.history["intercept"].append(self.intercept)
            self.history["grad_slope"].append(grad_slope)
            self.history["grad_intercept"].append(grad_intercept)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
        
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