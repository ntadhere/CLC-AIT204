import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

'''
each function gets called by something in app.py
'''

class LinearRegression:
    def __init__(self, learning_rate, n_iterations):
        #init variables
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.slope = 0
        self.intercept = 0
        self.history = {
            'slope': [],
            'intercept': [],
            'grad_slope': [],
            'grad_intercept': []
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data["x"],self.data["y"],test_size=test_size,random_state=random_state)

    def fit(self):
        #implements gradient descent to train the model
        X = np.array(self.X_train).flatten()
        y = np.array(self.y_train).flatten()

        for iteration in range(1,self.n_iterations + 1):
            y_pred = self.slope * X + self.intercept
            n = len(X)
            grad_slope = (1/n) * sum((y_pred - y) * X)
            grad_intercept = (1/n) * sum(y_pred - y)

            self.slope = self.slope - self.learning_rate * grad_slope
            self.intercept = self.intercept - self.learning_rate * grad_intercept

            self.history["slope"].append(self.slope)
            self.history["intercept"].append(self.intercept)
            self.history["grad_slope"].append(grad_slope)
            self.history["grad_intercept"].append(grad_intercept)
        
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
        ss_total = np.sum((y-y.mean()) ** 2)
        r2 = 1 - (ss_res/ss_total)

        #MSE
        mse = np.mean((y-y_pred) ** 2)

        #RMSE
        rmse = np.sqrt(mse)

        #MAE
        mae = np.mean(np.abs(y-y_pred))

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