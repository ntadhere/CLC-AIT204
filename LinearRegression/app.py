from model import LinearRegression
import pandas as pd
import numpy as np
import plotly.express as plx

learning_rate = 0.01 #value 0 < learning_rate < 1
n_iterations = 100

m = LinearRegression(learning_rate,n_iterations)

m.data_split() #optional parameters: test_size & random_state

m.fit()

metrics = m.calc_metrics()

history = m.get_history() #returns {'slope': [],'intercept': [],'grad_slope': [],'grad_intercept': []}

print("Metrics")
print(f"R^2: {metrics["R^2"]}")
print(f"MSE: {metrics["MSE"]}")
print(f"RMSE: {metrics["RMSE"]}")
print(f"MAE: {metrics["MAE"]}")

print("\n")

print("History")
for idx in range(len(history["slope"])):
    print("-----------------------------------------------")
    print(f"Slope (m): {history["slope"][idx]}")
    print(f"Intercept (b): {history["intercept"][idx]}")
    print(f"Gradient Slope: {history["grad_slope"][idx]}")
    print(f"Gradient Intercept: {history["grad_intercept"][idx]}")
print("-----------------------------------------------")