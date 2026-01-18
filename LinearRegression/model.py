import pandas as pd
import numpy as np
import plotly.express as plx
import os

class LinearRegression:
    def __init__(self, learning_rate, n_iterations):
        #initialize learning rate and number of iterations
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        #import data
        cwd = os.getcwd()
        data_path = os.path.join(cwd,"LinearRegression\\data\\synthetic_data_Simple_Linear.csv")
        self.data = pd.read_csv(data_path)

    def test_init(self):
        print(self.data.head())
        return (self.learning_rate,self.n_iterations)