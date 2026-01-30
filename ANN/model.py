# Required Libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense
import keras

# Load Dataset
cwd = os.getcwd()
data = pd.read_csv(f"{cwd}\\ANN\\data\\all_seasons.csv")

optimal_stats = ["net_rating","ts_pct","usg_pct","ast_pct","dreb_pct","oreb_pct","player_name"]

optimal_data_points = pd.DataFrame(data=data[optimal_stats])

print("Head\n",optimal_data_points.head())
print("Describe\n",optimal_data_points.describe())
print("Shape\n",optimal_data_points.shape)
print("Info\n",optimal_data_points.info())
print("Null Values", optimal_data_points.columns[optimal_data_points.isnull().any()].tolist()) #checks for null values

X, y = optimal_data_points[optimal_stats[0:len(optimal_stats)-1]], optimal_data_points["player_name"]

print("Head X\n",X.head())
print("Head Y\n",y.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Design Neural Network Architecture
model = Sequential()
# TODO: Add Input layer
model.add(keras.layers.Input(shape=(1,)))
# TODO: Add First Hidden Layer
# TODO: Add Second Hidden Layer
# TODO: Add Output Layer

# Compile the Model
# TODO: Compile model specifying optimizer, loss and metrics

# Train the Model
# TODO: Fit the model using training data

# Evaluate the Model
# TODO: Predict on test data and evaluate the performance