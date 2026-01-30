# Required Libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
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

unique_y = y.unique().tolist()
unq_y_size = len(unique_y)

print("Head X\n",X.head())
print("Head Y\n",y.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''
Problem is below. The X and y shapes differ and they must be the same or else we cannot train the model.
Current shape is
- X Tensor Shape:  (10275, 6)
- Y Tensor Shape:  (10275,)

X has 10275 rows, each with 6 data points ("net_rating","ts_pct","usg_pct","ast_pct","dreb_pct", and "oreb_pct")
Y has 10275 rows, each with 1 data points ("player_name")

Thinking we need to use reshape or maybe reformat the data somehow
'''

X_train = tf.constant(X_train)
X_test = tf.constant(X_test)
y_train = tf.constant(y_train)
y_test = tf.constant(y_test)

print("X Tensor Shape: ", X_train.shape)
print("Y Tensor Shape: ", y_train.shape)

# Design Neural Network Architecture
model = Sequential()
# TODO: Add Input layer
model.add(keras.layers.Input(shape=(X_train.shape[1],)))
# TODO: Add First Hidden Layer
model.add(keras.layers.Dense(512,activation="relu"))
model.add(keras.layers.Dropout(0.2))
# TODO: Add Second Hidden Layer
model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dropout(0.2))
# TODO: Add Output Layer
model.add(keras.layers.Dense(unq_y_size,activation="softmax"))

# Compile the Model
# TODO: Compile model specifying optimizer, loss and metrics
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])

# Train the Model
# TODO: Fit the model using training data
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test,y_test)
)

if model is not None:
    model.save(f"{cwd}\\ANN")
    print(f"Model saved to: {cwd}\\ANN")

# Evaluate the Model
# TODO: Predict on test data and evaluate the performance
results = model.evaluate(X_test,y_test)
print(f"\n\nTest Loss: {results[0]}")
print(f"Test Accuracy: {results[1]}")