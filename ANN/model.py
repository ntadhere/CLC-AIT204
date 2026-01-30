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

optimal_stats = ["net_rating","ts_pct","usg_pct","ast_pct","dreb_pct","oreb_pct"]

optimal_data = pd.DataFrame(data=data[optimal_stats])

score_weights = {
    "net_rating": 0.5,
    "ts_pct": 0.25,
    "usg_pct": -0.05,
    "ast_pct": 0.15,
    "dreb_pct": 0.1,
    "oreb_pct": 0.05
}

optimal_data["score"] = sum(optimal_data[col] * w for col, w in score_weights.items())

print("Head\n",optimal_data.head())
print("Describe\n",optimal_data.describe())
print("Shape\n",optimal_data.shape)
print("Info\n",optimal_data.info())
print("Null Values", optimal_data.columns[optimal_data.isnull().any()].tolist()) #checks for null values

X, y = optimal_data[optimal_stats[0:len(optimal_stats)-1]], optimal_data["score"]

print("Head X\n",X.head())
print("Head Y\n",y.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = tf.constant(X_train)
X_test = tf.constant(X_test)
y_train = tf.constant(y_train)
y_test = tf.constant(y_test)

print("X Tensor Shape: ", X_train.shape)
print("- X_train.shape[1]: ", X_train.shape[1])
print("Y Tensor Shape: ", y_train.shape)

# Design Neural Network Architecture
model = Sequential()
# TODO: Add Input layer
model.add(keras.layers.Input(shape=(X_train.shape[1],)))
#Second Hidden Layer
model.add(keras.layers.Dense(64,activation="relu"))
#Third Hidden Layer
model.add(keras.layers.Dense(32,activation="relu"))
#Fourth Hidden Layer
model.add(keras.layers.Dense(16,activation="relu"))
#Fifth Hidden Layer
model.add(keras.layers.Dense(8,activation="relu"))
# TODO: Add Output Layer
model.add(keras.layers.Dense(1))

# Compile the Model
# TODO: Compile model specifying optimizer, loss and metrics
model.compile(optimizer="adam",loss="mse",metrics=['mae'])

# Train the Model
# TODO: Fit the model using training data
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test,y_test)
)

if model is not None:
    model.save(f"{cwd}\\ANN\\le_model.h5")
    print(f"Model saved to: {cwd}\\ANN")

# Evaluate the Model
# TODO: Predict on test data and evaluate the performance
results = model.evaluate(X_test,y_test)
print(f"\n\nTest Loss: {results[0]}")
print(f"Test MAE: {results[1]}")