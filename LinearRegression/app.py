from model import LinearRegression
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("📚 Linear Regression Model")
st.markdown("Predict $y$ based $x$ using linear regression!")

# Sidebar configuration
st.sidebar.header("Configuration")
# TODO: Add all configuration sliders
learning_rate = 0.001 #value 0 < learning_rate < 1
n_iterations = 500

m = LinearRegression(learning_rate,n_iterations)
m.data_split() #optional parameters: test_size & random_state
data = m.get_data()

# Generate data
# TODO: Implement data generation

# Display data
st.header("1️⃣ Synthetic Data")
# TODO: Show statistics and scatter plot
st.dataframe(data)

# Train model
st.header("2️⃣ Model Training")
# TODO: Train and display results
st.write("Learning Rate:", learning_rate)
st.write("n_Iterations:", n_iterations)
m.fit()
metrics = m.calc_metrics()
history = m.get_history() #returns {'slope': [],'intercept': [],'grad_slope': [],'grad_intercept': []}

# Visualizations
st.header("3️⃣ Training Visualizations")
# TODO: Create plots

# I found that with a learning rate of 0.001 it has a good best fit line
# Anything bigger that it gets a negative trend line or no line at all.

x = np.array(data["x"]).flatten()
y = np.array(data["y"]).flatten()

fig = go.Figure()
fig.add_trace(go.Scatter(x=x,y=y,mode="markers",name="Synthetic Data"))

fig.add_trace(go.Scatter(x=x,y=m.predict(x),mode="lines",name="Best Fit",line=dict(color="red")))

fig.update_layout(title="Line of Best Fit", xaxis_title="X", yaxis_title="Y")

st.plotly_chart(fig, use_container_width=True)

# Interactive predictions
st.header("4️⃣ Make Predictions")
# TODO: Add prediction interface
pred = m.predict(35) #input any singular integer
st.write(pred)


#console printing tests
print("Metrics")
print(f"R^2: {metrics['R^2']}")
print(f"MSE: {metrics['MSE']}")
print(f"RMSE: {metrics['RMSE']}")
print(f"MAE: {metrics['MAE']}")

print("\n")

print("History")
for idx in range(len(history["slope"])):
    print("-----------------------------------------------")
    print(f"Slope (m): {history['slope'][idx]}")
    print(f"Intercept (b): {history['intercept'][idx]}")
    print(f"Gradient Slope: {history['grad_slope'][idx]}")
    print(f"Gradient Intercept: {history['grad_intercept'][idx]}")
    print("-----------------------------------------------")