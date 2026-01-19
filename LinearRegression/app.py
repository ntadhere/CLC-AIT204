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
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, 0.001, 0.0001, 
                                   help="Controls step size in gradient descent")
n_iterations = st.sidebar.slider("Number of Iterations", 100, 2000, 500, 100,
                                  help="Number of training iterations")
random_seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, 
                                       value=42, step=1,
                                       help="Seed for reproducible random initialization")

m = LinearRegression(learning_rate, n_iterations, random_seed=random_seed)
m.data_split()  # optional parameters: test_size & random_state
data = m.get_data()

# Get initial parameters before training
initial_params = m.get_initial_params()

# Display data
st.header("1️⃣ Synthetic Data")

col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(data, use_container_width=True)
with col2:
    st.markdown("### Dataset Statistics")
    st.write(f"**Number of samples:** {len(data)}")
    st.write(f"**X range:** [{data['x'].min():.2f}, {data['x'].max():.2f}]")
    st.write(f"**Y range:** [{data['y'].min():.2f}, {data['y'].max():.2f}]")
    st.write(f"**X mean:** {data['x'].mean():.2f}")
    st.write(f"**Y mean:** {data['y'].mean():.2f}")

# Train model
st.header("2️⃣ Model Training")

# Show initial parameters BEFORE training
st.subheader("Initial Parameters (Before Training)")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Initial Slope (w₀)", f"{initial_params['initial_slope']:.6f}")
with col2:
    st.metric("Initial Intercept (b₀)", f"{initial_params['initial_intercept']:.6f}")
with col3:
    st.info(f"Random Seed: {random_seed}")

st.markdown(f"**Initial Equation:** y = {initial_params['initial_intercept']:.4f} + {initial_params['initial_slope']:.4f}x")

# Visualize initial (pre-training) regression line
st.subheader("Pre-Training Visualization")
x = np.array(data["x"]).flatten()
y = np.array(data["y"]).flatten()

fig_initial = go.Figure()

# Add data points
fig_initial.add_trace(go.Scatter(
    x=x,
    y=y,
    mode="markers",
    name="Synthetic Data",
    marker=dict(color='lightblue', size=8, opacity=0.6)
))

# Add initial regression line (before training)
y_initial = m.predict_with_params(x, initial_params['initial_slope'], initial_params['initial_intercept'])
fig_initial.add_trace(go.Scatter(
    x=x,
    y=y_initial,
    mode="lines",
    name="Initial Line (Pre-Training)",
    line=dict(color="orange", width=3, dash='dash')
))

fig_initial.update_layout(
    title="Initial Regression Line Before Training",
    xaxis_title="X",
    yaxis_title="Y",
    template='plotly_white',
    hovermode='closest'
)

st.plotly_chart(fig_initial, use_container_width=True)

st.markdown("""
**Note:** The orange dashed line shows the initial random regression line before any training. 
This demonstrates that the model starts with random parameters and learns from the data through gradient descent.
""")

st.divider()

# Training configuration
st.subheader("Training Configuration")
st.write("Learning Rate:", learning_rate)
st.write("Number of Iterations:", n_iterations)

with st.spinner("Training model..."):
    m.fit()

metrics = m.calc_metrics()
history = m.get_history()

# Display final loss values
final_loss = m.get_final_loss()
col1, col2 = st.columns(2)
with col1:
    st.metric("Final Training Loss (MSE)", f"{final_loss['final_train_loss']:.4f}")
with col2:
    st.metric("Final Validation Loss (MSE)", f"{final_loss['final_val_loss']:.4f}")

# Visualizations
st.header("3️⃣ Training Visualizations")

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Loss Curves", "Best Fit Line", "Parameter Evolution"])

with tab1:
    st.subheader("Training and Validation Loss Over Iterations")
    
    # Create loss curve plot
    iterations = list(range(1, n_iterations + 1))
    
    fig_loss = go.Figure()
    
    # Add training loss
    fig_loss.add_trace(go.Scatter(
        x=iterations,
        y=history['train_loss'],
        mode='lines',
        name='Training Loss',
        line=dict(color='blue', width=2)
    ))
    
    # Add validation loss
    fig_loss.add_trace(go.Scatter(
        x=iterations,
        y=history['val_loss'],
        mode='lines',
        name='Validation Loss',
        line=dict(color='red', width=2)
    ))
    
    fig_loss.update_layout(
        title="Loss (MSE) vs Iterations",
        xaxis_title="Iteration",
        yaxis_title="Mean Squared Error (MSE)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig_loss, use_container_width=True)
    
    # Analysis text
    st.markdown("### Loss Curve Analysis")
    
    # Calculate convergence metrics
    train_loss_diff = history['train_loss'][0] - history['train_loss'][-1]
    val_loss_diff = history['val_loss'][0] - history['val_loss'][-1]
    
    st.write(f"""
    **Initial Training Loss:** {history['train_loss'][0]:.4f}  
    **Final Training Loss:** {history['train_loss'][-1]:.4f}  
    **Loss Reduction:** {train_loss_diff:.4f} ({(train_loss_diff/history['train_loss'][0]*100):.2f}%)
    
    **Initial Validation Loss:** {history['val_loss'][0]:.4f}  
    **Final Validation Loss:** {history['val_loss'][-1]:.4f}  
    **Loss Reduction:** {val_loss_diff:.4f} ({(val_loss_diff/history['val_loss'][0]*100):.2f}%)
    """)
    
    # Check for overfitting
    if final_loss['final_val_loss'] > final_loss['final_train_loss'] * 1.2:
        st.warning("⚠️ Potential overfitting detected: Validation loss is significantly higher than training loss.")
    elif abs(final_loss['final_val_loss'] - final_loss['final_train_loss']) < 0.01:
        st.success("✅ Good generalization: Training and validation losses are similar.")

with tab2:
    st.subheader("Before vs After Training Comparison")
    
    fig_comparison = go.Figure()
    
    # Add data points
    fig_comparison.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="Synthetic Data",
        marker=dict(color='lightblue', size=8, opacity=0.6)
    ))
    
    # Add initial regression line (before training)
    y_initial = m.predict_with_params(x, initial_params['initial_slope'], initial_params['initial_intercept'])
    fig_comparison.add_trace(go.Scatter(
        x=x,
        y=y_initial,
        mode="lines",
        name="Initial Line (Before Training)",
        line=dict(color="orange", width=2, dash='dash')
    ))

    # Add final regression line (after training)
    fig_comparison.add_trace(go.Scatter(
        x=x,
        y=m.predict(x),
        mode="lines",
        name="Final Line (After Training)",
        line=dict(color="red", width=3)
    ))

    fig_comparison.update_layout(
        title="Regression Line: Before vs After Training",
        xaxis_title="X",
        yaxis_title="Y",
        template='plotly_white'
    )

    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Display parameter comparison
    st.markdown("### Parameter Comparison")
    
    comparison_df = pd.DataFrame({
        'Parameter': ['Slope (w)', 'Intercept (b)'],
        'Initial Value': [initial_params['initial_slope'], initial_params['initial_intercept']],
        'Final Value': [m.slope, m.intercept],
        'Change': [m.slope - initial_params['initial_slope'], 
                   m.intercept - initial_params['initial_intercept']]
    })
    
    st.dataframe(comparison_df, use_container_width=True)
    
    st.write(f"**Initial Equation:** y = {initial_params['initial_intercept']:.4f} + {initial_params['initial_slope']:.4f}x")
    st.write(f"**Final Equation:** y = {m.intercept:.4f} + {m.slope:.4f}x")

with tab3:
    st.subheader("Parameter Evolution During Training")
    
    fig_params = go.Figure()
    
    # Plot slope evolution
    fig_params.add_trace(go.Scatter(
        x=iterations,
        y=history['slope'],
        mode='lines',
        name='Slope (w)',
        line=dict(color='green', width=2)
    ))
    
    # Plot intercept evolution
    fig_params.add_trace(go.Scatter(
        x=iterations,
        y=history['intercept'],
        mode='lines',
        name='Intercept (b)',
        line=dict(color='purple', width=2)
    ))
    
    fig_params.update_layout(
        title="Parameter Values Over Iterations",
        xaxis_title="Iteration",
        yaxis_title="Parameter Value",
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig_params, use_container_width=True)

# Model Evaluation
st.header("4️⃣ Model Evaluation Metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("R² Score", f"{metrics['R^2']:.4f}")
with col2:
    st.metric("MSE", f"{metrics['MSE']:.4f}")
with col3:
    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
with col4:
    st.metric("MAE", f"{metrics['MAE']:.4f}")

# Interactive predictions
st.header("5️⃣ Make Predictions")
x_input = st.number_input("Enter a value for x:", value=35.0, step=1.0)
pred = m.predict(x_input)
st.success(f"Predicted y value: **{pred[0]:.4f}**")

# Console printing tests
st.divider()
with st.expander("📊 View Detailed Training History"):
    st.write("### Initial vs Final Parameters")
    st.write(f"**Initial Slope (w₀):** {initial_params['initial_slope']:.6f}")
    st.write(f"**Final Slope (w):** {history['slope'][-1]:.6f}")
    st.write(f"**Slope Change:** {history['slope'][-1] - initial_params['initial_slope']:.6f}")
    st.write("")
    st.write(f"**Initial Intercept (b₀):** {initial_params['initial_intercept']:.6f}")
    st.write(f"**Final Intercept (b):** {history['intercept'][-1]:.6f}")
    st.write(f"**Intercept Change:** {history['intercept'][-1] - initial_params['initial_intercept']:.6f}")
    st.write("")
    st.write(f"**Final Gradient (Slope):** {history['grad_slope'][-1]:.6f}")
    st.write(f"**Final Gradient (Intercept):** {history['grad_intercept'][-1]:.6f}")