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

# Visualize train/validation split
st.subheader("Train-Validation Split Visualization")

X_train_split = np.array(m.X_train).flatten()
y_train_split = np.array(m.y_train).flatten()
X_val_split = np.array(m.X_test).flatten()
y_val_split = np.array(m.y_test).flatten()

fig_split = go.Figure()

# Add training data
fig_split.add_trace(go.Scatter(
    x=X_train_split,
    y=y_train_split,
    mode='markers',
    name=f'Training Set (80%, n={len(X_train_split)})',
    marker=dict(color='blue', size=8, opacity=0.7, symbol='circle')
))

# Add validation data
fig_split.add_trace(go.Scatter(
    x=X_val_split,
    y=y_val_split,
    mode='markers',
    name=f'Validation Set (20%, n={len(X_val_split)})',
    marker=dict(color='red', size=8, opacity=0.7, symbol='x')
))

fig_split.update_layout(
    title="Data Split: Training vs Validation Sets",
    xaxis_title="X",
    yaxis_title="Y",
    template='plotly_white',
    hovermode='closest'
)

st.plotly_chart(fig_split, use_container_width=True)

st.markdown("""
**Split Details:**
- Training data (blue circles) is used to learn the model parameters
- Validation data (red X's) is held out to evaluate generalization performance
- Random split ensures both sets are representative of the overall distribution
""")

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
tab1, tab2, tab3, tab4 = st.tabs(["Loss Curves", "Best Fit Line", "Predicted vs Actual", "Parameter Evolution"])

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
    st.subheader("Predicted vs Actual Values Analysis")
    
    # Get training and validation data
    X_train = np.array(m.X_train).flatten()
    y_train = np.array(m.y_train).flatten()
    X_val = np.array(m.X_test).flatten()
    y_val = np.array(m.y_test).flatten()
    
    # Generate predictions
    y_train_pred = m.predict(X_train)
    y_val_pred = m.predict(X_val)
    
    # Create two columns for train and validation plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training Set")
        
        # Scatter plot: Predicted vs Actual
        fig_train = go.Figure()
        
        # Add scatter points
        fig_train.add_trace(go.Scatter(
            x=y_train,
            y=y_train_pred,
            mode='markers',
            name='Training Data',
            marker=dict(color='blue', size=8, opacity=0.6),
            text=[f'Actual: {y_train[i]:.2f}<br>Predicted: {y_train_pred[i]:.2f}<br>Error: {y_train[i]-y_train_pred[i]:.2f}' 
                  for i in range(len(y_train))],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add perfect prediction line (y = x)
        min_val = min(y_train.min(), y_train_pred.min())
        max_val = max(y_train.max(), y_train_pred.max())
        fig_train.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_train.update_layout(
            title="Training Set: Predicted vs Actual",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            template='plotly_white',
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig_train, use_container_width=True)
        
        # Training set statistics
        train_residuals = y_train - y_train_pred
        st.write(f"**Training MSE:** {np.mean(train_residuals**2):.4f}")
        st.write(f"**Training MAE:** {np.mean(np.abs(train_residuals)):.4f}")
        st.write(f"**Mean Residual:** {np.mean(train_residuals):.4f}")
        st.write(f"**Std Residual:** {np.std(train_residuals):.4f}")
    
    with col2:
        st.markdown("#### Validation Set")
        
        # Scatter plot: Predicted vs Actual
        fig_val = go.Figure()
        
        # Add scatter points
        fig_val.add_trace(go.Scatter(
            x=y_val,
            y=y_val_pred,
            mode='markers',
            name='Validation Data',
            marker=dict(color='green', size=8, opacity=0.6),
            text=[f'Actual: {y_val[i]:.2f}<br>Predicted: {y_val_pred[i]:.2f}<br>Error: {y_val[i]-y_val_pred[i]:.2f}' 
                  for i in range(len(y_val))],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add perfect prediction line (y = x)
        min_val = min(y_val.min(), y_val_pred.min())
        max_val = max(y_val.max(), y_val_pred.max())
        fig_val.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_val.update_layout(
            title="Validation Set: Predicted vs Actual",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            template='plotly_white',
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig_val, use_container_width=True)
        
        # Validation set statistics
        val_residuals = y_val - y_val_pred
        st.write(f"**Validation MSE:** {np.mean(val_residuals**2):.4f}")
        st.write(f"**Validation MAE:** {np.mean(np.abs(val_residuals)):.4f}")
        st.write(f"**Mean Residual:** {np.mean(val_residuals):.4f}")
        st.write(f"**Std Residual:** {np.std(val_residuals):.4f}")
    
    st.divider()
    
    # Combined residual plot
    st.markdown("#### Residual Analysis")
    
    fig_residuals = go.Figure()
    
    # Training residuals
    fig_residuals.add_trace(go.Scatter(
        x=X_train,
        y=train_residuals,
        mode='markers',
        name='Training Residuals',
        marker=dict(color='blue', size=6, opacity=0.6)
    ))
    
    # Validation residuals
    fig_residuals.add_trace(go.Scatter(
        x=X_val,
        y=val_residuals,
        mode='markers',
        name='Validation Residuals',
        marker=dict(color='green', size=6, opacity=0.6)
    ))
    
    # Zero line
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", 
                            annotation_text="Zero Error Line")
    
    fig_residuals.update_layout(
        title="Residual Plot: Errors vs Input Values",
        xaxis_title="X (Input Values)",
        yaxis_title="Residuals (Actual - Predicted)",
        template='plotly_white',
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig_residuals, use_container_width=True)
    
    st.markdown("""
    **Interpretation Guide:**
    - **Points on red diagonal line** = Perfect predictions (actual = predicted)
    - **Points above the line** = Model underpredicts (actual > predicted)
    - **Points below the line** = Model overpredicts (actual < predicted)
    - **Residuals near zero** = Good predictions
    - **Random scatter in residuals** = Good model fit (no pattern means no systematic bias)
    - **Pattern in residuals** = Model is missing something (non-linear relationship, etc.)
    """)
    
    # Statistical comparison
    st.markdown("#### Train vs Validation Comparison")
    comparison_metrics = pd.DataFrame({
        'Metric': ['MSE', 'MAE', 'Mean Residual', 'Std Residual'],
        'Training Set': [
            np.mean(train_residuals**2),
            np.mean(np.abs(train_residuals)),
            np.mean(train_residuals),
            np.std(train_residuals)
        ],
        'Validation Set': [
            np.mean(val_residuals**2),
            np.mean(np.abs(val_residuals)),
            np.mean(val_residuals),
            np.std(val_residuals)
        ]
    })
    comparison_metrics['Difference'] = comparison_metrics['Validation Set'] - comparison_metrics['Training Set']
    
    st.dataframe(comparison_metrics.style.format({
        'Training Set': '{:.4f}',
        'Validation Set': '{:.4f}',
        'Difference': '{:.4f}'
    }), use_container_width=True)

with tab4:
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