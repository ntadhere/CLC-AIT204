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
    st.dataframe(data, width='stretch')
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

st.plotly_chart(fig_split, width='stretch')

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

st.plotly_chart(fig_initial, width='stretch')

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
    m.fit()  # Will work with your current model.py

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

# Get error statistics
error_stats = m.get_error_statistics()

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Loss Curves", 
    "Error Analysis", 
    "Best Fit Line", 
    "Predicted vs Actual", 
    "Parameter Evolution",
    "Gradient Dynamics",
    "Advanced Analysis"
])

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
    
    st.plotly_chart(fig_loss, width='stretch')
    
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
    st.subheader("Comprehensive Error and Loss Analysis")
    
    # Error distribution comparison
    st.markdown("### 1. Error Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training Set Errors")
        
        # Histogram of training errors
        fig_train_err = go.Figure()
        fig_train_err.add_trace(go.Histogram(
            x=error_stats['train']['errors'],
            nbinsx=30,
            name='Training Errors',
            marker=dict(color='blue', opacity=0.7),
            showlegend=False
        ))
        
        fig_train_err.update_layout(
            title="Distribution of Training Errors",
            xaxis_title="Error (Actual - Predicted)",
            yaxis_title="Frequency",
            template='plotly_white',
            height=350
        )
        
        st.plotly_chart(fig_train_err, width='stretch')
        
        # Statistics
        train_errors = error_stats['train']['errors']
        st.write(f"**Mean Error:** {np.mean(train_errors):.4f}")
        st.write(f"**Median Error:** {np.median(train_errors):.4f}")
        st.write(f"**Std Dev:** {np.std(train_errors):.4f}")
        st.write(f"**Min Error:** {np.min(train_errors):.4f}")
        st.write(f"**Max Error:** {np.max(train_errors):.4f}")
        st.write(f"**Range:** {np.max(train_errors) - np.min(train_errors):.4f}")
    
    with col2:
        st.markdown("#### Validation Set Errors")
        
        # Histogram of validation errors
        fig_val_err = go.Figure()
        fig_val_err.add_trace(go.Histogram(
            x=error_stats['validation']['errors'],
            nbinsx=30,
            name='Validation Errors',
            marker=dict(color='green', opacity=0.7),
            showlegend=False
        ))
        
        fig_val_err.update_layout(
            title="Distribution of Validation Errors",
            xaxis_title="Error (Actual - Predicted)",
            yaxis_title="Frequency",
            template='plotly_white',
            height=350
        )
        
        st.plotly_chart(fig_val_err, width='stretch')
        
        # Statistics
        val_errors = error_stats['validation']['errors']
        st.write(f"**Mean Error:** {np.mean(val_errors):.4f}")
        st.write(f"**Median Error:** {np.median(val_errors):.4f}")
        st.write(f"**Std Dev:** {np.std(val_errors):.4f}")
        st.write(f"**Min Error:** {np.min(val_errors):.4f}")
        st.write(f"**Max Error:** {np.max(val_errors):.4f}")
        st.write(f"**Range:** {np.max(val_errors) - np.min(val_errors):.4f}")
    
    st.divider()
    
    # Combined error distribution
    st.markdown("### 2. Combined Error Distribution")
    
    fig_combined_err = go.Figure()
    
    fig_combined_err.add_trace(go.Histogram(
        x=error_stats['train']['errors'],
        name='Training Errors',
        marker=dict(color='blue', opacity=0.6),
        nbinsx=30
    ))
    
    fig_combined_err.add_trace(go.Histogram(
        x=error_stats['validation']['errors'],
        name='Validation Errors',
        marker=dict(color='green', opacity=0.6),
        nbinsx=30
    ))
    
    fig_combined_err.update_layout(
        title="Training vs Validation Error Distribution",
        xaxis_title="Error (Actual - Predicted)",
        yaxis_title="Frequency",
        barmode='overlay',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_combined_err, width='stretch')
    
    st.markdown("""
    **Interpretation:**
    - Errors centered around zero indicate unbiased predictions
    - Similar distributions suggest good generalization
    - Normal (bell-shaped) distribution validates linear regression assumptions
    """)
    
    st.divider()
    
    # Loss metrics comparison
    st.markdown("### 3. Loss Function Metrics Comparison")
    
    loss_comparison = pd.DataFrame({
        'Loss Function': ['MSE (Mean Squared Error)', 'MAE (Mean Absolute Error)', 'RMSE (Root Mean Squared Error)'],
        'Training Set': [
            error_stats['train']['mse'],
            error_stats['train']['mae'],
            error_stats['train']['rmse']
        ],
        'Validation Set': [
            error_stats['validation']['mse'],
            error_stats['validation']['mae'],
            error_stats['validation']['rmse']
        ],
        'Formula': [
            '(1/n) Σ(yᵢ - ŷᵢ)²',
            '(1/n) Σ|yᵢ - ŷᵢ|',
            '√[(1/n) Σ(yᵢ - ŷᵢ)²]'
        ]
    })
    
    loss_comparison['Absolute Difference'] = abs(loss_comparison['Validation Set'] - loss_comparison['Training Set'])
    loss_comparison['Relative Difference (%)'] = (
        loss_comparison['Absolute Difference'] / loss_comparison['Training Set'] * 100
    )
    
    st.dataframe(loss_comparison.style.format({
        'Training Set': '{:.6f}',
        'Validation Set': '{:.6f}',
        'Absolute Difference': '{:.6f}',
        'Relative Difference (%)': '{:.2f}%'
    }), width='stretch')
    
    st.markdown("""
    **Loss Function Characteristics:**
    
    | Metric | Sensitivity | Units | Use Case |
    |--------|-------------|-------|----------|
    | **MSE** | High (squares errors) | y² | Optimization (gradient descent) |
    | **MAE** | Medium (absolute values) | y | Robust to outliers |
    | **RMSE** | High (squares errors) | y | Interpretable in original units |
    """)
    
    st.divider()
    
    # Squared errors visualization
    st.markdown("### 4. Squared Errors Visualization")
    
    X_train_viz = np.array(m.X_train).flatten()
    X_val_viz = np.array(m.X_test).flatten()
    
    train_squared_errors = error_stats['train']['errors'] ** 2
    val_squared_errors = error_stats['validation']['errors'] ** 2
    
    fig_squared = go.Figure()
    
    fig_squared.add_trace(go.Scatter(
        x=X_train_viz,
        y=train_squared_errors,
        mode='markers',
        name='Training Squared Errors',
        marker=dict(color='blue', size=8, opacity=0.6)
    ))
    
    fig_squared.add_trace(go.Scatter(
        x=X_val_viz,
        y=val_squared_errors,
        mode='markers',
        name='Validation Squared Errors',
        marker=dict(color='green', size=8, opacity=0.6)
    ))
    
    # Add mean squared error lines
    fig_squared.add_hline(
        y=error_stats['train']['mse'],
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Train MSE: {error_stats['train']['mse']:.4f}",
        annotation_position="right"
    )
    
    fig_squared.add_hline(
        y=error_stats['validation']['mse'],
        line_dash="dash",
        line_color="green",
        annotation_text=f"Val MSE: {error_stats['validation']['mse']:.4f}",
        annotation_position="right"
    )
    
    fig_squared.update_layout(
        title="Squared Errors vs Input Values",
        xaxis_title="X (Input Values)",
        yaxis_title="Squared Error (yᵢ - ŷᵢ)²",
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_squared, width='stretch')
    
    st.markdown("""
    **Why Squared Errors Matter:**
    - MSE is the average of these squared errors
    - Larger errors are penalized more heavily (quadratic penalty)
    - MSE is differentiable, making it suitable for gradient descent
    - Outliers (points far from horizontal lines) contribute most to loss
    """)
    
    st.divider()
    
    # Error percentiles
    st.markdown("### 5. Error Percentile Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training Set Percentiles")
        train_percentiles = pd.DataFrame({
            'Percentile': ['Min (0%)', '25th', '50th (Median)', '75th', 'Max (100%)'],
            'Error Value': [
                np.percentile(train_errors, 0),
                np.percentile(train_errors, 25),
                np.percentile(train_errors, 50),
                np.percentile(train_errors, 75),
                np.percentile(train_errors, 100)
            ]
        })
        st.dataframe(train_percentiles.style.format({'Error Value': '{:.4f}'}), width='stretch')
    
    with col2:
        st.markdown("#### Validation Set Percentiles")
        val_percentiles = pd.DataFrame({
            'Percentile': ['Min (0%)', '25th', '50th (Median)', '75th', 'Max (100%)'],
            'Error Value': [
                np.percentile(val_errors, 0),
                np.percentile(val_errors, 25),
                np.percentile(val_errors, 50),
                np.percentile(val_errors, 75),
                np.percentile(val_errors, 100)
            ]
        })
        st.dataframe(val_percentiles.style.format({'Error Value': '{:.4f}'}), width='stretch')

with tab6:
    st.subheader("Gradient Descent Dynamics")
    
    iterations = list(range(1, n_iterations + 1))
    
    # 1. Gradient Evolution Over Time
    st.markdown("### 1. Gradient Values Over Iterations")
    
    fig_grad_evolution = go.Figure()
    
    # Add slope gradient
    fig_grad_evolution.add_trace(go.Scatter(
        x=iterations,
        y=history['grad_slope'],
        mode='lines',
        name='∂L/∂w (Slope Gradient)',
        line=dict(color='blue', width=2)
    ))
    
    # Add intercept gradient
    fig_grad_evolution.add_trace(go.Scatter(
        x=iterations,
        y=history['grad_intercept'],
        mode='lines',
        name='∂L/∂b (Intercept Gradient)',
        line=dict(color='red', width=2)
    ))
    
    # Add zero line
    fig_grad_evolution.add_hline(y=0, line_dash="dash", line_color="gray",
                                  annotation_text="Zero Gradient (Convergence)")
    
    fig_grad_evolution.update_layout(
        title="Gradient Components Over Training",
        xaxis_title="Iteration",
        yaxis_title="Gradient Value",
        template='plotly_white',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_grad_evolution, width='stretch')
    
    st.markdown("""
    **Interpretation:**
    - **Gradients approaching zero** = Model converging to optimal parameters
    - **Large initial gradients** = Parameters far from optimal
    - **Oscillations** = Learning rate may be too high
    - **Slow decay** = Learning rate may be too low
    """)
    
    st.divider()
    
    # 2. Gradient Magnitude
    st.markdown("### 2. Gradient Magnitude (L2 Norm)")
    
    fig_grad_mag = go.Figure()
    
    fig_grad_mag.add_trace(go.Scatter(
        x=iterations,
        y=history['grad_magnitude'],
        mode='lines',
        name='||∇L|| = √[(∂L/∂w)² + (∂L/∂b)²]',
        line=dict(color='purple', width=2),
        fill='tozeroy',
        fillcolor='rgba(128, 0, 128, 0.2)'
    ))
    
    fig_grad_mag.update_layout(
        title="Gradient Magnitude Over Training",
        xaxis_title="Iteration",
        yaxis_title="Gradient Magnitude ||∇L||",
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_grad_mag, width='stretch')
    
    # Gradient statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Initial Gradient Magnitude", f"{history['grad_magnitude'][0]:.6f}")
    with col2:
        st.metric("Final Gradient Magnitude", f"{history['grad_magnitude'][-1]:.6f}")
    with col3:
        reduction = history['grad_magnitude'][0] - history['grad_magnitude'][-1]
        st.metric("Reduction", f"{reduction:.6f}")
    with col4:
        pct_reduction = (reduction / history['grad_magnitude'][0]) * 100
        st.metric("% Reduction", f"{pct_reduction:.2f}%")
    
    st.markdown("""
    **Gradient Magnitude Interpretation:**
    - **||∇L||** measures the overall "steepness" at current position
    - **Decreasing magnitude** = Moving toward flatter region (minimum)
    - **Final magnitude near zero** = Convergence achieved
    - **High final magnitude** = Need more iterations or higher learning rate
    """)
    
    st.divider()
    
    # 3. Parameter Update Magnitudes
    st.markdown("### 3. Parameter Change Magnitude Per Iteration")
    
    fig_param_change = go.Figure()
    
    fig_param_change.add_trace(go.Scatter(
        x=iterations,
        y=history['param_change'],
        mode='lines',
        name='||Δθ|| = √[(Δw)² + (Δb)²]',
        line=dict(color='orange', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 165, 0, 0.2)'
    ))
    
    fig_param_change.update_layout(
        title="Parameter Update Magnitude Over Training",
        xaxis_title="Iteration",
        yaxis_title="Parameter Change Magnitude ||Δθ||",
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_param_change, width='stretch')
    
    st.markdown("""
    **Parameter Change Interpretation:**
    - **Δθ = α · ∇L** where α is learning rate
    - **Large changes early** = Taking big steps when far from optimum
    - **Small changes later** = Fine-tuning near optimum
    - **Relationship**: Δθ ∝ ||∇L|| × learning_rate
    """)
    
    st.divider()
    
    # 4. Gradient vs Loss Correlation
    st.markdown("### 4. Gradient-Loss Relationship")
    
    fig_grad_loss = go.Figure()
    
    # Create secondary y-axis for loss
    fig_grad_loss.add_trace(go.Scatter(
        x=iterations,
        y=history['grad_magnitude'],
        mode='lines',
        name='Gradient Magnitude',
        line=dict(color='purple', width=2),
        yaxis='y'
    ))
    
    fig_grad_loss.add_trace(go.Scatter(
        x=iterations,
        y=history['train_loss'],
        mode='lines',
        name='Training Loss',
        line=dict(color='blue', width=2),
        yaxis='y2'
    ))
    
    fig_grad_loss.update_layout(
        title="Gradient Magnitude vs Training Loss",
        xaxis_title="Iteration",
        yaxis=dict(
            title=dict(text="Gradient Magnitude", font=dict(color='purple')),
            tickfont=dict(color='purple')
        ),
        yaxis2=dict(
            title=dict(text="Training Loss (MSE)", font=dict(color='blue')),
            tickfont=dict(color='blue'),
            overlaying='y',
            side='right'
        ),
        template='plotly_white',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_grad_loss, width='stretch')
    
    st.markdown("""
    **Key Relationship:**
    - As **loss decreases**, **gradient magnitude decreases**
    - Both should approach zero together
    - Gradient tells us how much loss will change with parameter updates
    - At minimum: ∇L = 0 and L is minimized
    """)
    
    st.divider()
    
    # 5. Gradient Direction Analysis
    st.markdown("### 5. Gradient Direction (Vector Field)")
    
    # Create 2D vector plot showing gradient direction
    sample_iterations = [0, len(iterations)//4, len(iterations)//2, 
                        3*len(iterations)//4, len(iterations)-1]
    
    fig_grad_vector = go.Figure()
    
    for idx in sample_iterations:
        # Plot parameter position
        fig_grad_vector.add_trace(go.Scatter(
            x=[history['slope'][idx]],
            y=[history['intercept'][idx]],
            mode='markers',
            name=f'Iteration {idx+1}',
            marker=dict(size=10),
            showlegend=True
        ))
        
        # Add gradient vector (negative direction since we subtract)
        if idx < len(iterations) - 1:
            fig_grad_vector.add_annotation(
                x=history['slope'][idx],
                y=history['intercept'][idx],
                ax=history['slope'][idx] - learning_rate * history['grad_slope'][idx] * 100,
                ay=history['intercept'][idx] - learning_rate * history['grad_intercept'][idx] * 100,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red'
            )
    
    # Connect the path
    fig_grad_vector.add_trace(go.Scatter(
        x=history['slope'],
        y=history['intercept'],
        mode='lines',
        name='Optimization Path',
        line=dict(color='gray', width=1, dash='dot'),
        showlegend=True
    ))
    
    fig_grad_vector.update_layout(
        title="Parameter Space Trajectory",
        xaxis_title="w (Slope)",
        yaxis_title="b (Intercept)",
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig_grad_vector, width='stretch')
    
    st.markdown("""
    **Parameter Space Interpretation:**
    - Each point represents (w, b) values at that iteration
    - Red arrows show negative gradient direction (update direction)
    - Path shows optimization trajectory from start to end
    - Arrows should point toward the optimal parameters
    """)
    
    st.divider()
    
    # 6. Detailed Gradient Statistics
    st.markdown("### 6. Gradient Statistics Summary")
    
    grad_stats = pd.DataFrame({
        'Gradient Component': [
            '∂L/∂w (Slope)',
            '∂L/∂b (Intercept)',
            'Magnitude ||∇L||'
        ],
        'Initial Value': [
            history['grad_slope'][0],
            history['grad_intercept'][0],
            history['grad_magnitude'][0]
        ],
        'Final Value': [
            history['grad_slope'][-1],
            history['grad_intercept'][-1],
            history['grad_magnitude'][-1]
        ],
        'Max Absolute Value': [
            max(abs(min(history['grad_slope'])), abs(max(history['grad_slope']))),
            max(abs(min(history['grad_intercept'])), abs(max(history['grad_intercept']))),
            max(history['grad_magnitude'])
        ],
        'Mean Absolute Value': [
            np.mean(np.abs(history['grad_slope'])),
            np.mean(np.abs(history['grad_intercept'])),
            np.mean(history['grad_magnitude'])
        ]
    })
    
    st.dataframe(grad_stats.style.format({
        'Initial Value': '{:.6f}',
        'Final Value': '{:.6f}',
        'Max Absolute Value': '{:.6f}',
        'Mean Absolute Value': '{:.6f}'
    }), width='stretch')
    
    st.divider()
    
    # 7. Convergence Analysis
    st.markdown("### 7. Convergence Analysis")
    
    # Check convergence criteria
    convergence_threshold = 1e-6
    converged = history['grad_magnitude'][-1] < convergence_threshold
    
    # Find when gradient magnitude falls below threshold (if ever)
    convergence_iter = None
    for i, mag in enumerate(history['grad_magnitude']):
        if mag < convergence_threshold:
            convergence_iter = i + 1
            break
    
    col1, col2 = st.columns(2)
    
    with col1:
        if converged:
            st.success(f"✅ **Converged** (||∇L|| < {convergence_threshold})")
            if convergence_iter:
                st.write(f"Converged at iteration: **{convergence_iter}** / {n_iterations}")
                st.write(f"Efficiency: Used **{(convergence_iter/n_iterations)*100:.1f}%** of iterations")
        else:
            st.warning(f"⚠️ **Not Fully Converged** (||∇L|| = {history['grad_magnitude'][-1]:.6f})")
            st.write(f"Consider: More iterations or higher learning rate")
    
    with col2:
        # Estimate iterations needed
        if len(history['grad_magnitude']) > 10:
            # Simple exponential decay fit
            early_grad = np.mean(history['grad_magnitude'][:10])
            late_grad = np.mean(history['grad_magnitude'][-10:])
            decay_rate = late_grad / early_grad
            
            if decay_rate < 1 and decay_rate > 0:
                # Estimate iterations to reach threshold
                est_iters = int(np.log(convergence_threshold / early_grad) / np.log(decay_rate))
                st.info(f"📊 **Estimated iterations for convergence:** ~{est_iters}")
                
                if est_iters > n_iterations:
                    st.write(f"Suggestion: Increase iterations to **{est_iters}**")
    
    st.markdown("""
    **Convergence Criteria:**
    - **Gradient magnitude < 10⁻⁶**: Strong convergence
    - **Loss not changing**: Practical convergence
    - **Parameters stable**: Optimization complete
    """)

with tab7:
    st.subheader("Advanced Analysis: Gradient Verification & Large Gradient Effects")
    
    # Check if model has been trained
    if len(history['train_loss']) == 0:
        st.warning("⚠️ Please train the model first by visiting the other tabs.")
        st.stop()
    
    # Section 1: Gradient Verification
    st.markdown("## 1. Gradient Verification Using Finite Differences")
    
    st.markdown("""
    **Purpose:** Verify that our analytical gradient computation is correct by comparing 
    it with numerical approximation using finite differences.
    
    **Method:**
    - **Analytical Gradient:** Computed using calculus (∂L/∂θ)
    - **Numerical Gradient:** Approximated using finite differences: 
      [L(θ + ε) - L(θ - ε)] / (2ε)
    """)
    
    # Perform gradient verification
    epsilon_values = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    
    st.markdown("### Gradient Verification at Current Parameters")
    
    try:
        verification_results = []
        for eps in epsilon_values:
            result = m.verify_gradients(epsilon=eps)
            verification_results.append({
                'Epsilon (ε)': eps,
                'Analytical ∂L/∂w': result['analytical']['grad_w'],
                'Numerical ∂L/∂w': result['numerical']['grad_w'],
                'Absolute Diff (w)': result['absolute_difference']['grad_w'],
                'Relative Error (w)': result['relative_error']['grad_w'],
                'Analytical ∂L/∂b': result['analytical']['grad_b'],
                'Numerical ∂L/∂b': result['numerical']['grad_b'],
                'Absolute Diff (b)': result['absolute_difference']['grad_b'],
                'Relative Error (b)': result['relative_error']['grad_b']
            })
    except Exception as e:
        st.error(f"Error during gradient verification: {str(e)}")
        st.info("This may happen if the model hasn't been properly initialized. Please refresh and try again.")
        st.stop()
    
    verification_df = pd.DataFrame(verification_results)
    
    st.dataframe(verification_df.style.format({
        'Epsilon (ε)': '{:.0e}',
        'Analytical ∂L/∂w': '{:.8f}',
        'Numerical ∂L/∂w': '{:.8f}',
        'Absolute Diff (w)': '{:.2e}',
        'Relative Error (w)': '{:.2e}',
        'Analytical ∂L/∂b': '{:.8f}',
        'Numerical ∂L/∂b': '{:.8f}',
        'Absolute Diff (b)': '{:.2e}',
        'Relative Error (b)': '{:.2e}'
    }), use_container_width=True)
    
    # Best epsilon analysis
    best_eps_w = verification_df.loc[verification_df['Relative Error (w)'].idxmin(), 'Epsilon (ε)']
    best_eps_b = verification_df.loc[verification_df['Relative Error (b)'].idxmin(), 'Epsilon (ε)']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Best ε for ∂L/∂w", f"{best_eps_w:.0e}")
        min_error_w = verification_df['Relative Error (w)'].min()
        if min_error_w < 1e-5:
            st.success(f"✅ Excellent match (error: {min_error_w:.2e})")
        elif min_error_w < 1e-3:
            st.info(f"✓ Good match (error: {min_error_w:.2e})")
        else:
            st.warning(f"⚠️ Poor match (error: {min_error_w:.2e})")
    
    with col2:
        st.metric("Best ε for ∂L/∂b", f"{best_eps_b:.0e}")
        min_error_b = verification_df['Relative Error (b)'].min()
        if min_error_b < 1e-5:
            st.success(f"✅ Excellent match (error: {min_error_b:.2e})")
        elif min_error_b < 1e-3:
            st.info(f"✓ Good match (error: {min_error_b:.2e})")
        else:
            st.warning(f"⚠️ Poor match (error: {min_error_b:.2e})")
    
    # Visualization of verification
    fig_verification = go.Figure()
    
    fig_verification.add_trace(go.Scatter(
        x=[f"{eps:.0e}" for eps in epsilon_values],
        y=verification_df['Relative Error (w)'],
        mode='lines+markers',
        name='Relative Error (∂L/∂w)',
        line=dict(color='blue', width=2),
        marker=dict(size=10)
    ))
    
    fig_verification.add_trace(go.Scatter(
        x=[f"{eps:.0e}" for eps in epsilon_values],
        y=verification_df['Relative Error (b)'],
        mode='lines+markers',
        name='Relative Error (∂L/∂b)',
        line=dict(color='red', width=2),
        marker=dict(size=10)
    ))
    
    fig_verification.update_layout(
        title="Gradient Verification: Relative Error vs Epsilon",
        xaxis_title="Epsilon (ε)",
        yaxis_title="Relative Error",
        yaxis_type="log",
        template='plotly_white',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_verification, width='stretch')
    
    st.markdown("""
    **Interpretation:**
    - **Relative error < 10⁻⁵**: Gradients are correctly implemented
    - **Relative error 10⁻⁵ to 10⁻³**: Acceptable numerical precision
    - **Relative error > 10⁻³**: Potential implementation error
    
    **Epsilon selection:**
    - **Too large** (> 10⁻⁵): Finite difference approximation is inaccurate
    - **Too small** (< 10⁻⁹): Numerical precision errors dominate
    - **Optimal** (10⁻⁷ to 10⁻⁸): Best balance for double precision floats
    """)
    
    # Mathematical derivation
    with st.expander("📐 Mathematical Derivation of Gradients"):
        st.markdown("""
        ### Analytical Gradient Derivation
        
        Given loss function:
        ```
        L(w, b) = (1/n) Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
                = (1/n) Σᵢ₌₁ⁿ (yᵢ - (b + w·xᵢ))²
        ```
        
        **Gradient with respect to w:**
        ```
        ∂L/∂w = ∂/∂w [(1/n) Σᵢ (yᵢ - b - w·xᵢ)²]
              = (1/n) Σᵢ ∂/∂w [(yᵢ - b - w·xᵢ)²]
              = (1/n) Σᵢ 2(yᵢ - b - w·xᵢ) · (-xᵢ)
              = (1/n) Σᵢ -2(yᵢ - ŷᵢ) · xᵢ
              = (1/n) Σᵢ 2(ŷᵢ - yᵢ) · xᵢ
        ```
        (We absorb the factor of 2 into the learning rate)
        
        **Gradient with respect to b:**
        ```
        ∂L/∂b = ∂/∂b [(1/n) Σᵢ (yᵢ - b - w·xᵢ)²]
              = (1/n) Σᵢ 2(yᵢ - b - w·xᵢ) · (-1)
              = (1/n) Σᵢ 2(ŷᵢ - yᵢ)
        ```
        
        ### Numerical Gradient Approximation
        
        **Central Difference Method:**
        ```
        ∂L/∂w ≈ [L(w + ε, b) - L(w - ε, b)] / (2ε)
        ∂L/∂b ≈ [L(w, b + ε) - L(w, b - ε)] / (2ε)
        ```
        
        This is more accurate than forward difference: [L(w + ε) - L(w)] / ε
        
        **Error bound:** O(ε²) for central difference vs O(ε) for forward difference
        """)
    
    st.divider()
    
    # Section 2: Large Gradient Analysis
    st.markdown("## 2. Large Gradient Analysis & Gradient Explosion")
    
    st.markdown("""
    **Purpose:** Demonstrate what happens when gradients become too large, 
    typically caused by:
    - Learning rate too high
    - Poor parameter initialization
    - Ill-conditioned data
    - Numerical instability
    """)
    
    # User controls for demonstration
    col1, col2 = st.columns(2)
    with col1:
        extreme_lr = st.slider(
            "Extreme Learning Rate",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            help="Very high learning rate to demonstrate instability"
        )
    with col2:
        explosion_iters = st.slider(
            "Iterations to Run",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="Number of iterations before stopping"
        )
    
    if st.button("🚀 Run Large Gradient Demonstration", type="primary"):
        with st.spinner("Running unstable training demonstration..."):
            unstable_history = m.demonstrate_large_gradients(
                extreme_learning_rate=extreme_lr,
                iterations=explosion_iters
            )
        
        # Display results
        if unstable_history['exploded']:
            st.error(f"💥 **Gradient Explosion Occurred at Iteration {unstable_history['explosion_iteration']}!**")
            st.warning("Training became unstable and produced NaN/Inf values.")
        else:
            st.warning("⚠️ Training completed but likely diverged (loss increased)")
        
        # Compare stable vs unstable training
        st.markdown("### Unstable vs Stable Training Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Stable Training (Your Model)")
            st.write(f"**Learning Rate:** {learning_rate}")
            st.write(f"**Initial Loss:** {history['train_loss'][0]:.4f}")
            st.write(f"**Final Loss:** {history['train_loss'][-1]:.4f}")
            st.write(f"**Loss Reduction:** {(history['train_loss'][0] - history['train_loss'][-1]):.4f}")
            st.write(f"**Final Gradient Magnitude:** {history['grad_magnitude'][-1]:.6f}")
        
        with col2:
            st.markdown("#### Unstable Training (Large LR)")
            st.write(f"**Learning Rate:** {extreme_lr}")
            if len(unstable_history['train_loss']) > 0:
                st.write(f"**Initial Loss:** {unstable_history['train_loss'][0]:.4f}")
                st.write(f"**Final Loss:** {unstable_history['train_loss'][-1]:.4e}")
                loss_change = unstable_history['train_loss'][-1] - unstable_history['train_loss'][0]
                st.write(f"**Loss Change:** {loss_change:.4e}")
                st.write(f"**Final Gradient Magnitude:** {unstable_history['grad_magnitude'][-1]:.6e}")
        
        # Loss comparison plot
        st.markdown("### Loss Evolution Comparison")
        
        fig_loss_compare = go.Figure()
        
        # Stable training
        stable_iters = list(range(1, len(history['train_loss']) + 1))
        fig_loss_compare.add_trace(go.Scatter(
            x=stable_iters[:len(unstable_history['train_loss'])],
            y=history['train_loss'][:len(unstable_history['train_loss'])],
            mode='lines',
            name=f'Stable (LR={learning_rate})',
            line=dict(color='green', width=2)
        ))
        
        # Unstable training
        unstable_iters = list(range(1, len(unstable_history['train_loss']) + 1))
        fig_loss_compare.add_trace(go.Scatter(
            x=unstable_iters,
            y=unstable_history['train_loss'],
            mode='lines',
            name=f'Unstable (LR={extreme_lr})',
            line=dict(color='red', width=2)
        ))
        
        fig_loss_compare.update_layout(
            title="Loss Evolution: Stable vs Unstable Training",
            xaxis_title="Iteration",
            yaxis_title="Training Loss (MSE)",
            yaxis_type="log",
            template='plotly_white',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_loss_compare, width='stretch')
        
        # Gradient magnitude comparison
        st.markdown("### Gradient Magnitude Comparison")
        
        fig_grad_compare = go.Figure()
        
        # Stable gradients
        fig_grad_compare.add_trace(go.Scatter(
            x=stable_iters[:len(unstable_history['grad_magnitude'])],
            y=history['grad_magnitude'][:len(unstable_history['grad_magnitude'])],
            mode='lines',
            name=f'Stable (LR={learning_rate})',
            line=dict(color='green', width=2)
        ))
        
        # Unstable gradients
        fig_grad_compare.add_trace(go.Scatter(
            x=unstable_iters,
            y=unstable_history['grad_magnitude'],
            mode='lines',
            name=f'Unstable (LR={extreme_lr})',
            line=dict(color='red', width=2)
        ))
        
        fig_grad_compare.update_layout(
            title="Gradient Magnitude: Stable vs Unstable Training",
            xaxis_title="Iteration",
            yaxis_title="Gradient Magnitude ||∇L||",
            yaxis_type="log",
            template='plotly_white',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_grad_compare, width='stretch')
        
        # Parameter trajectory comparison
        st.markdown("### Parameter Space Trajectory Comparison")
        
        fig_traj_compare = go.Figure()
        
        # Stable trajectory
        fig_traj_compare.add_trace(go.Scatter(
            x=history['slope'][:len(unstable_history['slope'])],
            y=history['intercept'][:len(unstable_history['intercept'])],
            mode='lines+markers',
            name='Stable Path',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ))
        
        # Unstable trajectory
        fig_traj_compare.add_trace(go.Scatter(
            x=unstable_history['slope'],
            y=unstable_history['intercept'],
            mode='lines+markers',
            name='Unstable Path',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))
        
        # Mark starting points
        fig_traj_compare.add_trace(go.Scatter(
            x=[history['slope'][0]],
            y=[history['intercept'][0]],
            mode='markers',
            name='Start (Stable)',
            marker=dict(color='green', size=15, symbol='star')
        ))
        
        fig_traj_compare.add_trace(go.Scatter(
            x=[unstable_history['slope'][0]],
            y=[unstable_history['intercept'][0]],
            mode='markers',
            name='Start (Unstable)',
            marker=dict(color='red', size=15, symbol='star')
        ))
        
        fig_traj_compare.update_layout(
            title="Parameter Space: Stable vs Unstable Optimization Paths",
            xaxis_title="w (Slope)",
            yaxis_title="b (Intercept)",
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig_traj_compare, width='stretch')
        
        st.markdown("""
        **Observations:**
        - **Stable training** (green): Smooth convergence toward optimal parameters
        - **Unstable training** (red): Erratic behavior, divergence, or explosion
        - **Large learning rate** causes overshooting and oscillations
        - **Gradient explosion** occurs when updates become unbounded
        """)
    
    # Theoretical analysis
    st.divider()
    st.markdown("## 3. Mathematical Analysis of Large Gradients")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Geometric Interpretation
        
        **Gradient Descent Update:**
        ```
        θₙₑw = θₒₗd - α · ∇L(θₒₗd)
        ```
        
        **When α is too large:**
        - Step size > distance to minimum
        - **Overshooting** occurs
        - Can jump past the minimum
        - May diverge to infinity
        
        **Visualize as:**
        - Ball rolling down a hill
        - Too much momentum → flies past valley
        - Bounces back and forth with increasing amplitude
        """)
    
    with col2:
        st.markdown("""
        ### Numerical Stability
        
        **Gradient explosion causes:**
        1. **Parameter explosion**: |θ| → ∞
        2. **Loss explosion**: L → ∞
        3. **Numerical overflow**: Values exceed float limits
        4. **NaN propagation**: 0/0, ∞/∞, etc.
        
        **Prevention strategies:**
        1. **Learning rate decay**: Reduce α over time
        2. **Gradient clipping**: Cap ||∇L|| at threshold
        3. **Adaptive learning rates**: Adam, RMSprop
        4. **Batch normalization**: Stabilize activations
        5. **Proper initialization**: Start near optimum
        """)
    
    # Stability condition
    with st.expander("📊 Mathematical Stability Condition"):
        st.markdown("""
        ### Convergence Criterion for Gradient Descent
        
        For **convex** quadratic loss L(θ) = ½θᵀAθ - bᵀθ:
        
        **Stability condition:**
        ```
        0 < α < 2/λₘₐₓ(A)
        ```
        
        where λₘₐₓ is the largest eigenvalue of the Hessian matrix A.
        
        **For linear regression:**
        ```
        A = (1/n) XᵀX
        λₘₐₓ ≈ max(xᵢ²)
        ```
        
        **Practical guideline:**
        ```
        α < 1 / max(|xᵢ|²)
        ```
        
        **If α > 2/λₘₐₓ:**
        - Eigenvalue of update matrix > 1
        - Iterates grow exponentially
        - Gradient explosion occurs
        
        **Example:**
        - If max(xᵢ²) = 100
        - Then α should be < 0.01 for guaranteed convergence
        - Your learning rate of {learning_rate} is {('safe' if learning_rate < 0.01 else 'potentially unstable')}
        """)

# Rest of tabs remain unchanged...
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

    st.plotly_chart(fig_comparison, width='stretch')
    
    # Display parameter comparison
    st.markdown("### Parameter Comparison")
    
    comparison_df = pd.DataFrame({
        'Parameter': ['Slope (w)', 'Intercept (b)'],
        'Initial Value': [initial_params['initial_slope'], initial_params['initial_intercept']],
        'Final Value': [m.slope, m.intercept],
        'Change': [m.slope - initial_params['initial_slope'], 
                   m.intercept - initial_params['initial_intercept']]
    })
    
    st.dataframe(comparison_df, width='stretch')
    
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
        
        st.plotly_chart(fig_train, width='stretch')
        
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
        
        st.plotly_chart(fig_val, width='stretch')
        
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
    
    st.plotly_chart(fig_residuals, width='stretch')
    
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
    }), width='stretch')

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
    
    st.plotly_chart(fig_params, width='stretch')

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