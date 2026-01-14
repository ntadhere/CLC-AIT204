import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Data Explorer")
st.markdown("Upload a CSV file and explore your data interactively!")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Show basic info
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Memory", f"{df.memory_usage().sum() / 1024:.2f} KB")

    # Display data
    st.subheader("Raw Data")
    st.dataframe(df)

    # Statistics
    st.subheader("Statistics")
    st.dataframe(df.describe())

    # Visualization
    st.subheader("Visualization")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis", numeric_cols)
        with col2:
            y_axis = st.selectbox("Y-axis", numeric_cols, index=1)

        fig = px.scatter(df, x=x_axis, y=y_axis)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👆 Upload a CSV file to get started!")