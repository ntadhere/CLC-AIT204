"""
My First Streamlit App
A template to get you started with your own Streamlit project.

To run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="My First App",  # Browser tab title
    page_icon="🎯",              # Browser tab icon
    layout="wide",               # Use full width
    initial_sidebar_state="expanded"  # Sidebar open by default
)

# ===== MAIN APP =====

# Title and description
st.title("🎯 My First Streamlit App")
st.markdown("""
Welcome to your first Streamlit app! This is a template to help you get started.
Modify this file to build your own interactive web application.
""")

st.divider()

# ===== SIDEBAR =====
st.sidebar.header("⚙️ Settings")
st.sidebar.markdown("Add your controls here")

# Example sidebar widget
user_name = st.sidebar.text_input("Your name:", placeholder="Enter your name")

# ===== MAIN CONTENT =====

# Greet user if name is provided
if user_name:
    st.success(f"👋 Hello, {user_name}!")

# Example: Create some sample data
st.header("📊 Sample Data")

# Generate random data
@st.cache_data
def generate_sample_data(n_rows=100):
    """Generate sample dataset."""
    return pd.DataFrame({
        'Date': pd.date_range(start='2024-01-01', periods=n_rows, freq='D'),
        'Value A': np.random.randn(n_rows).cumsum(),
        'Value B': np.random.randn(n_rows).cumsum(),
        'Category': np.random.choice(['Cat 1', 'Cat 2', 'Cat 3'], n_rows)
    })

# Create data
df = generate_sample_data()

# Display data
tab1, tab2, tab3 = st.tabs(["📋 Data", "📈 Chart", "📊 Statistics"])

with tab1:
    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("Line Chart")
    st.line_chart(df.set_index('Date')[['Value A', 'Value B']])

with tab3:
    st.subheader("Statistics")
    st.dataframe(df.describe())

st.divider()

# ===== INTERACTIVE SECTION =====
st.header("🎮 Interactive Demo")

col1, col2 = st.columns(2)

with col1:
    number = st.slider("Select a number:", 0, 100, 50)
    st.metric("Your selection", number)

with col2:
    option = st.selectbox("Choose an option:", ["Option A", "Option B", "Option C"])
    st.write(f"You selected: {option}")

# Button example
if st.button("Click me! 🎉"):
    st.balloons()
    st.success("Button clicked!")

st.divider()

# ===== FOOTER =====
st.markdown("""
---
**💡 Next Steps:**
1. Modify this template to suit your needs
2. Add more widgets and visualizations
3. Deploy to Streamlit Cloud (see DEPLOYMENT_GUIDE.md)

**📚 Resources:**
- [Streamlit Documentation](https://docs.streamlit.io)
- [API Reference](https://docs.streamlit.io/library/api-reference)
- [Gallery](https://streamlit.io/gallery)
""")
