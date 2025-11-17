
"""
Liver Disease Prediction - Streamlit Application
Main entry point for the multi-page Streamlit app
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Main page content
st.markdown('<h1 class="main-header">Liver Disease Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning Application for Medical Diagnosis Support</p>', unsafe_allow_html=True)

st.markdown("---")

# Introduction
st.header("Welcome to the Liver Disease Prediction Application")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("📊 **Exploratory Analysis**\n\nDiscover insights from the Indian Liver Patient Dataset")

with col2:
    st.success("🤖 **ML Models**\n\nCompare performance of multiple classification algorithms")

with col3:
    st.warning("🔮 **Make Predictions**\n\nPredict liver disease risk using trained models")

st.markdown("---")

# Project overview
st.header("Project Overview")

st.markdown("""
### About This Project

This application demonstrates an end-to-end machine learning solution for predicting liver disease
in patients using the Indian Liver Patient Dataset (ILPD).

#### Dataset Information
- **Source**: UCI Machine Learning Repository
- **Samples**: 584 patient records
- **Features**: 10 clinical and demographic variables
- **Target**: Binary classification (Liver Disease / No Disease)

#### Key Features
- Age, Gender
- Liver function tests: Bilirubin levels, Alkaline Phosphotase
- Liver enzymes: ALT, AST
- Protein markers: Total Proteins, Albumin, A/G Ratio

#### Machine Learning Pipeline
1. **Data Preprocessing**: Handling missing values, outliers, and duplicates
2. **Feature Engineering**: Creating meaningful derived features
3. **Model Training**: Multiple algorithms with hyperparameter tuning
4. **Evaluation**: Comprehensive metrics and visualizations
5. **Deployment**: Interactive prediction interface
""")

st.markdown("---")

# Navigation guide
st.header("Navigation Guide")

st.markdown("""
Use the sidebar to navigate through different sections:

1. **📊 Data Exploration** - Visualize and analyze the dataset
2. **🔧 Preprocessing** - View data cleaning and preparation steps
3. **🤖 Model Training** - Compare different ML algorithms
4. **📈 Model Evaluation** - Detailed performance analysis
5. **🔮 Prediction** - Make predictions on new data
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>Machine Learning and Data Analytics Coursework</p>
    <p>Westminster International University in Tashkent (WIUT)</p>
</div>
""", unsafe_allow_html=True)
