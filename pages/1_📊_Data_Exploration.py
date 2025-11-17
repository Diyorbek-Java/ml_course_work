"""
Data Exploration Page
Interactive exploratory data analysis and visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")

# Title
st.title("📊 Data Exploration")
st.markdown("Explore and visualize the Indian Liver Patient Dataset")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    column_names = [
        'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
        'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
        'Aspartate_Aminotransferase', 'Total_Proteins', 'Albumin',
        'Albumin_Globulin_Ratio', 'Target'
    ]
    df = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv', names=column_names)
    return df

try:
    df = load_data()

    # Dataset Overview
    st.header("1. Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1] - 1)
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Duplicates", df.duplicated().sum())

    # Display data
    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

    # Data types and info
    with st.expander("📋 Dataset Information"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Data Type': df.dtypes.values
            })
            st.dataframe(dtype_df, use_container_width=True)

        with col2:
            st.subheader("Missing Values")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)

    # Statistical Summary
    st.header("2. Statistical Summary")
    st.dataframe(df.describe().T, use_container_width=True)

    # Target Distribution
    st.header("3. Target Variable Distribution")

    col1, col2 = st.columns(2)

    with col1:
        target_counts = df['Target'].value_counts()
        fig = px.bar(
            x=['Liver Patient (1)', 'Non-Liver Patient (2)'],
            y=target_counts.values,
            labels={'x': 'Class', 'y': 'Count'},
            title='Target Distribution',
            color=target_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            values=target_counts.values,
            names=['Liver Patient (1)', 'Non-Liver Patient (2)'],
            title='Target Distribution (%)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature Distributions
    st.header("4. Feature Distributions")

    numerical_cols = df.select_dtypes(include=[np.number]).columns.drop('Target')

    selected_feature = st.selectbox("Select a feature to visualize:", numerical_cols)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x=selected_feature,
            nbins=30,
            title=f'Distribution of {selected_feature}',
            labels={selected_feature: selected_feature, 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df, y=selected_feature,
            title=f'Box Plot of {selected_feature}',
            labels={selected_feature: selected_feature}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature Distribution by Target
    st.subheader("Feature Distribution by Target Class")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x=selected_feature, color='Target',
            nbins=30,
            title=f'{selected_feature} by Target Class',
            barmode='overlay',
            labels={selected_feature: selected_feature}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df, x='Target', y=selected_feature,
            title=f'{selected_feature} by Target Class',
            labels={'Target': 'Target Class', selected_feature: selected_feature}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Analysis
    st.header("5. Correlation Analysis")

    # Encode gender for correlation
    df_temp = df.copy()
    df_temp['Gender'] = df_temp['Gender'].map({'Male': 1, 'Female': 0})

    correlation_matrix = df_temp.corr()

    # Heatmap
    fig = px.imshow(
        correlation_matrix,
        text_auto='.2f',
        aspect='auto',
        title='Correlation Heatmap',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    st.plotly_chart(fig, use_container_width=True)

    # Correlation with target
    st.subheader("Correlation with Target Variable")
    target_corr = correlation_matrix['Target'].drop('Target').abs().sort_values(ascending=True)

    fig = px.bar(
        x=target_corr.values,
        y=target_corr.index,
        orientation='h',
        title='Absolute Correlation with Target',
        labels={'x': 'Absolute Correlation', 'y': 'Features'},
        color=target_corr.values,
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Gender Analysis
    st.header("6. Categorical Features Analysis")

    col1, col2 = st.columns(2)

    with col1:
        gender_counts = df['Gender'].value_counts()
        fig = px.bar(
            x=gender_counts.index,
            y=gender_counts.values,
            title='Gender Distribution',
            labels={'x': 'Gender', 'y': 'Count'},
            color=gender_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_target = pd.crosstab(df['Gender'], df['Target'])
        fig = px.bar(
            gender_target,
            barmode='group',
            title='Gender vs Target Distribution',
            labels={'value': 'Count', 'variable': 'Target Class'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Age Analysis
    st.header("7. Age Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x='Age', color='Target',
            nbins=20,
            title='Age Distribution by Target Class',
            barmode='overlay',
            labels={'Age': 'Age (years)'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        age_stats = df.groupby('Target')['Age'].describe()
        st.subheader("Age Statistics by Target")
        st.dataframe(age_stats, use_container_width=True)

    # Scatter Plots
    st.header("8. Feature Relationships")

    col1, col2 = st.columns(2)

    with col1:
        x_feature = st.selectbox("Select X-axis feature:", numerical_cols, index=0, key='x_scatter')

    with col2:
        y_feature = st.selectbox("Select Y-axis feature:", numerical_cols, index=1, key='y_scatter')

    fig = px.scatter(
        df, x=x_feature, y=y_feature, color='Target',
        title=f'{x_feature} vs {y_feature}',
        labels={'Target': 'Target Class'},
        opacity=0.6
    )
    st.plotly_chart(fig, use_container_width=True)

    # Download data
    st.header("9. Download Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Dataset as CSV",
        data=csv,
        file_name='liver_patient_dataset.csv',
        mime='text/csv',
    )

except FileNotFoundError:
    st.error("Dataset file not found! Please ensure 'Indian Liver Patient Dataset (ILPD).csv' is in the project directory.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
