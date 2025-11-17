"""
Data Preprocessing Page
Visualize and explain data preprocessing steps
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Data Preprocessing", page_icon="🔧", layout="wide")

st.title("🔧 Data Preprocessing")
st.markdown("Explore the data cleaning and preparation pipeline")
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

    # 1. Missing Values
    st.header("1. Handling Missing Values")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Before Imputation")
        missing_before = df.isnull().sum()
        missing_df_before = pd.DataFrame({
            'Column': missing_before.index,
            'Missing Count': missing_before.values,
            'Percentage': (missing_before.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df_before[missing_df_before['Missing Count'] > 0], use_container_width=True)

    df_processed = df.copy()
    if df_processed['Albumin_Globulin_Ratio'].isnull().sum() > 0:
        median_value = df_processed['Albumin_Globulin_Ratio'].median()
        df_processed['Albumin_Globulin_Ratio'].fillna(median_value, inplace=True)

    with col2:
        st.subheader("After Imputation")
        missing_after = df_processed.isnull().sum()
        missing_df_after = pd.DataFrame({
            'Column': missing_after.index,
            'Missing Count': missing_after.values,
            'Percentage': (missing_after.values / len(df_processed) * 100).round(2)
        })
        st.dataframe(missing_df_after, use_container_width=True)

    st.success("**Strategy**: Filled missing A/G Ratio values with median (robust to outliers)")

    # 2. Duplicates
    st.header("2. Handling Duplicates")

    before_dup = len(df_processed)
    df_processed = df_processed.drop_duplicates()
    after_dup = len(df_processed)
    removed_dup = before_dup - after_dup

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Samples", before_dup)
    with col2:
        st.metric("Duplicates Removed", removed_dup)
    with col3:
        st.metric("Final Samples", after_dup)

    st.info("**Strategy**: Removed duplicate records to prevent overfitting")

    # 3. Outlier Detection
    st.header("3. Outlier Detection & Handling")

    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.drop('Target')
    selected_col = st.selectbox("Select feature to analyze outliers:", numerical_cols)

    def detect_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound

    outliers, lower, upper = detect_outliers_iqr(df_processed, selected_col)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Outliers Detected", len(outliers))
    with col2:
        st.metric("Lower Bound", f"{lower:.2f}")
    with col3:
        st.metric("Upper Bound", f"{upper:.2f}")

    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Box(y=df_processed[selected_col], name='With Outliers'))

    # Cap outliers
    df_capped = df_processed.copy()
    df_capped[selected_col] = df_capped[selected_col].clip(lower=lower, upper=upper)
    fig.add_trace(go.Box(y=df_capped[selected_col], name='After Capping'))

    fig.update_layout(title=f"Outlier Treatment - {selected_col}", yaxis_title=selected_col)
    st.plotly_chart(fig, use_container_width=True)

    st.success("**Strategy**: Capped outliers at IQR boundaries instead of removing to preserve sample size")

    # 4. Feature Engineering
    st.header("4. Feature Engineering")

    st.subheader("Created Features")

    # Create engineered features
    df_engineered = df_processed.copy()

    # AST/ALT Ratio
    df_engineered['AST_ALT_Ratio'] = (
        df_engineered['Aspartate_Aminotransferase'] /
        (df_engineered['Alamine_Aminotransferase'] + 1e-5)
    )

    # TP/ALB Ratio
    df_engineered['TP_ALB_Ratio'] = (
        df_engineered['Total_Proteins'] /
        (df_engineered['Albumin'] + 1e-5)
    )

    # Bilirubin Ratio
    df_engineered['Bilirubin_Ratio'] = (
        df_engineered['Direct_Bilirubin'] /
        (df_engineered['Total_Bilirubin'] + 1e-5)
    )

    # Age groups
    df_engineered['Age_Group'] = pd.cut(
        df_engineered['Age'],
        bins=[0, 20, 40, 60, 100],
        labels=['Young', 'Adult', 'Middle_Aged', 'Senior']
    )

    features_info = pd.DataFrame({
        'Feature Name': ['AST_ALT_Ratio', 'TP_ALB_Ratio', 'Bilirubin_Ratio', 'Age_Group'],
        'Description': [
            'De Ritis ratio - Important liver function indicator',
            'Total Protein to Albumin ratio',
            'Direct to Total Bilirubin ratio',
            'Categorical age groups'
        ],
        'Justification': [
            'Captures relationship between liver enzymes',
            'Indicates protein synthesis function',
            'Indicates type of jaundice',
            'Non-linear age effects'
        ]
    })

    st.dataframe(features_info, use_container_width=True)

    # Show sample of new features
    st.subheader("Sample of Engineered Features")
    display_cols = ['Age', 'AST_ALT_Ratio', 'TP_ALB_Ratio', 'Bilirubin_Ratio', 'Age_Group']
    st.dataframe(df_engineered[display_cols].head(10), use_container_width=True)

    # 5. Encoding
    st.header("5. Categorical Variable Encoding")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gender Encoding")
        gender_encoding = pd.DataFrame({
            'Original': ['Male', 'Female'],
            'Encoded': [1, 0]
        })
        st.dataframe(gender_encoding, use_container_width=True)
        st.info("**Method**: Label Encoding for binary categorical variable")

    with col2:
        st.subheader("Age Group Encoding")
        st.write("One-hot encoding applied:")
        age_groups = ['Young', 'Adult', 'Middle_Aged', 'Senior']
        for group in age_groups:
            st.write(f"- Age_Group_{group}")
        st.info("**Method**: One-Hot Encoding for multi-class categorical variable")

    # 6. Feature Scaling
    st.header("6. Feature Scaling")

    st.markdown("""
    **Justification**: Many ML algorithms (SVM, KNN, Logistic Regression) are sensitive to feature scales.
    StandardScaler ensures zero mean and unit variance.
    """)

    # Demonstrate scaling
    sample_features = df_engineered[['Age', 'Total_Bilirubin', 'Alkaline_Phosphotase']].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(sample_features)
    scaled_df = pd.DataFrame(scaled_features, columns=sample_features.columns)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Before Scaling")
        st.dataframe(sample_features.describe(), use_container_width=True)

    with col2:
        st.subheader("After Scaling")
        st.dataframe(scaled_df.describe(), use_container_width=True)

    # Visualization
    fig = go.Figure()

    for col in sample_features.columns:
        fig.add_trace(go.Box(y=sample_features[col], name=f'{col} (Original)'))
        fig.add_trace(go.Box(y=scaled_df[col], name=f'{col} (Scaled)'))

    fig.update_layout(title="Feature Scaling Comparison", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

    # 7. Train-Test Split
    st.header("7. Train-Test Split Strategy")

    total_samples = len(df_engineered)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", total_samples)
    with col2:
        st.metric("Training Set (80%)", train_size)
    with col3:
        st.metric("Test Set (20%)", test_size)

    # Pie chart
    fig = px.pie(
        values=[train_size, test_size],
        names=['Training Set', 'Test Set'],
        title='Train-Test Split Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success("**Strategy**: 80-20 split with stratification to maintain class balance")

    # 8. Class Imbalance
    st.header("8. Handling Class Imbalance")

    target_counts = df_engineered['Target'].value_counts()
    imbalance_ratio = target_counts[1] / target_counts[2] if 2 in target_counts.index else 0

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Class 1 (Liver Patient)", target_counts.get(1, 0))
        st.metric("Class 2 (Non-Liver Patient)", target_counts.get(2, 0))
        st.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}")

    with col2:
        fig = px.bar(
            x=['Before SMOTE', 'After SMOTE'],
            y=[target_counts.get(2, 0), target_counts.get(1, 0)],
            title='Class Balance: Before and After SMOTE',
            labels={'x': 'Stage', 'y': 'Count'},
            color=['Before SMOTE', 'After SMOTE']
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Strategy**: Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance classes
    - Creates synthetic samples of minority class
    - Prevents model bias toward majority class
    - Improves recall for minority class
    """)

    # Summary
    st.header("9. Preprocessing Summary")

    summary_data = {
        'Step': [
            '1. Missing Values',
            '2. Duplicates',
            '3. Outliers',
            '4. Feature Engineering',
            '5. Encoding',
            '6. Scaling',
            '7. Train-Test Split',
            '8. Class Imbalance'
        ],
        'Method': [
            'Median Imputation',
            'Removal',
            'IQR Capping',
            'Domain Knowledge Features',
            'Label + One-Hot',
            'StandardScaler',
            '80-20 Stratified',
            'SMOTE'
        ],
        'Justification': [
            'Robust to outliers',
            'Prevent overfitting',
            'Preserve sample size',
            'Capture complex relationships',
            'ML algorithm compatibility',
            'Normalize feature scales',
            'Maintain class distribution',
            'Balance minority class'
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

except FileNotFoundError:
    st.error("Dataset file not found!")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
