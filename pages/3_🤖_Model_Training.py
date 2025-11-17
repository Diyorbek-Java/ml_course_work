"""
Model Training Page
Train and compare multiple machine learning models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

st.title("🤖 Model Training & Comparison")
st.markdown("Train and evaluate multiple machine learning algorithms")
st.markdown("---")

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    column_names = [
        'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
        'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
        'Aspartate_Aminotransferase', 'Total_Proteins', 'Albumin',
        'Albumin_Globulin_Ratio', 'Target'
    ]
    df = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv', names=column_names)

    # Preprocessing
    df['Albumin_Globulin_Ratio'].fillna(df['Albumin_Globulin_Ratio'].median(), inplace=True)
    df = df.drop_duplicates()

    # Feature engineering
    df['AST_ALT_Ratio'] = df['Aspartate_Aminotransferase'] / (df['Alamine_Aminotransferase'] + 1e-5)
    df['TP_ALB_Ratio'] = df['Total_Proteins'] / (df['Albumin'] + 1e-5)
    df['Bilirubin_Ratio'] = df['Direct_Bilirubin'] / (df['Total_Bilirubin'] + 1e-5)

    # Encoding
    le = LabelEncoder()
    df['Gender_Encoded'] = le.fit_transform(df['Gender'])

    # Convert target to binary
    df['Target'] = df['Target'].map({1: 1, 2: 0})

    # Select features
    feature_cols = ['Age', 'Gender_Encoded', 'Total_Bilirubin', 'Direct_Bilirubin',
                   'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                   'Aspartate_Aminotransferase', 'Total_Proteins', 'Albumin',
                   'Albumin_Globulin_Ratio', 'AST_ALT_Ratio', 'TP_ALB_Ratio',
                   'Bilirubin_Ratio']

    X = df[feature_cols]
    y = df['Target']

    return X, y, df

try:
    X, y, df = load_and_preprocess_data()

    # Model Selection
    st.header("1. Model Selection")

    st.markdown("""
    We compare the following algorithms:

    1. **Logistic Regression** - Simple, interpretable baseline
    2. **Decision Tree** - Non-linear, handles interactions
    3. **Random Forest** - Ensemble method, robust to overfitting
    4. **Gradient Boosting** - Sequential ensemble, high performance
    5. **Support Vector Machine (SVM)** - Effective in high-dimensional space
    6. **K-Nearest Neighbors (KNN)** - Non-parametric, simple
    7. **Naive Bayes** - Probabilistic classifier
    """)

    # Training Configuration
    st.header("2. Training Configuration")

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        use_smote = st.checkbox("Use SMOTE for class balancing", value=True)

    with col2:
        random_state = st.number_input("Random State", 0, 100, 42)
        st.info(f"Training Set: {int((1-test_size)*100)}%\\nTest Set: {int(test_size*100)}%")

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train

    st.success(f"Data prepared: {len(X_train_balanced)} training samples, {len(X_test_scaled)} test samples")

    # Train Models
    st.header("3. Train Models")

    if st.button("🚀 Train All Models"):
        with st.spinner("Training models..."):

            models = {
                'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
                'Decision Tree': DecisionTreeClassifier(random_state=random_state),
                'Random Forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
                'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
                'SVM': SVC(probability=True, random_state=random_state),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB()
            }

            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, (name, model) in enumerate(models.items()):
                status_text.text(f"Training {name}...")

                # Train
                model.fit(X_train_balanced, y_train_balanced)

                # Predict
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                # Metrics
                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()

                results.append({
                    'Model': name,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, zero_division=0),
                    'Recall': recall_score(y_test, y_pred, zero_division=0),
                    'F1-Score': f1_score(y_test, y_pred, zero_division=0),
                    'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
                    'TPR': tp / (tp + fn) if (tp + fn) > 0 else 0,
                    'TNR': tn / (tn + fp) if (tn + fp) > 0 else 0
                })

                progress_bar.progress((idx + 1) / len(models))

            status_text.text("Training completed!")
            progress_bar.empty()

            # Results DataFrame
            results_df = pd.DataFrame(results)
            results_df = results_df.round(4)

            st.session_state['results_df'] = results_df

    # Display Results
    if 'results_df' in st.session_state:
        st.header("4. Model Comparison Results")

        results_df = st.session_state['results_df']

        # Table
        st.subheader("Performance Metrics")
        st.dataframe(results_df, use_container_width=True)

        # Best models
        st.subheader("Best Performing Models by Metric")

        metrics_to_show = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'TPR', 'TNR']
        best_models = []

        for metric in metrics_to_show:
            best_model = results_df.loc[results_df[metric].idxmax()]
            best_models.append({
                'Metric': metric,
                'Best Model': best_model['Model'],
                'Score': best_model[metric]
            })

        best_models_df = pd.DataFrame(best_models)
        st.dataframe(best_models_df, use_container_width=True)

        # Visualizations
        st.header("5. Performance Visualizations")

        # Metric selection
        col1, col2 = st.columns(2)

        with col1:
            metric_to_plot = st.selectbox(
                "Select metric to visualize:",
                ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'TPR', 'TNR']
            )

        # Bar chart
        fig = px.bar(
            results_df,
            x='Model',
            y=metric_to_plot,
            title=f'Model Comparison - {metric_to_plot}',
            color=metric_to_plot,
            color_continuous_scale='Viridis',
            text=metric_to_plot
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Multi-metric comparison
        st.subheader("Multi-Metric Comparison")

        # Radar chart
        fig = go.Figure()

        for idx, row in results_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['AUC-ROC']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                fill='toself',
                name=row['Model']
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Multi-Metric Performance Radar Chart"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        st.subheader("Performance Heatmap")

        metrics_for_heatmap = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'TPR', 'TNR']
        heatmap_data = results_df[['Model'] + metrics_for_heatmap].set_index('Model')

        fig = px.imshow(
            heatmap_data.T,
            labels=dict(x="Model", y="Metric", color="Score"),
            x=heatmap_data.index,
            y=metrics_for_heatmap,
            color_continuous_scale='RdYlGn',
            aspect='auto',
            text_auto='.3f'
        )
        fig.update_layout(title="Model Performance Heatmap")
        st.plotly_chart(fig, use_container_width=True)

        # Model Ranking
        st.header("6. Overall Model Ranking")

        # Calculate average rank
        ranking_data = []
        for metric in metrics_for_heatmap:
            ranks = results_df[metric].rank(ascending=False)
            for idx, model in enumerate(results_df['Model']):
                ranking_data.append({
                    'Model': model,
                    'Metric': metric,
                    'Rank': ranks.iloc[idx]
                })

        ranking_df = pd.DataFrame(ranking_data)
        avg_ranking = ranking_df.groupby('Model')['Rank'].mean().sort_values()

        st.subheader("Average Ranking Across All Metrics")

        fig = px.bar(
            x=avg_ranking.values,
            y=avg_ranking.index,
            orientation='h',
            labels={'x': 'Average Rank (Lower is Better)', 'y': 'Model'},
            title='Overall Model Ranking',
            color=avg_ranking.values,
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"🏆 Best Overall Model: **{avg_ranking.index[0]}** (Avg Rank: {avg_ranking.values[0]:.2f})")

except FileNotFoundError:
    st.error("Dataset file not found!")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
