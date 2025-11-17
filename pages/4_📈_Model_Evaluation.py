"""
Model Evaluation Page
Detailed evaluation and analysis of trained models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")

st.title("📈 Model Evaluation & Analysis")
st.markdown("Deep dive into model performance and evaluation metrics")
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

    # Convert target
    df['Target'] = df['Target'].map({1: 1, 2: 0})

    # Select features
    feature_cols = ['Age', 'Gender_Encoded', 'Total_Bilirubin', 'Direct_Bilirubin',
                   'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                   'Aspartate_Aminotransferase', 'Total_Proteins', 'Albumin',
                   'Albumin_Globulin_Ratio', 'AST_ALT_Ratio', 'TP_ALB_Ratio',
                   'Bilirubin_Ratio']

    X = df[feature_cols]
    y = df['Target']

    return X, y, feature_cols

try:
    X, y, feature_cols = load_and_preprocess_data()

    # Train model
    @st.cache_resource
    def train_model():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            random_state=42
        )
        model.fit(X_train_balanced, y_train_balanced)

        return model, X_test_scaled, y_test, scaler

    model, X_test_scaled, y_test, scaler = train_model()

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 1. Confusion Matrix
    st.header("1. Confusion Matrix Analysis")

    cm = confusion_matrix(y_test, y_pred)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Heatmap
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['No Disease', 'Disease'],
            y=['No Disease', 'Disease'],
            text_auto=True,
            color_continuous_scale='Blues',
            aspect='auto'
        )
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()

        st.subheader("Confusion Matrix Breakdown")
        st.metric("True Positives (TP)", tp)
        st.metric("True Negatives (TN)", tn)
        st.metric("False Positives (FP)", fp)
        st.metric("False Negatives (FN)", fn)

        st.markdown("""
        **Interpretation:**
        - **TP**: Correctly identified liver disease
        - **TN**: Correctly identified no disease
        - **FP**: False alarm (predicted disease when none)
        - **FN**: Missed disease (failed to detect)
        """)

    # 2. ROC Curve
    st.header("2. ROC Curve Analysis")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash', width=2)
        ))
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("AUC-ROC Score", f"{roc_auc:.4f}")

        st.markdown("""
        **AUC-ROC Interpretation:**
        - 1.0: Perfect classifier
        - 0.9-1.0: Excellent
        - 0.8-0.9: Very good
        - 0.7-0.8: Good
        - 0.6-0.7: Acceptable
        - 0.5: No better than random
        """)

        if roc_auc >= 0.9:
            st.success("Excellent performance!")
        elif roc_auc >= 0.8:
            st.success("Very good performance!")
        elif roc_auc >= 0.7:
            st.info("Good performance")
        else:
            st.warning("Room for improvement")

    # 3. Precision-Recall Curve
    st.header("3. Precision-Recall Curve")

    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name='Precision-Recall Curve',
        line=dict(color='green', width=2),
        fill='tozeroy'
    ))
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        hovermode='closest'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretation:**
    - High precision & high recall = Ideal model
    - High precision & low recall = Conservative predictions
    - Low precision & high recall = Aggressive predictions
    - Trade-off between precision and recall
    """)

    # 4. Classification Report
    st.header("4. Detailed Classification Report")

    report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df.round(3), use_container_width=True)

    st.markdown("""
    **Metrics Explanation:**
    - **Precision**: Of all positive predictions, how many were correct?
    - **Recall (Sensitivity)**: Of all actual positives, how many were detected?
    - **F1-Score**: Harmonic mean of precision and recall
    - **Support**: Number of actual occurrences in the test set
    """)

    # 5. Feature Importance
    st.header("5. Feature Importance Analysis")

    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance Ranking',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 5 Features")
        st.dataframe(feature_importance.head().round(4), use_container_width=True)

        st.markdown("""
        **Key Insights:**
        Top features indicate which clinical markers
        are most predictive of liver disease.
        """)

    # 6. Cross-Validation Analysis
    st.header("6. Cross-Validation Analysis")

    with st.spinner("Performing cross-validation..."):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Prepare data for CV
        X_train_full, _, y_train_full, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler_cv = StandardScaler()
        X_scaled_cv = scaler_cv.fit_transform(X_train_full)

        smote_cv = SMOTE(random_state=42)
        X_balanced_cv, y_balanced_cv = smote_cv.fit_resample(X_scaled_cv, y_train_full)

        cv_scores = cross_val_score(
            model, X_balanced_cv, y_balanced_cv,
            cv=cv, scoring='roc_auc', n_jobs=-1
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean CV Score", f"{cv_scores.mean():.4f}")
    with col2:
        st.metric("Std Deviation", f"{cv_scores.std():.4f}")
    with col3:
        st.metric("Min-Max Range", f"{cv_scores.min():.3f} - {cv_scores.max():.3f}")

    # Plot CV scores
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f'Fold {i+1}' for i in range(len(cv_scores))],
        y=cv_scores,
        name='CV Scores',
        marker_color='lightblue'
    ))
    fig.add_hline(
        y=cv_scores.mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {cv_scores.mean():.4f}"
    )
    fig.update_layout(title='Cross-Validation Scores by Fold', yaxis_title='AUC-ROC Score')
    st.plotly_chart(fig, use_container_width=True)

    # 7. Error Analysis
    st.header("7. Prediction Error Analysis")

    # Get misclassified samples
    errors = X_test_scaled[y_test != y_pred]
    error_targets = y_test[y_test != y_pred]
    error_predictions = y_pred[y_test != y_pred]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Errors", len(errors))
        st.metric("Error Rate", f"{(len(errors) / len(y_test) * 100):.2f}%")

    with col2:
        fp_count = ((y_test == 0) & (y_pred == 1)).sum()
        fn_count = ((y_test == 1) & (y_pred == 0)).sum()

        st.metric("False Positives", fp_count)
        st.metric("False Negatives", fn_count)

    # Error distribution
    error_type = []
    for i in range(len(error_targets)):
        if error_targets.iloc[i] == 0:
            error_type.append('False Positive')
        else:
            error_type.append('False Negative')

    error_df = pd.DataFrame({'Error Type': error_type})
    error_counts = error_df['Error Type'].value_counts()

    fig = px.pie(
        values=error_counts.values,
        names=error_counts.index,
        title='Error Type Distribution',
        color=error_counts.index,
        color_discrete_map={'False Positive': '#ff7f0e', 'False Negative': '#d62728'}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Clinical Impact:**
    - **False Negatives**: More critical - missed disease diagnosis
    - **False Positives**: Additional testing burden, but safer
    """)

    # 8. Model Evaluation Summary
    st.header("8. Evaluation Summary")

    summary_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Specificity'],
        'Score': [
            (y_test == y_pred).mean(),
            report['Disease']['precision'],
            report['Disease']['recall'],
            report['Disease']['f1-score'],
            roc_auc,
            report['No Disease']['recall']
        ],
        'Interpretation': [
            'Overall correctness',
            'Positive prediction accuracy',
            'Disease detection rate',
            'Balance of precision & recall',
            'Discriminative ability',
            'True negative rate'
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df['Score'] = summary_df['Score'].round(4)
    st.dataframe(summary_df, use_container_width=True)

    # Overall assessment
    avg_score = summary_df['Score'].mean()

    if avg_score >= 0.9:
        st.success(f"🌟 Excellent Model Performance! Average Score: {avg_score:.3f}")
    elif avg_score >= 0.8:
        st.success(f"✅ Very Good Model Performance! Average Score: {avg_score:.3f}")
    elif avg_score >= 0.7:
        st.info(f"👍 Good Model Performance! Average Score: {avg_score:.3f}")
    else:
        st.warning(f"⚠️ Model Performance Needs Improvement. Average Score: {avg_score:.3f}")

except FileNotFoundError:
    st.error("Dataset file not found!")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
