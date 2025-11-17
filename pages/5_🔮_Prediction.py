"""
Prediction Page
Make predictions on new patient data
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Liver Disease Prediction")
st.markdown("Enter patient information to predict liver disease risk")
st.markdown("---")

# Load and train model
@st.cache_resource
def load_model():
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
    df['Target'] = df['Target'].map({1: 1, 2: 0})

    feature_cols = ['Age', 'Gender_Encoded', 'Total_Bilirubin', 'Direct_Bilirubin',
                   'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                   'Aspartate_Aminotransferase', 'Total_Proteins', 'Albumin',
                   'Albumin_Globulin_Ratio', 'AST_ALT_Ratio', 'TP_ALB_Ratio',
                   'Bilirubin_Ratio']

    X = df[feature_cols]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        random_state=42
    )
    model.fit(X_train_balanced, y_train_balanced)

    return model, scaler, le

try:
    model, scaler, label_encoder = load_model()

    st.success("✅ Model loaded successfully!")

    # Input Section
    st.header("1. Enter Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographic Information")

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=40,
            help="Patient's age in years"
        )

        gender = st.selectbox(
            "Gender",
            options=['Male', 'Female'],
            help="Patient's gender"
        )

        st.subheader("Bilirubin Levels")

        total_bilirubin = st.number_input(
            "Total Bilirubin (mg/dL)",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=0.1,
            help="Normal range: 0.1-1.2 mg/dL"
        )

        direct_bilirubin = st.number_input(
            "Direct Bilirubin (mg/dL)",
            min_value=0.0,
            max_value=50.0,
            value=0.3,
            step=0.1,
            help="Normal range: 0.0-0.3 mg/dL"
        )

        st.subheader("Enzymes")

        alkaline_phosphotase = st.number_input(
            "Alkaline Phosphotase (IU/L)",
            min_value=0,
            max_value=3000,
            value=200,
            step=10,
            help="Normal range: 44-147 IU/L"
        )

        alamine_aminotransferase = st.number_input(
            "Alamine Aminotransferase / ALT (IU/L)",
            min_value=0,
            max_value=3000,
            value=30,
            step=5,
            help="Normal range: 7-56 IU/L"
        )

    with col2:
        aspartate_aminotransferase = st.number_input(
            "Aspartate Aminotransferase / AST (IU/L)",
            min_value=0,
            max_value=3000,
            value=35,
            step=5,
            help="Normal range: 10-40 IU/L"
        )

        st.subheader("Proteins")

        total_proteins = st.number_input(
            "Total Proteins (g/dL)",
            min_value=0.0,
            max_value=15.0,
            value=7.0,
            step=0.1,
            help="Normal range: 6.0-8.3 g/dL"
        )

        albumin = st.number_input(
            "Albumin (g/dL)",
            min_value=0.0,
            max_value=10.0,
            value=3.5,
            step=0.1,
            help="Normal range: 3.5-5.5 g/dL"
        )

        albumin_globulin_ratio = st.number_input(
            "Albumin/Globulin Ratio",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Normal range: 1.0-2.5"
        )

    # Calculate derived features
    gender_encoded = 1 if gender == 'Male' else 0
    ast_alt_ratio = aspartate_aminotransferase / (alamine_aminotransferase + 1e-5)
    tp_alb_ratio = total_proteins / (albumin + 1e-5)
    bilirubin_ratio = direct_bilirubin / (total_bilirubin + 1e-5)

    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender_Encoded': [gender_encoded],
        'Total_Bilirubin': [total_bilirubin],
        'Direct_Bilirubin': [direct_bilirubin],
        'Alkaline_Phosphotase': [alkaline_phosphotase],
        'Alamine_Aminotransferase': [alamine_aminotransferase],
        'Aspartate_Aminotransferase': [aspartate_aminotransferase],
        'Total_Proteins': [total_proteins],
        'Albumin': [albumin],
        'Albumin_Globulin_Ratio': [albumin_globulin_ratio],
        'AST_ALT_Ratio': [ast_alt_ratio],
        'TP_ALB_Ratio': [tp_alb_ratio],
        'Bilirubin_Ratio': [bilirubin_ratio]
    })

    # Show input summary
    st.header("2. Input Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Features")
        original_features = pd.DataFrame({
            'Feature': ['Age', 'Gender', 'Total Bilirubin', 'Direct Bilirubin',
                       'Alkaline Phosphotase', 'ALT', 'AST', 'Total Proteins',
                       'Albumin', 'A/G Ratio'],
            'Value': [age, gender, total_bilirubin, direct_bilirubin,
                     alkaline_phosphotase, alamine_aminotransferase,
                     aspartate_aminotransferase, total_proteins,
                     albumin, albumin_globulin_ratio]
        })
        st.dataframe(original_features, use_container_width=True)

    with col2:
        st.subheader("Engineered Features")
        engineered_features = pd.DataFrame({
            'Feature': ['AST/ALT Ratio', 'TP/ALB Ratio', 'Bilirubin Ratio'],
            'Value': [f"{ast_alt_ratio:.3f}", f"{tp_alb_ratio:.3f}", f"{bilirubin_ratio:.3f}"]
        })
        st.dataframe(engineered_features, use_container_width=True)

    # Make Prediction
    st.header("3. Prediction")

    if st.button("🔮 Predict Liver Disease Risk", type="primary"):
        with st.spinner("Making prediction..."):
            # Scale input
            input_scaled = scaler.transform(input_data)

            # Predict
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]

            # Display results
            st.markdown("---")

            if prediction == 1:
                st.error("⚠️ **HIGH RISK** - Liver Disease Detected")
                confidence = prediction_proba[1] * 100

                st.metric("Confidence", f"{confidence:.1f}%")

                st.markdown(f"""
                ### Risk Assessment

                The model predicts a **high likelihood** of liver disease with {confidence:.1f}% confidence.

                **Important Note:**
                This is a machine learning prediction and should NOT replace professional medical diagnosis.

                **Recommended Actions:**
                - Consult a healthcare professional immediately
                - Schedule comprehensive liver function tests
                - Discuss symptoms and medical history with a doctor
                - Consider lifestyle modifications as advised by medical professionals
                """)

            else:
                st.success("✅ **LOW RISK** - No Liver Disease Detected")
                confidence = prediction_proba[0] * 100

                st.metric("Confidence", f"{confidence:.1f}%")

                st.markdown(f"""
                ### Risk Assessment

                The model predicts a **low likelihood** of liver disease with {confidence:.1f}% confidence.

                **Important Note:**
                This prediction does not guarantee absence of liver disease.

                **Recommended Actions:**
                - Maintain regular health check-ups
                - Continue healthy lifestyle practices
                - Monitor any symptoms
                - Consult a doctor if you experience any concerning symptoms
                """)

            # Probability chart
            st.subheader("Prediction Probabilities")

            prob_df = pd.DataFrame({
                'Class': ['No Disease', 'Liver Disease'],
                'Probability': [prediction_proba[0], prediction_proba[1]]
            })

            import plotly.express as px
            fig = px.bar(
                prob_df,
                x='Class',
                y='Probability',
                title='Prediction Probability Distribution',
                color='Probability',
                color_continuous_scale='RdYlGn_r',
                text='Probability'
            )
            fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig.update_layout(yaxis_title="Probability", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Feature contribution (simplified)
            st.subheader("Key Risk Factors")

            st.markdown("""
            The following factors contribute most to the prediction:
            """)

            # Get feature importance
            feature_importance = pd.DataFrame({
                'Feature': ['Age', 'Gender', 'Total Bilirubin', 'Direct Bilirubin',
                           'Alkaline Phosphotase', 'ALT', 'AST', 'Total Proteins',
                           'Albumin', 'A/G Ratio', 'AST/ALT Ratio', 'TP/ALB Ratio',
                           'Bilirubin Ratio'],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(5)

            st.dataframe(feature_importance, use_container_width=True)

    # Disclaimer
    st.markdown("---")
    st.warning("""
    **⚠️ Medical Disclaimer:**

    This prediction tool is for educational and informational purposes only.
    It is NOT a substitute for professional medical advice, diagnosis, or treatment.

    - Always seek the advice of qualified health providers
    - Never disregard professional medical advice based on this prediction
    - In case of medical emergency, contact emergency services immediately
    - This model has limitations and may not account for all clinical factors

    The developers of this application are not responsible for any decisions made based on these predictions.
    """)

    # Sample predictions
    st.header("4. Quick Test - Sample Cases")

    st.markdown("Try these pre-filled sample cases:")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Sample: Healthy Patient"):
            st.info("""
            **Healthy Patient Sample:**
            - Age: 35
            - Gender: Male
            - Total Bilirubin: 0.8
            - Direct Bilirubin: 0.2
            - Alkaline Phosphotase: 150
            - ALT: 25
            - AST: 30
            - Total Proteins: 7.0
            - Albumin: 4.0
            - A/G Ratio: 1.3
            """)

    with col2:
        if st.button("Sample: Moderate Risk"):
            st.warning("""
            **Moderate Risk Sample:**
            - Age: 50
            - Gender: Male
            - Total Bilirubin: 2.5
            - Direct Bilirubin: 1.0
            - Alkaline Phosphotase: 300
            - ALT: 80
            - AST: 95
            - Total Proteins: 6.5
            - Albumin: 3.0
            - A/G Ratio: 0.9
            """)

    with col3:
        if st.button("Sample: High Risk"):
            st.error("""
            **High Risk Sample:**
            - Age: 65
            - Gender: Male
            - Total Bilirubin: 5.0
            - Direct Bilirubin: 2.5
            - Alkaline Phosphotase: 500
            - ALT: 150
            - AST: 200
            - Total Proteins: 5.5
            - Albumin: 2.5
            - A/G Ratio: 0.7
            """)

except FileNotFoundError:
    st.error("Dataset file not found! Please ensure the dataset is in the correct location.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
