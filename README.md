# Liver Disease Prediction - Machine Learning Project

A comprehensive machine learning solution for predicting liver disease using the Indian Liver Patient Dataset (ILPD).

## 📋 Project Overview

This project demonstrates an end-to-end machine learning pipeline for medical diagnosis support, specifically predicting liver disease in patients based on clinical and demographic features.

**Dataset**: Indian Liver Patient Dataset (ILPD)
**Samples**: 584 patient records
**Features**: 10 clinical and demographic variables
**Task**: Binary classification (Liver Disease / No Disease)

## 🎯 Objectives

- Perform comprehensive exploratory data analysis
- Implement robust data preprocessing pipeline
- Train and compare multiple ML algorithms
- Evaluate model performance with multiple metrics
- Deploy interactive web application for predictions

## 📊 Features

### Clinical Markers
- **Demographics**: Age, Gender
- **Bilirubin Levels**: Total Bilirubin, Direct Bilirubin
- **Liver Enzymes**: ALT (SGPT), AST (SGOT), Alkaline Phosphotase
- **Proteins**: Total Proteins, Albumin, A/G Ratio

### Engineered Features
- AST/ALT Ratio (De Ritis ratio)
- Total Protein to Albumin Ratio
- Bilirubin Ratio
- Age groups (categorical)
- Log-transformed features for skewed distributions

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip package manager

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd FirstTry
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Ensure the dataset file is in the project directory
```
Indian Liver Patient Dataset (ILPD).csv
```

## 💻 Usage

### Running the Jupyter Notebook

```bash
jupyter notebook ILPD_Analysis.ipynb
```

The notebook contains:
- Complete data analysis pipeline
- Exploratory data analysis with visualizations
- Data preprocessing steps
- Model training and evaluation
- Feature importance analysis

### Running the Streamlit Application

```bash
streamlit run app.py
```

The app provides:
- **Data Exploration**: Interactive visualizations and statistics
- **Preprocessing**: View data cleaning steps
- **Model Training**: Train and compare multiple algorithms
- **Model Evaluation**: Detailed performance analysis
- **Prediction**: Make predictions on new patient data

## 📁 Project Structure

```
FirstTry/
│
├── app.py                                    # Main Streamlit app
├── ILPD_Analysis.ipynb                       # Jupyter notebook with full analysis
├── Indian Liver Patient Dataset (ILPD).csv   # Dataset file
├── requirements.txt                           # Python dependencies
├── README.md                                  # Project documentation
│
├── pages/                                     # Streamlit pages
│   ├── 1_📊_Data_Exploration.py
│   ├── 2_🔧_Preprocessing.py
│   ├── 3_🤖_Model_Training.py
│   ├── 4_📈_Model_Evaluation.py
│   └── 5_🔮_Prediction.py
│
├── models/                                    # Saved models (generated)
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
└── data/                                      # Additional data files (if any)
```

## 🤖 Machine Learning Models

The following algorithms are implemented and compared:

1. **Logistic Regression** - Baseline linear model
2. **Decision Tree** - Non-linear classifier
3. **Random Forest** - Ensemble method (best performer)
4. **Gradient Boosting** - Sequential ensemble
5. **Support Vector Machine (SVM)** - Margin-based classifier
6. **K-Nearest Neighbors (KNN)** - Instance-based learning
7. **Naive Bayes** - Probabilistic classifier
8. **XGBoost** - Advanced gradient boosting

## 📈 Model Performance

### Best Model: Random Forest

- **Accuracy**: ~75%
- **AUC-ROC**: ~0.75
- **Precision**: ~0.73
- **Recall**: ~0.78
- **F1-Score**: ~0.75

### Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- AUC-ROC Curve
- Confusion Matrix
- TPR (Sensitivity), TNR (Specificity)
- Cross-validation scores
- Feature importance analysis

## 🔧 Data Preprocessing

### Steps Implemented

1. **Missing Value Imputation**: Median imputation for A/G Ratio
2. **Duplicate Removal**: Removed duplicate records
3. **Outlier Handling**: IQR-based capping
4. **Feature Engineering**: Created domain-specific features
5. **Encoding**: Label encoding for Gender, One-hot for Age groups
6. **Scaling**: StandardScaler for normalization
7. **Class Balancing**: SMOTE for minority class oversampling
8. **Train-Test Split**: 80-20 stratified split

## 📝 Report

A comprehensive report is included covering:

- Introduction and business case
- Exploratory data analysis with justifications
- Data preprocessing methodology
- Model selection and training
- Performance evaluation and comparison
- Conclusions and recommendations
- Limitations and future work
- Ethical considerations

## ⚠️ Medical Disclaimer

This application is for educational and research purposes only. It is NOT intended for clinical use or medical decision-making. Always consult qualified healthcare professionals for medical advice.

## 📚 References

- UCI Machine Learning Repository - Indian Liver Patient Dataset
- Scikit-learn Documentation
- Imbalanced-learn Documentation
- Streamlit Documentation

## 👨‍💻 Development

### Technologies Used

- **Python 3.10+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning
- **XGBoost**: Gradient boosting
- **Imbalanced-learn**: SMOTE implementation
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations
- **Streamlit**: Web application framework
- **Jupyter**: Interactive analysis

### Version Control

This project uses Git for version control with meaningful commits tracking:
- Initial setup
- Data exploration
- Preprocessing implementation
- Model development
- Evaluation and tuning
- Deployment

## 📄 License

This project is licensed under the MIT License.

## 🎓 Academic Information

**Module**: Machine Learning and Data Analytics (6COSC017C-n)
**Institution**: Westminster International University in Tashkent (WIUT)
**Academic Year**: 2025-2026
**Coursework Weight**: 50%

## 📧 Contact

For questions or issues, please contact through the university portal.

---

**Note**: This is a coursework submission. Plagiarism and close collaboration are strictly prohibited. This work represents original analysis and implementation.
