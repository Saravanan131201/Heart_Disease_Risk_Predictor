# Heart Disease Risk Prediction System

An end-to-end Machine Learning project that predicts the likelihood of heart disease using patient clinical data and deploys the model through an interactive Streamlit web application.

This project focuses on interpretability, healthcare-relevant evaluation metrics, and real-world deployment.

---

## 📌 Problem Statement

Heart disease is one of the leading causes of death worldwide.  
Early risk assessment can help enable timely medical intervention and lifestyle changes.

🎯 **Goal:**  
Predict whether a patient is at higher or lower risk of heart disease using clinical health parameters.

---

## 📊 Dataset Overview

- Structured medical dataset
- Features include:
  - Age, Sex
  - Chest Pain Type
  - Resting Blood Pressure, Cholesterol
  - Fasting Blood Sugar (FBS)
  - ECG results
  - Exercise-induced angina
  - ST depression, Slope
  - Major vessels, Thalassemia
- Target variable:
  - Heart Disease (Yes / No)

---

## 🔍 Exploratory Data Analysis (EDA)

- Analyzed feature distributions across healthy and diseased patients
- Checked class balance of the target variable
- Identified important clinical indicators using:
  - Histograms
  - Distribution plots
  - Correlation heatmap

---

## ⚙️ Machine Learning Pipeline

1. Data loading & preprocessing  
2. Train-test split with stratification  
3. Feature scaling using StandardScaler  
4. Model training  
5. Hyperparameter tuning  
6. Model evaluation & comparison  
7. Deployment using Streamlit  

---

## 🤖 Models Used

### Logistic Regression
- Baseline and interpretable model
- Optimized using GridSearchCV
- Recall-focused optimization (healthcare-oriented)

### Random Forest Classifier
- Ensemble learning approach
- Captures non-linear relationships
- Handles feature interactions effectively

---

## 📈 Model Evaluation

Evaluation metrics used:
- Accuracy
- Recall
- Precision
- ROC-AUC
- Confusion Matrix
- ROC Curve comparison

Recall is emphasized due to its importance in healthcare risk prediction.

---

## 🖥️ Streamlit Web Application

Features:
- Model selector (Logistic Regression / Random Forest)
- Interactive sliders for clinical inputs
- Real-time prediction output
- Probability-based risk interpretation
- Gauge visualization for risk level
- Feature-based risk explanation
- Educational medical disclaimer

---

## ⚠️ Risk Interpretation

- Predictions are probability-based, not diagnoses
- Results are categorized into Higher Risk and Lower Risk
- Designed for awareness and early screening
- Not a substitute for professional medical advice

---

## 🛠️ Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib & Seaborn
- Streamlit
- Plotly
- Joblib

---

