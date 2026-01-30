import streamlit as st
import numpy as np
import joblib
from PIL import Image
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    lr_model = joblib.load("heart_logistic.pkl")
    rf_model = joblib.load("heart_rf.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    return lr_model, rf_model, scaler

lr_model, rf_model, scaler = load_models()

# ---------------- SIDEBAR (UNCHANGED) ----------------


st.sidebar.title("📘 About This App")

image = Image.open("heart.png")
st.sidebar.image(image, width=340)

st.sidebar.markdown("""
This app predicts whether a person is likely to have **heart disease**
using a Logistic Regression and Random Forest.
""")

st.sidebar.info("""
    **Heart-Healthy Lifestyle Tips:**
    - Regular health check-ups
    - Balanced diet (low fat & low salt)
    - Exercise at least 30 minutes daily
    - Monitor blood pressure & cholesterol
    - Avoid smoking and excess alcohol
    """)

st.sidebar.markdown("---")
st.sidebar.caption("This is for educational purposes only")
st.sidebar.caption("Not a substitute for professional medical advice")

# ---------------- MAIN UI ----------------
st.title("🫀 AI-Based Heart Disease Risk Prediction System")


# -------- MODEL SELECTION --------

st.subheader("🔀 Select Prediction Model")

st.markdown(
        """
        <style>
        div[data-baseweb="select"] {
            max-width: 450px;
            font-size: 18px;

        }
        </style>
        """,
        unsafe_allow_html=True
    )

model_choice = st.selectbox(
        " ",
        [
            "Logistic Regression (Optimized)",
            "Random Forest Classifier (Best Performance)"
        ],
        index=1,
        label_visibility="collapsed"
    )


model = lr_model if "Logistic" in model_choice else rf_model

# -------- INPUT SECTION --------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", [1, 0],
                       format_func=lambda x: "Male" if x == 1 else "Female")

    st.subheader("Heart Metrics")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
        format_func=lambda x: [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ][x])

    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.selectbox("Resting ECG", [0, 1, 2],
        format_func=lambda x: [
            "Normal",
            "ST-T Abnormality",
            "LV Hypertrophy"
        ][x])

with col2:
    st.subheader("Exercise & Vessels")
    thalach = st.slider("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
    slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2],
        format_func=lambda x: [
            "Upsloping",
            "Flat",
            "Downsloping"
        ][x])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3],
        format_func=lambda x: [
            "Normal",
            "Fixed Defect",
            "Reversible Defect"
        ][x - 1])

# ---------------- PREDICTION ----------------
st.markdown("---")

if st.button("🔍 Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                             restecg, thalach, exang, oldpeak,
                             slope, ca, thal]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]



  
    colA, spacer, colB = st.columns([2.3, 0.3, 1])

    with colA:  
        # ---------------- RESULT ----------------
        st.subheader("🎯 Prediction Results")

        if prediction == 1:
            st.error("⚠️ **ELEVATED CARDIAC RISK**")
        else:
            st.success("✅ **LOW CARDIAC RISK**")


        # -------- PROBABILITY BREAKDOWN --------
        st.markdown("### 📊 Probability Breakdown")

        prob_low = 1 - probability
        prob_high = probability

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div style="
                    background: {'#FDFFB8' if prob_low < 0.5 else '#bbf7d0'};
                    padding: 3px;
                    border-radius: 14px;
                    text-align: center;
                    font-weight: 600;
                    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
                ">
                    <h4 style="margin-bottom:6px;">Lower Risk</h4>
                    <h2 style="margin-top:0;">{prob_low*100:.1f}%</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(prob_low)

        with col2:
            st.markdown(
                f"""
                <div style="
                    background: #fee2e2;
                    padding: 3px;
                    border-radius: 14px;
                    text-align: center;
                    font-weight: 600;
                    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
                ">
                    <h4 style="margin-bottom:6px;">Higher Risk</h4>
                    <h2 style="margin-top:0;">{prob_high*100:.1f}%</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(prob_high)
         
    # -------- GAUGE CHART --------
    with colB:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Risk Level"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig, width="stretch")




    # -------- RISK FACTOR ANALYSIS --------
    # st.markdown("### ⚠️ Risk Factor Analysis")

    # positives = []
    # warnings = []

    # # Chest Pain
    # if cp == 0:
    #     warnings.append("Typical angina symptoms detected")
    # elif cp == 1:
    #     warnings.append("Atypical angina symptoms")
    # elif cp == 2:
    #     positives.append("Non-anginal chest pain")
    # else:
    #     warnings.append("Asymptomatic chest pain (silent risk)")

    # # Blood Pressure
    # if trestbps < 130:
    #     positives.append("Normal blood pressure")
    # else:
    #     warnings.append("High blood pressure")

    # # Cholesterol
    # if chol < 200:
    #     positives.append("Healthy cholesterol level")
    # elif chol < 240:
    #     warnings.append("Borderline high cholesterol")
    # else:
    #     warnings.append("High cholesterol")

    # # Fasting Blood Sugar
    # if fbs == 1:
    #     warnings.append("High fasting blood sugar (possible diabetes risk)")
    # else:
    #     positives.append("Normal fasting blood sugar")

    # # ECG
    # if restecg == 0:
    #     positives.append("Normal ECG result")
    # else:
    #     warnings.append("Abnormal ECG pattern")

    # # Max Heart Rate
    # if thalach >= 150:
    #     positives.append("Good maximum heart rate")
    # else:
    #     warnings.append("Low exercise heart rate capacity")

    # # Exercise Angina
    # if exang == 1:
    #     warnings.append("Exercise-induced angina detected")

    # # ST Depression
    # if oldpeak > 2.0:
    #     warnings.append("Significant ST depression detected")
    # else:
    #     positives.append("Normal ST depression")

    # # Slope
    # if slope == 1 or slope == 2:
    #     warnings.append("Abnormal ST slope pattern")
    # else:
    #     positives.append("Normal ST slope pattern")

    # # Major Vessels
    # if ca >= 1:
    #     warnings.append(f"{ca} major vessels affected")
    # else:
    #     positives.append("No major vessels affected")

    # # Thalassemia
    # if thal != 1:
    #     warnings.append("Abnormal thalassemia test")
    # else:
    #     positives.append("Normal thalassemia test")

    # # ---------------- DISPLAY ----------------
    # if positives:
    #     st.success("**Positive Health Indicators:**")
    #     for p in positives:
    #         st.write("🟢", p)

    # if warnings:
    #     st.warning("**Potential Risk Factors:**")
    #     for w in warnings:
    #         st.write("🔴", w)


    # -------- RECOMMENDATIONS --------
    st.markdown("### 💡 Recommendations")



    if prediction == 1:
        st.error("""
        **High Risk – Take Action:**
        - Consult a cardiologist or healthcare professional
        - Schedule a heart health screening (ECG, lipid profile, BP)
        - Track blood pressure and heart rate
        - Reduce saturated fats, sugar, and sodium
        - Avoid smoking and limit alcohol
        """)
    else:
        st.success("""
        **Low Risk – Keep It Up:**
        - Maintain your current healthy habits
        - Stay physically active
        - Manage stress and sleep well
        - Maintain a healthy weight
        - Get routine heart check-ups
        """)

