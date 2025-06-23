
import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('../model/random_forest_model.pkl')
encoders = joblib.load('../model/label_encoders.pkl')
feature_order = joblib.load('../model/feature_order.pkl')

st.set_page_config(page_title="Retention Probability Predictor", layout="centered")
st.title("🎓 Student Retention Probability Predictor")

st.markdown("This app predicts the **probability** of a student being retained after their first year.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["F", "M"])
        cap_flag = st.selectbox("CAP Eligible", ["Y", "N"])
        pell_eligibility = st.selectbox("Pell Eligibility", ["Y", "N"])
    with col2:
        hs_gpa = st.slider("High School GPA", 0.0, 5.0, 3.0, 0.1)
        math_sat = st.slider("Math SAT Score", 200, 800, 500, 10)
        verbal_sat = st.slider("Verbal SAT Score", 200, 800, 500, 10)

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        "Gender": gender,
        "CapFlag": cap_flag,
        "PellEligibility": pell_eligibility,
        "HSGPA": hs_gpa,
        "MathSAT": math_sat,
        "VerbalSAT": verbal_sat
    }])

    for col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

    input_df = input_df[feature_order]
    pred_prob = model.predict_proba(input_df)[0][1]

    st.success(f"🔮 Predicted Retention Probability: **{pred_prob * 100:.2f}%**")
