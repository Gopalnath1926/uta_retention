
import os
import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
MODEL_PATH = os.path.join(os.path.dirname(__file__), "logistic_model.joblib")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoders.joblib")

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)

st.title("UTA Retention Probability Predictor (Logistic Regression)")

def user_input():
    input_data = {}
    for feature in model.feature_names_in_:
        if feature in label_encoders:
            options = label_encoders[feature].classes_.tolist()
            choice = st.selectbox(f"{feature}", options)
            input_data[feature] = label_encoders[feature].transform([choice])[0]
        else:
            input_data[feature] = st.number_input(f"{feature}", value=0.0)
    return pd.DataFrame([input_data])

input_df = user_input()

if st.button("Predict Retention Probability"):
    proba = model.predict_proba(input_df)[0, 1]
    st.success(f"Estimated Retention Probability: **{proba * 100:.2f}%**")
