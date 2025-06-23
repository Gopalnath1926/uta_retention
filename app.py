# app.py
import streamlit as st
import pandas as pd
import pickle

# Load trained model and feature names
with open('logistic_model.pkl', 'rb') as f:
    model, feature_names = pickle.load(f)

st.title("UTA Retention Probability Predictor")

# Collect user input
inputs = {}
for feature in feature_names:
    user_input = st.number_input(f"Enter {feature}:", value=0.0)
    inputs[feature] = user_input

# Convert to DataFrame
input_df = pd.DataFrame([inputs])

# Predict
if st.button("Predict Retention Probability"):
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"Predicted probability of retention: {probability*100:.2f}%")


