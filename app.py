
import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model_rf.pkl", "rb") as file:
    model, feature_names = pickle.load(file)

st.title("UTA Retention Probability Predictor")

# Input form
input_data = {}
for feature in feature_names:
    if feature.lower().endswith("gpa"):
        input_data[feature] = st.number_input(f"{feature}", min_value=0.0, max_value=4.0, step=0.01)
    elif feature.lower().endswith("income") or "sat" in feature.lower():
        input_data[feature] = st.number_input(f"{feature}", min_value=0)
    else:
        input_data[feature] = st.text_input(f"{feature}")

# Predict
if st.button("Predict Retention Probability"):
    input_df = pd.DataFrame([input_data])
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df[col] = pd.factorize(input_df[col])[0]
    probability = model.predict_proba(input_df)[0][1] * 100
    st.success(f"Predicted Probability of Retention: {probability:.2f}%")
