import streamlit as st
import pickle
import pandas as pd

# Load the model
with open("logistic_model.pkl", "rb") as file:
    model, feature_names, label_encoders = pickle.load(file)

st.title("UTA Retention Probability Prediction (Logistic Regression)")

# User input fields
input_data = {}
for feature in feature_names:
    if feature in label_encoders:
        options = label_encoders[feature].classes_
        input_data[feature] = st.selectbox(f"{feature}:", options)
    else:
        input_data[feature] = st.number_input(f"{feature}:", step=0.1)

# Prediction
if st.button("Predict Retention Probability"):
    input_df = pd.DataFrame([input_data])

    # Apply encoding for categorical fields
    for col, le in label_encoders.items():
        input_df[col] = le.transform(input_df[col])

    probability = model.predict_proba(input_df)[0][1]
    st.success(f"Predicted probability of being retained: {probability * 100:.2f}%")

