import streamlit as st
import pandas as pd
import pickle

# Load model and metadata
with open("model_rf.pkl", "rb") as f:
    model, feature_names, label_encoders = pickle.load(f)

st.title("UTA One-Year Retention Predictor")

# Input form
user_input = {}
for feature in feature_names:
    if feature in label_encoders:
        options = label_encoders[feature].classes_
        user_input[feature] = st.selectbox(f"{feature}:", options)
    else:
        user_input[feature] = st.number_input(f"{feature}:", step=0.1)

# Prepare input DataFrame
input_df = pd.DataFrame([user_input])

# Encode using stored label encoders
for feature in label_encoders:
    input_df[feature] = label_encoders[feature].transform(input_df[feature])

# Predict
if st.button("Predict Retention Probability"):
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"Predicted probability of retention: {probability * 100:.2f}%")
