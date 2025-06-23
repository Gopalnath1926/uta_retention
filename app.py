
import streamlit as st
import pickle
import pandas as pd

st.title("UTA One-Year Retention Predictor")

# Load model and features
with open("model_rf.pkl", "rb") as file:
    model, feature_names = pickle.load(file)

# Collect input from user
input_data = {}
for feature in feature_names:
    input_data[feature] = st.text_input(f"Enter {feature}:")

# When user clicks Predict
if st.button("Predict Retention Probability"):
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df.astype({col: "float64" for col in input_df.columns})
        prob = model.predict_proba(input_df)[0][1]
        st.success(f"Estimated Probability of Retention: {prob * 100:.2f}%")
    except Exception as e:
        st.error(f"Error: {str(e)}")
