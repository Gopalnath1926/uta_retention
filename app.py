import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model_rf.pkl", "rb") as file:
    model, feature_names = pickle.load(file)

st.title("UTA Retention Predictor")

# Collect input
st.header("Enter student information")
user_input = {}
for feature in feature_names:
    if feature in ['Gender', 'CapFlag', 'ExtraCurricularActivities', 'PellEligibility', 'FirstTermEnrolledCollege']:
        user_input[feature] = st.selectbox(feature, options=['Y', 'N'] if feature != 'Gender' else ['male', 'female'])
    else:
        user_input[feature] = st.number_input(feature, step=1.0)

# Predict button
if st.button("Predict Retention Probability"):
    input_df = pd.DataFrame([user_input])
    # Ensure encoding
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = pd.factorize(input_df[col])[0]
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"Estimated Probability of Retention: {probability * 100:.2f}%")

