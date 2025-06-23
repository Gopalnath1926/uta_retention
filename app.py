import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load trained model
with open("model_rf.pkl", "rb") as file:
    model = pickle.load(file)

# Expected feature columns
expected_features = ['Age', 'Gender', 'College', ...]  # Fill in actual training features in order

# Dummy encoders (use saved encoders in real app)
encoders = {
    'Gender': LabelEncoder().fit(['Female', 'Male']),  # same classes as training
    'College': LabelEncoder().fit([...])  # same categories
}

# User input UI
st.title("UTA Retention Predictor")
gender = st.selectbox("Gender", ['Female', 'Male'])
age = st.number_input("Age", min_value=15, max_value=60, step=1)
college = st.selectbox("College", [...])  # same values as in training set

# Build input DataFrame
input_data = {
    'Age': [age],
    'Gender': [gender],
    'College': [college]
}

input_df = pd.DataFrame(input_data)

# Encode like training
for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

# Reindex to match training feature order
input_df = input_df.reindex(columns=expected_features)

# Predict
probability = model.predict_proba(input_df)[0][1]
st.write(f"Probability of Retention: {probability * 100:.2f}%")

