import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load model and expected features
with open("model_rf.pkl", "rb") as file:
    model, feature_names = pickle.load(file)

# Assume encoders used during training
gender_encoder = LabelEncoder().fit(['Female', 'Male'])
college_encoder = LabelEncoder().fit(['Engineering', 'Business', 'Science'])  # replace with actual

# Streamlit inputs
st.title("Retention Predictor")
age = st.slider("Age", 17, 60, 18)
gender = st.selectbox("Gender", ['Female', 'Male'])
college = st.selectbox("First Term College", ['Engineering', 'Business', 'Science'])  # match training

# DataFrame
input_data = {
    'Age': [age],
    'Gender': gender_encoder.transform([gender]),
    'FirstTermEnrolledCollege': college_encoder.transform([college])
}
input_df = pd.DataFrame(input_data)

# Reorder columns
input_df = input_df.reindex(columns=feature_names)

# Prediction
prob = model.predict_proba(input_df)[0][1]
st.success(f"Predicted probability of retention: {prob*100:.2f}%")
