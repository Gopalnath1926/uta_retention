import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model_rf.pkl", "rb") as file:
    model = pickle.load(file)

st.title("UTA Student Retention Predictor")

st.write("### Input Student Information")

# Define input fields (match features used in model)
gender = st.selectbox("Gender", ["Male", "Female"])
ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Hispanic", " "])
college = st.selectbox("First Term Enrolled College", ["Engineering", "Business", "Science", "Education", "Liberal Arts"])
age = st.slider("Age at Admission", 17, 60, 20)
gpa = st.slider("High School GPA", 0.0, 4.0, 3.0)

# Encode inputs manually for simplicity
gender_code = 1 if gender == "Male" else 0
ethnicity_code = {"White": 0, "Black": 1, "Hispanic": 2, " ": 3}[ethnicity]
college_code = {"Engineering": 0, "Business": 1, "Science": 2, "Education": 3, "Liberal Arts": 4}[college]

# Predict button
if st.button("Predict Retention Probability"):
    input_df = pd.DataFrame([[gender_code, ethnicity_code, college_code, age, gpa]],
                            columns=["Gender", "Ethnicity", "FirstTermEnrolledCollege", "Age", "HS_GPA"])
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"Probability of Retention: {probability * 100:.2f}%")
