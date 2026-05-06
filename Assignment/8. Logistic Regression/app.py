import streamlit as st
import pickle
import numpy as np

# load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("Diabetes Prediction App")

st.write("This app predicts whether a person is likely to have diabetes based on medical inputs.")

st.write("Enter patient details:")

# inputs
preg = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

# prediction
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    st.write("Model Prediction Completed")