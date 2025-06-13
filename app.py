import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("loan_approval_model.joblib")

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

st.title("🏦 Loan Approval Prediction Dashboard")
st.markdown("This app helps predict loan approvals using a trained machine learning model.")

st.markdown("---")
st.header("🔍 Check Your Loan Approval")

# Input form
with st.form("loan_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
    loan_term = st.selectbox("Loan Term (in months)", [360, 180, 240, 120, 60, 84, 300, 12])
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    submit = st.form_submit_button("Predict Loan Approval")

if submit:
    # Encode inputs to match model
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    property_area = property_map[property_area]
    dependents = 3 if dependents == "3+" else int(dependents)

    features = np.array([[gender, married, dependents, education, self_employed,
                          applicant_income, coapplicant_income, loan_amount,
                          loan_term, credit_history, property_area]])

    prediction = model.predict(features)[0]
    result = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.success(f"Prediction: {result}")

st.markdown("---")
st.subheader("📊 Model Explanation")
st.markdown("👉 [Click here to view the interactive ExplainerDashboard](https://loan-prediction-8fsr.onrender.com)", unsafe_allow_html=True)

st.info("🔧 This app is built with Streamlit and deployed on Render.")
