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
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1)
    loan_term = st.selectbox("Loan Term (in months)", [360, 180, 240, 120, 60, 84, 300, 12])
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    submit = st.form_submit_button("Predict Loan Approval")

if submit:
    # Encode categorical features
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    property_area = property_map[property_area]
    dependents = 3 if dependents == "3+" else int(dependents)

    # Derived features
    total_income = applicant_income + coapplicant_income
    income_to_loan_ratio = total_income / (loan_amount * 1000)
    emi = (loan_amount * 1000) / loan_term if loan_term else 0
    loan_per_month = loan_amount * 1000 / loan_term if loan_term else 0
    dependents_to_income = dependents / total_income if total_income else 0
    credit_history_income = credit_history * total_income

    # Binning features (you must ensure your model was trained with these bins)
    if total_income < 2500:
        income_bin = 0
    elif total_income < 4000:
        income_bin = 1
    elif total_income < 6000:
        income_bin = 2
    else:
        income_bin = 3

    if loan_amount < 100:
        loan_bin = 0
    elif loan_amount < 200:
        loan_bin = 1
    else:
        loan_bin = 2

    family_size = dependents + (1 if married == 1 else 0) + 1  # Self + dependents + spouse (if married)

    # Final 20 feature vector
    features = np.array([[gender, married, dependents, education, self_employed,
                          property_area, applicant_income, coapplicant_income, loan_amount,
                          loan_term, credit_history, total_income, income_to_loan_ratio, emi,
                          loan_per_month, dependents_to_income, credit_history_income,
                          income_bin, loan_bin, family_size]])

    # Prediction
    prediction = model.predict(features)[0]
    result = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.success(f"Prediction: {result}")

# Footer
st.markdown("---")
st.subheader("📊 Model Explanation")
st.markdown("👉 [Click here to view the interactive ExplainerDashboard](https://loan-prediction-8fsr.onrender.com)", unsafe_allow_html=True)
st.info("🔧 This app is built with Streamlit and deployed on Render.")
