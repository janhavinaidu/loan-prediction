import streamlit as st
import joblib
import pandas as pd

# Load trained pipeline model
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
    # Manual encoding + derived features
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    property_area = property_map[property_area]
    dependents = 3 if dependents == "3+" else int(dependents)

    total_income = applicant_income + coapplicant_income
    income_to_loan_ratio = total_income / (loan_amount + 1)
    emi = (loan_amount * 1000) / loan_term if loan_term else 0
    loan_per_month = loan_amount / (loan_term / 12) if loan_term else 0
    dependents_to_income = dependents / (total_income + 1)
    credit_income = credit_history * total_income
    income_bins = pd.cut([total_income], bins=[0, 2500, 4000, 6000, 81000], labels=[1, 2, 3, 4])[0]
    loan_bins = pd.cut([loan_amount], bins=[0, 100, 200, 700], labels=[1, 2, 3])[0]
    family_size = dependents + 1

    # Final DataFrame with expected feature columns
    input_df = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area,
        'TotalIncome': total_income,
        'IncomeToLoanRatio': income_to_loan_ratio,
        'EMI': emi,
        'LoanPerMonth': loan_per_month,
        'DependentsToIncome': dependents_to_income,
        'CreditHistory_Income': credit_income,
        'IncomeBins': int(income_bins),
        'LoanAmountBins': int(loan_bins),
        'FamilySize': family_size
    }])

    # Predict using model pipeline
    prediction = model.predict(input_df)[0]
    result = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.success(f"Prediction: {result}")

st.markdown("---")
st.subheader("📊 Model Explanation")
st.markdown("👉 [Click here to view the interactive ExplainerDashboard](https://loan-prediction-8fsr.onrender.com)", unsafe_allow_html=True)

st.info("🔧 This app is built with Streamlit and deployed on Render.")
