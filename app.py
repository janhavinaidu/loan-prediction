import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained pipeline
model = joblib.load("loan_approval_model.joblib")

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")
st.title("🏦 Loan Approval Prediction Dashboard")
st.markdown("This app helps predict loan approvals using a trained machine learning model.")
st.markdown("---")
st.header("🔍 Check Your Loan Approval")

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
    # Convert and encode values
    gender = "Male" if gender == "Male" else "Female"
    married = "Yes" if married == "Yes" else "No"
    education = "Graduate" if education == "Graduate" else "Not Graduate"
    self_employed = "Yes" if self_employed == "Yes" else "No"
    property_area = property_area
    dependents = dependents

    # Feature engineering
    total_income = applicant_income + coapplicant_income
    income_to_loan_ratio = total_income / loan_amount if loan_amount != 0 else 0
    emi = loan_amount * 1000 / loan_term if loan_term != 0 else 0
    loan_per_month = emi
    dependents_int = 3 if dependents == "3+" else int(dependents)
    dependents_to_income = dependents_int / total_income if total_income != 0 else 0
    credit_history_income = credit_history * total_income

    # Binning income
    try:
        income_bins = pd.cut([total_income], bins=[0, 2500, 4000, 6000, 8000, 10000, float("inf")], labels=[0, 1, 2, 3, 4, 5])[0]
        income_bins = int(income_bins) if not pd.isna(income_bins) else 0
    except:
        income_bins = 0

    # Binning loan amount
    try:
        loan_amount_bins = pd.cut([loan_amount], bins=[0, 100, 150, 200, 250, 300, float("inf")], labels=[0, 1, 2, 3, 4, 5])[0]
        loan_amount_bins = int(loan_amount_bins) if not pd.isna(loan_amount_bins) else 0
    except:
        loan_amount_bins = 0

    family_size = 1 + dependents_int

    # Final dataframe
    data = pd.DataFrame([{
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
        'CreditHistory_Income': credit_history_income,
        'IncomeBins': income_bins,
        'LoanAmountBins': loan_amount_bins,
        'FamilySize': family_size
    }])

    # Make prediction
    prediction = model.predict(data)[0]
    result = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.success(f"Prediction: {result}")

st.markdown("---")
st.subheader("📊 Model Explanation")
st.markdown("👉 [Click here to view the interactive ExplainerDashboard](https://loan-prediction-8fsr.onrender.com)", unsafe_allow_html=True)
st.info("🔧 This app is built with Streamlit and deployed on Render.")
