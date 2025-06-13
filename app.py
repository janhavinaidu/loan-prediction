import streamlit as st

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

st.title("🏦 Loan Approval Prediction Dashboard")
st.markdown("This app helps understand loan approval predictions using machine learning.")

st.markdown("""
### 📊 Model Explanation

👉 [Click here to view the interactive ExplainerDashboard](https://your-explainer-dashboard-url.onrender.com)

""", unsafe_allow_html=True)

st.markdown("---")
st.info("🔧 You can deploy this app and the dashboard as two services on Render or another cloud platform.")
