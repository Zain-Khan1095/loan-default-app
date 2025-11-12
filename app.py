import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -------------------------------
# Load the trained model
# -------------------------------
model = joblib.load("loan_default_model.pkl")

st.set_page_config(page_title="Loan Default Prediction", layout="centered")

st.title("🏦 Loan Default Prediction App")
st.write("Predict whether a customer will **repay** or **default** on a loan using machine learning.")

# -------------------------------
# Sidebar: user input section
# -------------------------------
st.sidebar.header("Enter Applicant Details")

age = st.sidebar.slider("Age", 18, 75, 35)
annual_income = st.sidebar.number_input("Annual Income ($)", 1000, 500000, 50000, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount ($)", 1000, 200000, 10000, step=500)
debt_to_income_ratio = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 700)
interest_rate = st.sidebar.slider("Interest Rate (%)", 1.0, 30.0, 10.0)
loan_term = st.sidebar.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
employment_status = st.sidebar.selectbox("Employment Status", 
                                         ["Employed", "Unemployed", "Self-employed", "Student", "Retired"])
education_level = st.sidebar.selectbox("Education Level", 
                                       ["High School", "Bachelor’s", "Master’s", "PhD", "Other"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", 
                                    ["Debt Consolidation", "Home Improvement", "Car", "Education", "Other"])

# -------------------------------
# Prepare data for model
# -------------------------------

# NOTE: Adjust this order/columns exactly as your model was trained
input_data = pd.DataFrame({
    'age': [age],
    'annual_income': [annual_income],
    'debt_to_income_ratio': [debt_to_income_ratio],
    'credit_score': [credit_score],
    'loan_amount': [loan_amount],
    'interest_rate': [interest_rate],
    'loan_term': [loan_term],
    'employment_status': [employment_status],
    'education_level': [education_level],
    'loan_purpose': [loan_purpose]
})

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Loan Outcome"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"✅ The model predicts this customer is **likely to repay** the loan.")
    else:
        st.error(f"⚠️ The model predicts this customer is **likely to default** on the loan.")

    st.write(f"**Probability of repayment:** {probability:.2%}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Made with ❤️ by **Zain Khan** | Data Scientist")
