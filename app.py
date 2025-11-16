import streamlit as st
import pandas as pd
import joblib

# Load trained model pipeline
model = joblib.load("loan_default_model.pkl")

st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("💳 Loan Default Prediction")

st.write("This demo uses fixed input values to predict loan payback.")

# --- Fixed input data ---
age = 35
annual_income = 60000
monthly_income = annual_income / 12
loan_amount = 20000
installment = 500
total_credit_limit = 100000
current_balance = 20000

data = {
    'age': [age],
    'gender': ['Male'],
    'marital_status': ['Single'],
    'education_level': ['Graduate'],
    'employment_status': ['Employed'],
    'annual_income': [annual_income],
    'debt_to_income_ratio': [20.0],
    'credit_score': [700],
    'loan_amount': [loan_amount],
    'loan_purpose': ['Debt Consolidation'],
    'interest_rate': [10.0],
    'loan_term': [60],
    'installment': [installment],
    'num_of_open_accounts': [5],
    'total_credit_limit': [total_credit_limit],
    'current_balance': [current_balance],
    'delinquency_history': [0],
    'public_records': [0],
    'num_of_delinquencies': [0],
    'grade': ['A'],
    'subgrade': ['A1']
}

input_df = pd.DataFrame(data)

# --- Compute missing engineered features ---
input_df['income_to_loan'] = input_df['annual_income'] / input_df['loan_amount']
input_df['credit_utilization'] = input_df['current_balance'] / input_df['total_credit_limit']
input_df['installment_to_income'] = input_df['installment'] / (input_df['annual_income'] / 12)

# --- Prediction ---
threshold = 0.8  # default threshold for high risk
st.header("🎯 Prediction Result")

try:
    proba = model.predict_proba(input_df)[0]
    prob_default = proba[0]
    prob_payback = proba[1]

    st.metric("Probability of Default", f"{prob_default*100:.1f}%")
    st.metric("Probability of Payback", f"{prob_payback*100:.1f}%")

    if prob_default >= threshold:
        st.error(f"🚨 HIGH RISK: {prob_default*100:.1f}% chance of DEFAULT.")
    else:
        st.success(f"💰 SAFE BORROWER: {prob_payback*100:.1f}% chance of PAYBACK.")

except Exception as e:
    st.error(f"⚠️ Prediction Error: {e}")
