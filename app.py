import streamlit as st
import pandas as pd
import joblib
import traceback

# --- Load the trained model ---
try:
    model = joblib.load("loan_default_model.pkl")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("💳 Loan Default Prediction (Raw Inputs Only)")

# --- Raw Example Data (no encodings) ---
data = {
    'age': [35],
    'gender': ['Male'],
    'marital_status': ['Single'],
    'education_level': ['Graduate'],
    'employment_status': ['Employed'],
    'annual_income': [60000],
    'debt_to_income_ratio': [20.0],
    'credit_score': [700],
    'loan_amount': [20000],
    'loan_purpose': ['Debt Consolidation'],
    'interest_rate': [10.0],
    'loan_term': [60],
    'installment': [500],
    'num_of_open_accounts': [5],
    'total_credit_limit': [100000],
    'current_balance': [20000],
    'delinquency_history': [0],
    'public_records': [0],
    'num_of_delinquencies': [0],
    'grade': ['A'],
    'subgrade': ['A1']
}

df = pd.DataFrame(data)

# --- Add engineered features (these should be numerics) ---
df['income_to_loan'] = df['annual_income'] / df['loan_amount']
df['credit_utilization'] = df['current_balance'] / df['total_credit_limit']
df['installment_to_income'] = df['installment'] / (df['annual_income'] / 12)

# --- Ensure feature order matches the pipeline (if saved) ---
if hasattr(model, "feature_names_in_"):
    df = df[list(model.feature_names_in_)]

st.header("🎯 Prediction Result")
threshold = 0.8  # high risk if default probability >= 80%

try:
    proba = model.predict_proba(df)[0]
    idx_default = list(model.classes_).index(1) if 1 in model.classes_ else 0
    idx_payback = list(model.classes_).index(0) if 0 in model.classes_ else 1

    prob_default = proba[idx_default]
    prob_payback = proba[idx_payback]

    st.metric("Probability of Default", f"{prob_default*100:.1f}%")
    st.metric("Probability of Payback", f"{prob_payback*100:.1f}%")

    if prob_default >= threshold:
        st.error(f"🚨 HIGH RISK: {prob_default*100:.1f}% chance of DEFAULT.")
    else:
        st.success(f"💰 SAFE BORROWER: {prob_payback*100:.1f}% chance of PAYBACK.")

except Exception as e:
    st.error(f"⚠️ Prediction Error: {e}\n{traceback.format_exc()}")
