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
st.title("💳 Loan Default Prediction (Fixed Inputs)")

# --- Raw data ---
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

# --- ONLY encode grade and subgrade ---
grade_map = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6}
subgrade_map = {
    'A1':0,'A2':1,'A3':2,'A4':3,'A5':4,
    'B1':5,'B2':6,'B3':7,'B4':8,'B5':9,
    'C1':10,'C2':11,'C3':12,'C4':13,'C5':14,
    'D1':15,'D2':16,'D3':17,'D4':18,'D5':19,
    'E1':20,'E2':21,'E3':22,'E4':23,'E5':24,
    'F1':25,'F2':26,'F3':27,'F4':28,'F5':29,
    'G1':30,'G2':31,'G3':32,'G4':33,'G5':34
}

df['grade'] = df['grade'].map(grade_map)
df['subgrade'] = df['subgrade'].map(subgrade_map)

# --- Add engineered features ---
df['income_to_loan'] = df['annual_income'] / df['loan_amount']
df['credit_utilization'] = df['current_balance'] / df['total_credit_limit']
df['installment_to_income'] = df['installment'] / (df['annual_income'] / 12)

# -- Use correct column order if available --
if hasattr(model, "feature_names_in_"):
    df = df[list(model.feature_names_in_)]

st.header("🎯 Prediction Result")
threshold = 0.8

try:
    proba = model.predict_proba(df)[0]
    # Most models: class 1 = default, 0 = payback. This may differ; adjust as needed!
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
