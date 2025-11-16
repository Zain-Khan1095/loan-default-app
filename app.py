import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load the trained RandomForest model ---
model = joblib.load("loan_default_model.pkl")

st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("💳 Loan Default Prediction (Fixed Inputs)")

# --- Fixed input values ---
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

# --- Manual encoding for all categorical columns ---
cat_maps = {
    'gender': {'Male':0,'Female':1},
    'marital_status': {'Single':0,'Married':1},
    'education_level': {'High School':0,'Graduate':1,'Master':2,'PhD':3},
    'employment_status': {'Employed':0,'Unemployed':1,'Student':2,'Retired':3,'Self-Employed':4},
    'grade': {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6},
    'subgrade': {'A1':0,'A2':1,'A3':2,'B1':3,'B2':4,'C1':5,'C2':6,'D1':7,'D2':8,'E1':9,'E2':10}
}

for col, mapping in cat_maps.items():
    df[col] = df[col].map(mapping)

# --- Compute missing engineered features ---
df['income_to_loan'] = df['annual_income'] / df['loan_amount']
df['credit_utilization'] = df['current_balance'] / df['total_credit_limit']
df['installment_to_income'] = df['installment'] / (df['annual_income'] / 12)

# --- Force all columns to numeric to avoid isnan errors ---
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

# --- Prediction ---
threshold = 0.8  # high risk if default probability >= 80%
st.header("🎯 Prediction Result")

try:
    proba = model.predict_proba(df)[0]
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
