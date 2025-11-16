import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("loan_default_model.pkl")

# Streamlit page setup
st.set_page_config(page_title="Loan Risk Predictor 💰", page_icon="💳", layout="wide")

# Custom CSS for modern look
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    h1 {
        text-align: center;
        color: #1a73e8;
        font-weight: 700;
    }
    .stButton button {
        width: 100%;
        background-color: #1a73e8 !important;
        color: white !important;
        border-radius: 10px;
        height: 3em;
        font-size: 16px;
        font-weight: 600;
    }
    .stSuccess {
        background-color: #e6f4ea !important;
    }
    .stError {
        background-color: #fce8e6 !important;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("💳 Loan Default Prediction Dashboard")
st.write("Use this app to predict whether a **loan will be paid back or defaulted** based on borrower information.")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Model Configuration")
threshold = st.sidebar.slider("Set Default Risk Threshold", 0.5, 0.95, 0.8, 0.05)
st.sidebar.info(f"The model flags a borrower as 'High Risk' only if default probability ≥ **{threshold:.2f}**")

# --- Input Section ---
st.header("📋 Borrower Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education_level = st.selectbox("Education Level", ["High School", "Graduate", "Master", "PhD"])
    employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Student", "Retired", "Self-Employed"])

with col2:
    annual_income = st.number_input("Annual Income ($)", 1000, 1000000, 60000)
    debt_to_income_ratio = st.number_input("Debt to Income Ratio (%)", 0.0, 100.0, 20.0)
    credit_score = st.number_input("Credit Score", 300, 900, 700)
    loan_amount = st.number_input("Loan Amount ($)", 1000, 1000000, 20000)
    loan_purpose = st.selectbox("Loan Purpose", ["Home Improvement", "Debt Consolidation", "Education", "Business", "Medical", "Car"])

with col3:
    interest_rate = st.number_input("Interest Rate (%)", 1.0, 30.0, 10.0)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    installment = st.number_input("Installment ($ per month)", 100, 10000, 500)
    total_credit_limit = st.number_input("Total Credit Limit ($)", 1000, 1000000, 100000)
    current_balance = st.number_input("Current Balance ($)", 0, 1000000, 20000)

# Additional Inputs
colA, colB, colC = st.columns(3)
with colA:
    num_of_open_accounts = st.number_input("Open Accounts", 0, 30, 5)
with colB:
    delinquency_history = st.number_input("Delinquency History", 0, 10, 0)
with colC:
    public_records = st.number_input("Public Records", 0, 10, 0)

colX, colY, colZ = st.columns(3)
with colX:
    num_of_delinquencies = st.number_input("Number of Delinquencies", 0, 10, 0)
with colY:
    grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
with colZ:
    subgrade = st.selectbox("Subgrade", ["A1","A2","A3","B1","B2","C1","C2","D1","D2","E1","E2"])

# --- Derived Features ---
income_to_loan = annual_income / loan_amount if loan_amount != 0 else 0
installment_to_income = installment / (annual_income / 12) if annual_income != 0 else 0
credit_utilization = current_balance / total_credit_limit if total_credit_limit != 0 else 0

# --- Prepare DataFrame ---
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'marital_status': [marital_status],
    'education_level': [education_level],
    'annual_income': [annual_income],
    'employment_status': [employment_status],
    'debt_to_income_ratio': [debt_to_income_ratio],
    'credit_score': [credit_score],
    'loan_amount': [loan_amount],
    'loan_purpose': [loan_purpose],
    'interest_rate': [interest_rate],
    'loan_term': [loan_term],
    'installment': [installment],
    'num_of_open_accounts': [num_of_open_accounts],
    'total_credit_limit': [total_credit_limit],
    'current_balance': [current_balance],
    'delinquency_history': [delinquency_history],
    'public_records': [public_records],
    'num_of_delinquencies': [num_of_delinquencies],
    'income_to_loan': [income_to_loan],
    'credit_utilization': [credit_utilization],
    'installment_to_income': [installment_to_income],
    'grade': [grade],
    'subgrade': [subgrade]
})

# --- Prediction Section ---
st.markdown("### 🎯 Prediction Result")

if st.button("Predict Loan Outcome 💡"):
    try:
        proba = model.predict_proba(input_data)[0]
        prob_default = proba[0]
        prob_payback = proba[1]

        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="Probability of Default", value=f"{prob_default*100:.2f}%")
        st.progress(float(prob_default))

        st.metric(label="Probability of Payback", value=f"{prob_payback*100:.2f}%")
        st.progress(float(prob_payback))
        st.markdown("</div>", unsafe_allow_html=True)

        if prob_default >= threshold:
            st.error(f"🚨 HIGH RISK: {prob_default*100:.1f}% chance of DEFAULT.")
        else:
            st.success(f"💰 SAFE BORROWER: {prob_payback*100:.1f}% chance of PAYBACK.")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        st.write("Make sure all input features are filled correctly and match the model schema.")
