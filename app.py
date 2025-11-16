import streamlit as st
import pandas as pd
import joblib

# --- Load Model ---
model = joblib.load("loan_default_model.pkl")  # your saved pipeline + model

# --- Page config ---
st.set_page_config(
    page_title="💳 Loan Default Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #4b6cb7, #182848);
                padding:20px; border-radius:10px; text-align:center;">
        <h1 style="color:white;">💳 Loan Default Prediction App</h1>
        <p style="color:white;">Predict whether a borrower is likely to repay or default on a loan.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Sidebar for inputs ---
st.sidebar.header("📊 Borrower Details")

# Categorical Inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
education_level = st.sidebar.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD", "Other"])
employment_status = st.sidebar.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed", "Student", "Retired"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Debt Consolidation", "Home Improvement", "Car", "Education", "Medical", "Business", "Vacation", "Other"])
grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
subgrade = st.sidebar.selectbox("Subgrade", ["A1","A2","A3","B1","B2","C1","C2","D1","E1","F1","G1"])

# --- Numeric Inputs with sliders ---
st.markdown("### 💰 Financial & Loan Details")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 35)
    annual_income = st.slider("Annual Income ($)", 0, 500000, 50000, step=1000)
    debt_to_income_ratio = st.slider("Debt-to-Income Ratio (%)", 0.0, 100.0, 20.0, step=0.1)
    credit_score = st.slider("Credit Score", 300, 850, 700)
    loan_amount = st.slider("Loan Amount ($)", 0, 500000, 10000, step=500)

with col2:
    interest_rate = st.slider("Interest Rate (%)", 0.0, 30.0, 10.0, step=0.1)
    loan_term = st.selectbox("Loan Term", ["36 months", "60 months"])
    installment = st.slider("Monthly Installment ($)", 0, 20000, 500)
    num_of_open_accounts = st.slider("Number of Open Accounts", 0, 50, 5)
    total_credit_limit = st.slider("Total Credit Limit ($)", 0, 1000000, 50000, step=1000)

with col3:
    current_balance = st.slider("Current Balance ($)", 0, 500000, 10000, step=1000)
    delinquency_history = st.slider("Delinquency History (count)", 0, 50, 0)
    public_records = st.slider("Public Records (count)", 0, 50, 0)
    num_of_delinquencies = st.slider("Number of Delinquencies", 0, 50, 0)
    income_to_loan = st.slider("Income to Loan Ratio", 0.0, 100.0, 10.0, step=0.1)
    credit_utilization = st.slider("Credit Utilization (%)", 0.0, 100.0, 50.0, step=0.1)
    installment_to_income = st.slider("Installment to Income Ratio", 0.0, 5.0, 0.2, step=0.01)

# --- Threshold Slider ---
st.sidebar.header("⚙️ Prediction Threshold")
threshold = st.sidebar.slider(
    "Default Probability Threshold (%)",
    min_value=0,
    max_value=100,
    value=80,
    step=1
) / 100  # convert percentage to 0-1 float

# --- Prepare DataFrame for model ---
input_data = pd.DataFrame({
    'gender':[gender],
    'marital_status':[marital_status],
    'education_level':[education_level],
    'employment_status':[employment_status],
    'loan_purpose':[loan_purpose],
    'grade':[grade],
    'subgrade':[subgrade],
    'age':[age],
    'annual_income':[annual_income],
    'debt_to_income_ratio':[debt_to_income_ratio],
    'credit_score':[credit_score],
    'loan_amount':[loan_amount],
    'interest_rate':[interest_rate],
    'loan_term':[loan_term],
    'installment':[installment],
    'num_of_open_accounts':[num_of_open_accounts],
    'total_credit_limit':[total_credit_limit],
    'current_balance':[current_balance],
    'delinquency_history':[delinquency_history],
    'public_records':[public_records],
    'num_of_delinquencies':[num_of_delinquencies],
    'income_to_loan':[income_to_loan],
    'credit_utilization':[credit_utilization],
    'installment_to_income':[installment_to_income],
})

# --- Fix numeric conversions ---
input_data['loan_term'] = input_data['loan_term'].str.extract('(\d+)').astype(int)  # "36 months" → 36
input_data['subgrade'] = input_data['subgrade'].str.extract('(\d+)').astype(int)  # "A1" → 1

# --- Predict ---
st.markdown("---")
if st.button("🔍 Predict Loan Default Risk"):
    try:
        proba = model.predict_proba(input_data)[0][1]  # probability of default

        st.markdown("### 📊 Prediction Result")
        if proba >= threshold:
            st.error(f"⚠️ High Risk: Loan likely to DEFAULT. Probability: {proba:.2f}")
        else:
            st.success(f"✅ Low Risk: Loan likely to be PAID BACK. Probability: {1 - proba:.2f}")

        # Show probability bar
        st.markdown("#### 🔹 Risk Probability")
        st.progress(int(proba*100))

    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")

# --- Footer ---
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:grey;">
    Developed by <b>Zain Khan</b> | Data Scientist 💻
    </p>
    """,
    unsafe_allow_html=True
)
