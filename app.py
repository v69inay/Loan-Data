import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

st.set_page_config(page_title="Loan Predictor", layout="centered")

@st.cache_resource
def load_models():
    try:
        model = joblib.load("Model.pkl")
        scaler = joblib.load("Scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

model, scaler = load_models()
if model is None or scaler is None:
    st.stop()

st.title("🏦 Loan Prediction App")
st.markdown("Enter your details, predict your loan approval, and download a CSV report.")

with st.form("loan_form"):
    name = st.text_input("Full Name")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    employment = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    credit_history = st.selectbox("Credit History", ["Good", "Poor"])
    applicant_income = st.number_input("Monthly Income (₹)", 1000, 100000, 5000, 500)
    coapplicant_income = st.number_input("Co-applicant Income (₹)", 0, 50000, 0, 500)
    loan_amount = st.number_input("Loan Amount (₹ thousands)", 10, 700, 150, 10)
    loan_term = st.selectbox("Loan Term (months)", [120, 180, 240, 300, 360, 480], index=4)
    submitted = st.form_submit_button("🔮 Predict Loan Status")

if submitted:
    total_income = applicant_income + coapplicant_income
    lir = (loan_amount * 1000) / (total_income * 12) if total_income > 0 else 0
    r = 0.10 / 12
    n = loan_term
    emi = (loan_amount * 1000 * r * (1 + r) ** n) / (((1 + r) ** n) - 1) if n > 0 else 0
    eir = emi / total_income if total_income > 0 else 0

    # Prepare input for model
    df = pd.DataFrame({
        'Gender': [1 if gender == "Male" else 0],
        'Married': [1 if married == "Yes" else 0],
        'Dependents': [0 if dependents == "0" else 1 if dependents == "1" else 2 if dependents == "2" else 3],
        'Education': [0 if education == "Graduate" else 1],
        'Self_Employed': [1 if employment == "Self-Employed" else 0],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [1.0 if credit_history == "Good" else 0.0],
        'Property_Area': [2 if property_area == "Urban" else 1 if property_area == "Semiurban" else 0]
    })

    X = scaler.transform(df)
    pred = model.predict(X)[0]
    conf = float(model.predict_proba(X)[0].max()) * 100 if hasattr(model, 'predict_proba') else 75.0
    status = "APPROVED" if pred == 0 else "REJECTED"
    if status == "APPROVED":
        st.success(f"🎉 Loan {status}! Confidence: {conf:.1f}%")
    else:
        st.error(f"❌ Loan {status}. Confidence: {conf:.1f}%")

    # Build a report dictionary
    report_dict = {
        "Name": [name],
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Employment": [employment],
        "Property Area": [property_area],
        "Credit History": [credit_history],
        "Applicant Income": [applicant_income],
        "Co-applicant Income": [coapplicant_income],
        "Total Income": [total_income],
        "Loan Amount": [loan_amount * 1000],
        "Loan Term": [loan_term],
        "Estimated EMI": [emi],
        "Loan/Income Ratio": [lir],
        "EMI/Income Ratio": [eir],
        "Prediction": [status],
        "Confidence": [conf]
    }
    report_df = pd.DataFrame(report_dict)
    csv = report_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Report as CSV",
        data=csv,
        file_name=f"loan_report_{name or 'applicant'}_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv"
    )
