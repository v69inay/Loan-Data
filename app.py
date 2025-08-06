import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF
from io import BytesIO
import datetime
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Loan Prediction App", layout="centered")

# ----------------------
# MODEL LOADING
# ----------------------
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

if model is not None and scaler is not None:
    st.title("🏦 Loan Prediction App")
    st.markdown("Enter your details. Predict your loan status and download a PDF report.")

    # ----------------------
    # FORM
    # ----------------------
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

    # ----------------------
    # PREDICT AND DISPLAY
    # ----------------------
    if submitted:
        # Calculations
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
        if hasattr(model, 'predict_proba'):
            conf = float(model.predict_proba(X)[0].max()) * 100
        else:
            conf = 75.0  # default confidence for SVM
        status = "APPROVED" if pred == 0 else "REJECTED"
        if status == "APPROVED":
            st.success(f"🎉 Loan {status}! Confidence: {conf:.1f}%")
        else:
            st.error(f"❌ Loan {status}. Confidence: {conf:.1f}%")

        # Quick Analysis
        risk, positive = [], []
        if credit_history == "Poor": risk.append("Poor credit history")
        else: positive.append("Good credit history")
        if lir > 4: risk.append("High loan-to-income ratio")
        elif lir < 2: positive.append("Conservative loan amount")
        if total_income < 3000: risk.append("Low total income")
        elif total_income > 8000: positive.append("High total income")
        if education == "Graduate": positive.append("Graduate education")
        if married == "Yes": positive.append("Married status")

        if risk or positive:
            st.subheader("📊 Analysis")
            c1, c2 = st.columns(2)
            with c1:
                if risk:
                    st.error("Risk Factors:")
                    for r_ in risk: st.write(f"⚠️ {r_}")
            with c2:
                if positive:
                    st.success("Positive Factors:")
                    for p in positive: st.write(f"✅ {p}")

        # ----------------------
        # PDF REPORT WITH FPDF2
        # ----------------------
        def pdf_loan_report():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            pdf.set_font("Arial", "B", 18)
            pdf.set_text_color(0, 32, 96)
            pdf.cell(0, 12, "Loan Prediction Report", ln=True, align="C")

            pdf.set_font("Arial", "", 12)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 8, f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", ln=True, align="C")
            pdf.ln(7)

            # Prediction Result
            pdf.set_font("Arial", "B", 15)
            pdf.set_text_color(0, 140, 0) if status == "APPROVED" else pdf.set_text_color(220, 0, 0)
            pdf.cell(0, 12, f"{status}", ln=True, align="C")
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 8, f"Confidence: {conf:.1f}%", ln=True, align="C")
            pdf.ln(10)

            # Applicant information
            pdf.set_font("Arial", "B", 13)
            pdf.cell(0, 8, "Applicant Information", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.cell(65, 7, "Name:", 0, 0)
            pdf.cell(0, 7, str(name), 0, 1)
            pdf.cell(65, 7, "Gender:", 0, 0)
            pdf.cell(0, 7, gender, 0, 1)
            pdf.cell(65, 7, "Marital Status:", 0, 0)
            pdf.cell(0, 7, married, 0, 1)
            pdf.cell(65, 7, "Dependents:", 0, 0)
            pdf.cell(0, 7, dependents, 0, 1)
            pdf.cell(65, 7, "Education:", 0, 0)
            pdf.cell(0, 7, education, 0, 1)
            pdf.cell(65, 7, "Employment:", 0, 0)
            pdf.cell(0, 7, employment, 0, 1)
            pdf.cell(65, 7, "Property Area:", 0, 0)
            pdf.cell(0, 7, property_area, 0, 1)
            pdf.cell(65, 7, "Credit History:", 0, 0)
            pdf.cell(0, 7, credit_history, 0, 1)
            pdf.ln(4)

            # Financial details
            pdf.set_font("Arial", "B", 13)
            pdf.cell(0, 8, "Financial Metrics", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.cell(65, 7, "Applicant Income:", 0, 0)
            pdf.cell(0, 7, f"₹{applicant_income:,}/month", 0, 1)
            pdf.cell(65, 7, "Co-applicant Income:", 0, 0)
            pdf.cell(0, 7, f"₹{coapplicant_income:,}/month", 0, 1)
            pdf.cell(65, 7, "Total Income:", 0, 0)
            pdf.cell(0, 7, f"₹{total_income:,}/month", 0, 1)
            pdf.cell(65, 7, "Loan Amount:", 0, 0)
            pdf.cell(0, 7, f"₹{loan_amount*1000:,}", 0, 1)
            pdf.cell(65, 7, "Loan Term:", 0, 0)
            pdf.cell(0, 7, f"{loan_term} months", 0, 1)
            pdf.cell(65, 7, "Estimated EMI:", 0, 0)
            pdf.cell(0, 7, f"₹{emi:,.0f}", 0, 1)
            pdf.cell(65, 7, "Loan-Income Ratio:", 0, 0)
            pdf.cell(0, 7, f"{lir:.1f}x", 0, 1)
            pdf.cell(65, 7, "EMI-Income Ratio:", 0, 0)
            pdf.cell(0, 7, f"{eir:.1%}", 0, 1)
            pdf.ln(4)

            # Analysis
            pdf.set_font("Arial", "B", 13)
            pdf.cell(0, 8, "Risk & Positive Factors", ln=True)
            pdf.set_font("Arial", "", 12)
            if risk:
                pdf.set_text_color(220, 0, 0)
                pdf.cell(0, 7, "Risk Factors:", ln=1)
                pdf.set_text_color(0, 0, 0)
                for r_ in risk:
                    pdf.cell(5)
                    pdf.cell(0, 7, f"- {r_}", ln=1)
            if positive:
                pdf.set_text_color(0, 140, 0)
                pdf.cell(0, 7, "Positive Factors:", ln=1)
                pdf.set_text_color(0, 0, 0)
                for p in positive:
                    pdf.cell(5)
                    pdf.cell(0, 7, f"- {p}", ln=1)
            pdf.ln(3)

            # Disclaimer
            pdf.set_font("Arial", "I", 8)
            pdf.set_text_color(127,127,127)
            pdf.multi_cell(0,6, "This prediction is for demonstration purposes only. Do not use for real-life loan decisions.\n© {}".format(datetime.datetime.now().year), 0, "C")
            pdf.set_text_color(0,0,0)

            # Output as bytes
            out = BytesIO()
            pdf.output(out)
            out.seek(0)
            return out

        # Download button
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_loan_report(),
            file_name=f"loan_report_{name or 'applicant'}_{datetime.datetime.now():%Y%m%d_%H%M%S}.pdf",
            mime="application/pdf"
        )

else:
    st.error("Could not load model or scaler. Prediction not available.")
