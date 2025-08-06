
import streamlit as st
import pandas as pd
import numpy as np 
import joblib
import warnings
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
import datetime

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="🏦 Loan Prediction App",
    page_icon="🏦",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load("Model.pkl")
        scaler = joblib.load("Scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Ensure 'Model.pkl' and 'Scaler.pkl' are present.")
        return None, None
    except Exception as e:
        st.error(f"⚠️ Error loading model files: {e}")
        return None, None

model, scaler = load_models()

# PDF report generator
def generate_pdf_report(applicant_data, prediction_result, financial_metrics, risk_analysis):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
        rightMargin=72,leftMargin=72,topMargin=72,bottomMargin=18)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                 fontSize=24, alignment=1, textColor=colors.darkblue)
    elements.append(Paragraph("🏦 LOAN PREDICTION REPORT", title_style))
    elements.append(Spacer(1, 12))
    # Timestamp
    elements.append(Paragraph(
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['Normal']))
    elements.append(Spacer(1, 12))
    # Prediction
    result_color = colors.darkgreen if prediction_result['status']=="APPROVED" else colors.darkred
    result_style = ParagraphStyle('Result', parent=styles['Heading2'],
                                  fontSize=18, alignment=1, textColor=result_color)
    elements.append(Paragraph(f"PREDICTION: {prediction_result['status']}", result_style))
    elements.append(Paragraph(f"Confidence: {prediction_result['confidence']:.1f}%", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Applicant Info Table
    elements.append(Paragraph("APPLICANT INFORMATION", styles['Heading2']))
    data = [["Field","Value"],
        ["Name", applicant_data.get("name","N/A")],
        ["Gender", applicant_data["gender"]],
        ["Marital Status", applicant_data["married"]],
        ["Dependents", str(applicant_data["dependents"])],
        ["Education", applicant_data["education"]],
        ["Employment", applicant_data["employment"]],
        ["Property Area", applicant_data["property_area"]],
        ["Credit History", applicant_data["credit_history"]]
    ]
    table = Table(data, colWidths=[2*inch,3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('GRID',(0,0),(-1,-1),1,colors.black),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige)
    ]))
    elements.append(table)
    elements.append(Spacer(1,12))

    # Financial Metrics Table
    elements.append(Paragraph("FINANCIAL INFORMATION", styles['Heading2']))
    data = [["Metric","Value"],
        ["Applicant Income", f"₹{applicant_data['applicant_income']:,}/month"],
        ["Co-applicant Income", f"₹{applicant_data['coapplicant_income']:,}/month"],
        ["Total Income", f"₹{financial_metrics['total_income']:,}/month"],
        ["Loan Amount", f"₹{applicant_data['loan_amount']*1000:,}"],
        ["Loan Term", f"{applicant_data['loan_term']} months"],
        ["Estimated EMI", f"₹{financial_metrics['monthly_emi']:,.0f}"],
        ["Loan-to-Income Ratio", f"{financial_metrics['loan_income_ratio']:.1f}x"],
        ["EMI-to-Income Ratio", f"{financial_metrics['emi_income_ratio']:.1%}"]
    ]
    table = Table(data, colWidths=[2.5*inch,2.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('GRID',(0,0),(-1,-1),1,colors.black),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige)
    ]))
    elements.append(table)
    elements.append(Spacer(1,12))

    # Risk Analysis
    elements.append(Paragraph("RISK ANALYSIS", styles['Heading2']))
    if risk_analysis['risk_factors']:
        elements.append(Paragraph("Risk Factors:",styles['Heading3']))
        for f in risk_analysis['risk_factors']:
            elements.append(Paragraph(f"• {f}", styles['Normal']))
    if risk_analysis['positive_factors']:
        elements.append(Paragraph("Positive Factors:",styles['Heading3']))
        for f in risk_analysis['positive_factors']:
            elements.append(Paragraph(f"• {f}", styles['Normal']))
    elements.append(Spacer(1,12))

    # Disclaimer
    dstyle = ParagraphStyle('Disclaimer', parent=styles['Normal'],
                             fontSize=8, textColor=colors.grey)
    elements.append(Paragraph(
        "DISCLAIMER: For educational purposes only.", dstyle))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# App UI
st.title("🏦 Simple Loan Prediction App")
st.markdown("Enter your details below to get an instant loan prediction and download a report.")

if model and scaler:
    with st.form("loan_form"):
        # Personal
        name = st.text_input("Full Name")
        gender = st.selectbox("Gender", ["Male","Female"])
        married = st.selectbox("Marital Status", ["Yes","No"])
        dependents = st.selectbox("Dependents", ["0","1","2","3+"])
        education = st.selectbox("Education", ["Graduate","Not Graduate"])
        employment = st.selectbox("Employment Type", ["Salaried","Self-Employed"])
        property_area = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])
        credit_history = st.selectbox("Credit History", ["Good","Poor"])

        # Financial
        applicant_income = st.number_input("Monthly Income (₹)",1000,100000,5000,500)
        coapplicant_income = st.number_input("Co-applicant Income (₹)",0,50000,0,500)
        loan_amount = st.number_input("Loan Amount (₹ thousands)",10,700,150,10)
        loan_term = st.selectbox("Loan Term (Months)", [120,180,240,300,360,480], index=4)

        submitted = st.form_submit_button("🔮 Predict Loan Status")
    if submitted:
        # Metrics
        total_income = applicant_income + coapplicant_income
        loan_income_ratio = (loan_amount*1000)/(total_income*12) if total_income>0 else 0
        r = 0.10/12; n=loan_term
        monthly_emi = (loan_amount*1000*r*(1+r)**n)/(((1+r)**n)-1) if n>0 else 0
        emi_income_ratio = monthly_emi/total_income if total_income>0 else 0

        # Prepare input
        df_in = pd.DataFrame({
            'Gender':[1 if gender=="Male" else 0],
            'Married':[1 if married=="Yes" else 0],
            'Dependents':[0 if dependents=="0" else 1 if dependents=="1" else 2 if dependents=="2" else 3],
            'Education':[0 if education=="Graduate" else 1],
            'Self_Employed':[1 if employment=="Self-Employed" else 0],
            'ApplicantIncome':[applicant_income],
            'CoapplicantIncome':[coapplicant_income],
            'LoanAmount':[loan_amount],
            'Loan_Amount_Term':[loan_term],
            'Credit_History':[1.0 if credit_history=="Good" else 0.0],
            'Property_Area':[2 if property_area=="Urban" else 1 if property_area=="Semiurban" else 0]
        })
        # Scale & predict
        X_scaled = scaler.transform(df_in)
        pred = model.predict(X_scaled)[0]
        if hasattr(model,'predict_proba'):
            proba = model.predict_proba(X_scaled)[0]
            conf = max(proba)*100
        else:
            conf = 75.0
        status = "APPROVED" if pred==0 else "REJECTED"
        if status=="APPROVED":
            st.success(f"🎉 Loan {status}! (Confidence: {conf:.1f}%)")
        else:
            st.error(f"❌ Loan {status} (Confidence: {conf:.1f}%)")
        # Display metrics
        col1,col2,col3 = st.columns(3)
        col1.metric("Total Income",f"₹{total_income:,}")
        col2.metric("EMI",f"₹{monthly_emi:,.0f}")
        col3.metric("EMI/Income",f"{emi_income_ratio:.1%}")
        # Risk analysis
        risk, pos = [], []
        if credit_history=="Poor": risk.append("Poor credit history")
        else: pos.append("Good credit history")
        if loan_income_ratio>4: risk.append("High loan/income ratio")
        elif loan_income_ratio8000: pos.append("High income")
        if education=="Graduate": pos.append("Graduate education")
        if married=="Yes": pos.append("Married status")
        if risk or pos:
            st.subheader("Analysis")
            c1,c2 = st.columns(2)
            with c1:
                if risk:
                    st.error("Risk Factors:")
                    for r in risk: st.write(f"⚠️ {r}")
            with c2:
                if pos:
                    st.success("Positive Factors:")
                    for p in pos: st.write(f"✅ {p}")
        # PDF report
        applicant_data = {
            'name':name,'gender':gender,'married':married,'dependents':dependents,
            'education':education,'employment':employment,
            'applicant_income':applicant_income,'coapplicant_income':coapplicant_income,
            'loan_amount':loan_amount,'loan_term':loan_term,
            'property_area':property_area,'credit_history':credit_history
        }
        pred_res = {'status':status,'confidence':conf}
        fin_m = {'total_income':total_income,'monthly_emi':monthly_emi,
                 'loan_income_ratio':loan_income_ratio,'emi_income_ratio':emi_income_ratio}
        risk_a = {'risk_factors':risk,'positive_factors':pos}
        try:
            pdf_buf = generate_pdf_report(applicant_data,pred_res,fin_m,risk_a)
            st.download_button("📥 Download PDF Report",pdf_buf.getvalue(),
                               file_name=f"report_{name or 'applicant'}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf",
                               mime="application/pdf")
        except Exception as e:
            st.error(f"PDF generation error: {e}. Install reportlab (pip install reportlab).")
else:
    st.error("Unable to load model files.")


