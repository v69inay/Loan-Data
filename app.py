
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from io import BytesIO
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Loan Prediction App", layout="wide")

# -- Load model & scaler --
@st.cache_resource
def load_models():
    try:
        m = joblib.load("Model.pkl")
        s = joblib.load("Scaler.pkl")
        return m, s
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

model, scaler = load_models()

def generate_pdf(data, result, metrics, analysis):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    elems = []

    # Title
    title = Paragraph("Loan Prediction Report", ParagraphStyle(
        'title', parent=styles['Heading1'], alignment=1, fontSize=18, textColor=colors.darkblue))
    elems.append(title)
    elems.append(Spacer(1, 12))
    elems.append(Paragraph(f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", styles['Normal']))
    elems.append(Spacer(1, 12))

    # Prediction
    color = colors.darkgreen if result['status']=="APPROVED" else colors.darkred
    pred_style = ParagraphStyle('pred', parent=styles['Heading2'], alignment=1, textColor=color)
    elems.append(Paragraph(f"PREDICTION: {result['status']}", pred_style))
    elems.append(Paragraph(f"Confidence: {result['confidence']:.1f}%", styles['Normal']))
    elems.append(Spacer(1, 12))

    # Applicant Info
    elems.append(Paragraph("Applicant Information", styles['Heading3']))
    info = [["Field","Value"],
            ["Name", data['name']],
            ["Gender", data['gender']],
            ["Married", data['married']],
            ["Dependents", data['dependents']],
            ["Education", data['education']],
            ["Employment", data['employment']],
            ["Property Area", data['property_area']],
            ["Credit History", data['credit_history']]]
    table = Table(info, colWidths=[2.5*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('GRID',(0,0),(-1,-1),0.5,colors.black),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
    ]))
    elems.append(table)
    elems.append(Spacer(1, 12))

    # Financial Metrics
    elems.append(Paragraph("Financial Metrics", styles['Heading3']))
    fm = [["Metric","Value"],
          ["Applicant Income", f"₹{data['applicant_income']:,}/mo"],
          ["Coapplicant Income", f"₹{data['coapplicant_income']:,}/mo"],
          ["Total Income", f"₹{metrics['total_income']:,}/mo"],
          ["Loan Amount", f"₹{data['loan_amount']*1000:,}"],
          ["Loan Term", f"{data['loan_term']} mo"],
          ["EMI", f"₹{metrics['emi']:,.0f}"],
          ["Loan/Income Ratio", f"{metrics['lir']:.1f}x"],
          ["EMI/Income Ratio", f"{metrics['eir']:.1%}"]]
    table = Table(fm, colWidths=[2.5*inch, 3*inch])
    table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.black)]))
    elems.append(table)
    elems.append(Spacer(1, 12))

    # Risk Analysis
    elems.append(Paragraph("Risk Analysis", styles['Heading3']))
    if analysis['risk']:
        elems.append(Paragraph("Risk Factors:", styles['Normal']))
        for f in analysis['risk']:
            elems.append(Paragraph(f"• {f}", styles['Normal']))
    if analysis['positive']:
        elems.append(Paragraph("Positive Factors:", styles['Normal']))
        for f in analysis['positive']:
            elems.append(Paragraph(f"• {f}", styles['Normal']))

    doc.build(elems)
    buf.seek(0)
    return buf

if model is None or scaler is None:
    st.stop()

st.title("🏦 Loan Prediction App")
st.markdown("Fill in details and predict loan approval, then download a PDF report.")

with st.form("form"):
    name = st.text_input("Full Name")
    gender = st.selectbox("Gender", ["Male","Female"])
    married = st.selectbox("Married", ["Yes","No"])
    dependents = st.selectbox("Dependents", ["0","1","2","3+"])
    education = st.selectbox("Education", ["Graduate","Not Graduate"])
    employment = st.selectbox("Employment", ["Salaried","Self-Employed"])
    property_area = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])
    credit_history = st.selectbox("Credit History", ["Good","Poor"])
    applicant_income = st.number_input("Monthly Income (₹)", 1000, 100000, 5000, 500)
    coapplicant_income = st.number_input("Coapplicant Income (₹)", 0, 50000, 0, 500)
    loan_amount = st.number_input("Loan Amount (₹ thousands)", 10, 700, 150, 10)
    loan_term = st.selectbox("Loan Term (months)", [120,180,240,300,360,480], index=4)
    submit = st.form_submit_button("Predict")

if submit:
    # Financial metrics
    total_income = applicant_income + coapplicant_income
    lir = (loan_amount*1000)/(total_income*12) if total_income>0 else 0
    r = 0.10/12; n=loan_term
    emi = (loan_amount*1000*r*(1+r)**n)/(((1+r)**n)-1) if n>0 else 0
    eir = emi/total_income if total_income>0 else 0

    # Prepare input
    df = pd.DataFrame({
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
    X = scaler.transform(df)
    pred = model.predict(X)[0]
    if hasattr(model,'predict_proba'):
        conf = max(model.predict_proba(X)[0])*100
    else:
        conf = 75.0
    status = "APPROVED" if pred==0 else "REJECTED"
    st.success(f"Loan {status}! Confidence: {conf:.1f}%") if status=="APPROVED" else st.error(f"Loan {status}. Confidence: {conf:.1f}%")

    # Risk/Positive analysis
    risk, positive = [], []
    if credit_history=="Poor": risk.append("Poor credit history")
    else: positive.append("Good credit history")
    if lir>4: risk.append("High loan/income ratio")
    elif lir8000: positive.append("High total income")
    if education=="Graduate": positive.append("Graduate education")
    if married=="Yes": positive.append("Married status")

    if risk or positive:
        st.subheader("Analysis")
        c1,c2 = st.columns(2)
        with c1:
            if risk:
                st.error("Risk Factors:")
                for r in risk: st.write(f"⚠️ {r}")
        with c2:
            if positive:
                st.success("Positive Factors:")
                for p in positive: st.write(f"✅ {p}")

    # Generate PDF
    data = {'name':name, 'gender':gender, 'married':married, 'dependents':dependents,
            'education':education, 'employment':employment,
            'applicant_income':applicant_income,'coapplicant_income':coapplicant_income,
            'loan_amount':loan_amount,'loan_term':loan_term,
            'property_area':property_area,'credit_history':credit_history}
    result = {'status':status,'confidence':conf}
    metrics = {'total_income':total_income,'emi':emi,'lir':lir,'eir':eir}
    analysis = {'risk':risk,'positive':positive}

    pdf = generate_pdf(data, result, metrics, analysis)
    st.download_button("Download PDF Report", pdf.getvalue(),
                       file_name=f"report_{name or 'applicant'}_{datetime.datetime.now():%Y%m%d_%H%M%S}.pdf",
                       mime="application/pdf")



