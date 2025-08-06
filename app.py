
import streamlit as st
import pandas as pd
import numpy as np 
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="🏦 Loan Prediction App",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the models (with error handling)
@st.cache_resource
def load_models():
    try:
        model = joblib.load("Model.pkl")
        scaler = joblib.load("Scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'Model.pkl' and 'Scaler.pkl' are in the app directory.")
        return None, None

model, scaler = load_models()

# App Title and Description
st.title("🏦 Loan Default Prediction App")
st.markdown("""
<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 30px;">
    <h4 style="color: #1f77b4; margin-bottom: 10px;">📊 About This App</h4>
    <p>This application uses machine learning to predict loan approval likelihood based on applicant information. 
    Fill in the details below to get an instant prediction with confidence score.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for app information
st.sidebar.title("ℹ️ App Information")
st.sidebar.info("""
**Prediction Factors:**
- Personal Information
- Financial Profile
- Employment Status
- Credit History
- Property Details

**Model Performance:**
- Trained on 614+ loan records
- Multiple ML algorithms tested
- Real-time predictions
""")

st.sidebar.title("📈 Approval Tips")
st.sidebar.success("""
✅ Maintain good credit history  
✅ Stable income source  
✅ Lower debt-to-income ratio  
✅ Complete documentation  
✅ Consider co-applicant
""")

# Main form
if model and scaler:
    with st.form("loan_prediction_form"):
        st.subheader("📝 Enter Applicant Details")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Personal Information**")
            
            # Gender
            gender = st.selectbox(
                "Gender",
                options=["Male", "Female"],
                index=0,
                help="Select applicant's gender"
            )
            
            # Marital Status
            married = st.selectbox(
                "Marital Status",
                options=["No", "Yes"],
                index=1,
                help="Is the applicant married?"
            )
            
            # Dependents
            dependents = st.selectbox(
                "Number of Dependents",
                options=["0", "1", "2", "3+"],
                index=0,
                help="Number of family members dependent on applicant"
            )
            
            # Education
            education = st.selectbox(
                "Education Level",
                options=["Graduate", "Not Graduate"],
                index=0,
                help="Highest education qualification"
            )
        
        with col2:
            st.markdown("**💼 Employment & Property**")
            
            # Self Employed
            self_employed = st.selectbox(
                "Employment Type",
                options=["No", "Yes"],
                format_func=lambda x: "Salaried" if x == "No" else "Self-Employed",
                index=0,
                help="Employment type of the applicant"
            )
            
            # Property Area
            property_area = st.selectbox(
                "Property Area",
                options=["Urban", "Semiurban", "Rural"],
                index=0,
                help="Location type of the property"
            )
            
            # Credit History
            credit_history = st.selectbox(
                "Credit History",
                options=[1.0, 0.0],
                format_func=lambda x: "Good" if x == 1.0 else "Poor",
                index=0,
                help="Past credit payment history"
            )
        
        with col3:
            st.markdown("**💰 Financial Information**")
            
            # Applicant Income
            applicant_income = st.number_input(
                "Monthly Income (₹)",
                min_value=150,
                max_value=100000,
                value=5000,
                step=500,
                help="Applicant's monthly income in INR"
            )
            
            # Co-applicant Income
            coapplicant_income = st.number_input(
                "Co-applicant Income (₹)",
                min_value=0,
                max_value=50000,
                value=0,
                step=500,
                help="Co-applicant's monthly income (if any)"
            )
            
            # Loan Amount
            loan_amount = st.number_input(
                "Loan Amount (₹ in thousands)",
                min_value=9,
                max_value=700,
                value=150,
                step=5,
                help="Requested loan amount in thousands"
            )
            
            # Loan Amount Term
            loan_term_options = [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]
            loan_amount_term = st.selectbox(
                "Loan Term (Months)",
                options=loan_term_options,
                index=8,  # Default to 360 months
                help="Loan repayment period in months"
            )
        
        # Calculate some derived metrics for display
        st.subheader("📊 Financial Analysis")
        col_a, col_b, col_c, col_d = st.columns(4)
        
        total_income = applicant_income + coapplicant_income
        loan_income_ratio = (loan_amount * 1000) / (total_income * 12) if total_income > 0 else 0
        monthly_emi = (loan_amount * 1000 * 0.1 * (1.1)**(loan_amount_term/12)) / (((1.1)**(loan_amount_term/12)) - 1) / 12 if loan_amount_term > 0 else 0
        emi_income_ratio = monthly_emi / total_income if total_income > 0 else 0
        
        with col_a:
            st.metric("Total Monthly Income", f"₹{total_income:,.0f}")
        with col_b:
            st.metric("Loan-to-Income Ratio", f"{loan_income_ratio:.1f}x")
        with col_c:
            st.metric("Estimated EMI", f"₹{monthly_emi:,.0f}")
        with col_d:
            st.metric("EMI-to-Income Ratio", f"{emi_income_ratio:.1%}")
        
        # Warning for high risk ratios
        if loan_income_ratio > 5:
            st.warning("⚠️ High loan-to-income ratio may reduce approval chances")
        if emi_income_ratio > 0.5:
            st.warning("⚠️ High EMI-to-income ratio detected")
        
        # Submit button
        submitted = st.form_submit_button(
            "🔮 Predict Loan Status",
            type="primary",
            use_container_width=True
        )
    
    # Process prediction when form is submitted
    if submitted:
        with st.spinner("🤖 Analyzing application..."):
            # Prepare input data in the exact format the model expects
            input_data = pd.DataFrame({
                'Gender': [1 if gender == 'Male' else 0],
                'Married': [1 if married == 'Yes' else 0],
                'Dependents': [0 if dependents == '0' else 1 if dependents == '1' else 2 if dependents == '2' else 3],
                'Education': [0 if education == 'Graduate' else 1],
                'Self_Employed': [1 if self_employed == 'Yes' else 0],
                'ApplicantIncome': [applicant_income],
                'CoapplicantIncome': [coapplicant_income],
                'LoanAmount': [loan_amount],
                'Loan_Amount_Term': [loan_amount_term],
                'Credit_History': [credit_history],
                'Property_Area': [2 if property_area == 'Urban' else 1 if property_area == 'Semiurban' else 0]
            })
            
            try:
                # Scale the features
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                
                # Display results
                st.subheader("🎯 Prediction Results")
                
                col_result1, col_result2 = st.columns([2, 1])
                
                with col_result1:
                    if prediction == 0:  # Assuming 0 = Approved, 1 = Rejected based on your encoding
                        st.success("🎉 **Loan Likely to be APPROVED!**")
                        st.balloons()
                        approval_prob = prediction_proba[0] * 100
                    else:
                        st.error("❌ **Loan Likely to be REJECTED**")
                        approval_prob = prediction_proba[1] * 100
                
                with col_result2:
                    confidence = max(prediction_proba) * 100
                    st.metric(
                        "Confidence Score", 
                        f"{confidence:.1f}%",
                        delta=f"{confidence-50:.1f}%" if confidence > 50 else None
                    )
                
                # Detailed breakdown
                st.subheader("📋 Application Summary")
                
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.write("**Personal Details:**")
                    st.write(f"• Gender: {gender}")
                    st.write(f"• Marital Status: {married}")
                    st.write(f"• Dependents: {dependents}")
                    st.write(f"• Education: {education}")
                    st.write(f"• Employment: {'Self-Employed' if self_employed == 'Yes' else 'Salaried'}")
                
                with summary_col2:
                    st.write("**Financial & Property Details:**")
                    st.write(f"• Applicant Income: ₹{applicant_income:,}/month")
                    st.write(f"• Co-applicant Income: ₹{coapplicant_income:,}/month")
                    st.write(f"• Loan Amount: ₹{loan_amount * 1000:,}")
                    st.write(f"• Loan Term: {loan_amount_term} months")
                    st.write(f"• Credit History: {'Good' if credit_history == 1.0 else 'Poor'}")
                    st.write(f"• Property Area: {property_area}")
                
                # Risk factors analysis
                st.subheader("⚖️ Risk Factor Analysis")
                risk_factors = []
                positive_factors = []
                
                # Analyze risk factors
                if credit_history == 0.0:
                    risk_factors.append("Poor credit history")
                else:
                    positive_factors.append("Good credit history")
                
                if loan_income_ratio > 4:
                    risk_factors.append("High loan-to-income ratio")
                elif loan_income_ratio < 2:
                    positive_factors.append("Conservative loan amount")
                
                if total_income < 3000:
                    risk_factors.append("Low total income")
                elif total_income > 8000:
                    positive_factors.append("High total income")
                
                if education == "Graduate":
                    positive_factors.append("Graduate education")
                
                if married == "Yes":
                    positive_factors.append("Married status")
                
                # Display risk factors
                risk_col1, risk_col2 = st.columns(2)
                
                with risk_col1:
                    if risk_factors:
                        st.error("**Risk Factors:**")
                        for factor in risk_factors:
                            st.write(f"⚠️ {factor}")
                    else:
                        st.success("**No Major Risk Factors Identified!**")
                
                with risk_col2:
                    if positive_factors:
                        st.success("**Positive Factors:**")
                        for factor in positive_factors:
                            st.write(f"✅ {factor}")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Please check if your model files are compatible with the input format.")

else:
    st.error("❌ Unable to load model files. Please check if 'Model.pkl' and 'Scaler.pkl' exist in the app directory.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🏦 Loan Prediction App | Built with Streamlit & Machine Learning</p>
    <p><small>⚠️ This prediction is for educational purposes only and should not be used for actual lending decisions.</small></p>
</div>
""", unsafe_allow_html=True)

# Additional features
with st.expander("🔧 Advanced Options"):
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch predictions",
        type=['csv'],
        help="Upload a CSV file with columns matching the input format"
    )
    
    if uploaded_file is not None and model and scaler:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("**Data Preview:**")
            st.dataframe(batch_df.head())
            
            if st.button("Process Batch Predictions"):
                # Process batch predictions here
                st.info("Batch prediction feature - implement based on your specific requirements")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    st.subheader("Model Information")
    if model:
        st.info(f"**Model Type:** {type(model).__name__}")
        try:
            if hasattr(model, 'feature_importances_'):
                st.write("**Top Features:**")
                feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                               'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                               'Loan_Amount_Term', 'Credit_History', 'Property_Area']
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                st.dataframe(importance_df.head())
        except:
            st.write("Feature importance not available for this model type.")
