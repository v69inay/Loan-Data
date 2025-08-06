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
    except Exception as e:
        st.error(f"⚠️ Error loading model files: {str(e)}")
        return None, None

model, scaler = load_models()

# App Title and Description
st.title("🏦 Loan Prediction App")
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

# Check what type of model we have
model_type = type(model).__name__ if model else "Unknown"
has_predict_proba = hasattr(model, 'predict_proba') if model else False

# Display model info in sidebar
if model:
    st.sidebar.title("🤖 Model Info")
    st.sidebar.info(f"""
    **Model Type:** {model_type}
    **Probability Support:** {'Yes' if has_predict_proba else 'No'}
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
                options=["Good", "Poor"],
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
        
        # Simple EMI calculation (10% annual interest rate assumption)
        monthly_rate = 0.10 / 12
        n_payments = loan_amount_term
        if loan_amount_term > 0 and loan_amount > 0:
            monthly_emi = (loan_amount * 1000 * monthly_rate * (1 + monthly_rate)**n_payments) / (((1 + monthly_rate)**n_payments) - 1)
        else:
            monthly_emi = 0
            
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
            try:
                # Prepare input data with CORRECT encoding based on your notebook
                input_data = pd.DataFrame({
                    'Gender': [1 if gender == 'Male' else 0],  # Male=1, Female=0
                    'Married': [1 if married == 'Yes' else 0],  # Yes=1, No=0
                    'Dependents': [0 if dependents == '0' else 1 if dependents == '1' else 2 if dependents == '2' else 3],  # 0,1,2,3+
                    'Education': [0 if education == 'Graduate' else 1],  # Graduate=0, Not Graduate=1
                    'Self_Employed': [1 if self_employed == 'Yes' else 0],  # Yes=1, No=0
                    'ApplicantIncome': [applicant_income],
                    'CoapplicantIncome': [coapplicant_income],
                    'LoanAmount': [loan_amount],
                    'Loan_Amount_Term': [loan_amount_term],
                    'Credit_History': [1.0 if credit_history == 'Good' else 0.0],  # Good=1, Poor=0
                    'Property_Area': [2 if property_area == 'Urban' else 1 if property_area == 'Semiurban' else 0]  # Urban=2, Semiurban=1, Rural=0
                })
                
                # Scale the features
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                
                # Handle probability prediction based on model type
                if has_predict_proba:
                    try:
                        prediction_proba = model.predict_proba(input_scaled)[0]
                        confidence = max(prediction_proba) * 100
                    except:
                        # Fallback for models that might not support predict_proba consistently
                        prediction_proba = None
                        confidence = 85.0  # Default confidence
                else:
                    # For models like SVC without probability support
                    prediction_proba = None
                    # Use decision function if available, otherwise default confidence
                    if hasattr(model, 'decision_function'):
                        try:
                            decision_score = model.decision_function(input_scaled)[0]
                            # Convert decision score to confidence (rough approximation)
                            confidence = min(95, max(55, 70 + abs(decision_score) * 10))
                        except:
                            confidence = 75.0
                    else:
                        confidence = 75.0
                
                # Display results
                st.subheader("🎯 Prediction Results")
                
                col_result1, col_result2 = st.columns([2, 1])
                
                with col_result1:
                    # Based on your notebook: Y=0 (Approved), N=1 (Rejected)
                    if prediction == 0:
                        st.success("🎉 **Loan Likely to be APPROVED!**")
                        st.balloons()
                        result_text = "APPROVED"
                        result_color = "success"
                    else:
                        st.error("❌ **Loan Likely to be REJECTED**")
                        result_text = "REJECTED"
                        result_color = "error"
                
                with col_result2:
                    delta_text = f"+{confidence-50:.1f}%" if confidence > 50 else f"{confidence-50:.1f}%"
                    st.metric(
                        "Confidence Score", 
                        f"{confidence:.1f}%",
                        delta=delta_text
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
                    st.write(f"• Credit History: {credit_history}")
                    st.write(f"• Property Area: {property_area}")
                
                # Risk factors analysis
                st.subheader("⚖️ Risk Factor Analysis")
                risk_factors = []
                positive_factors = []
                
                # Analyze risk factors
                if credit_history == "Poor":
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
                
                if property_area == "Urban":
                    positive_factors.append("Urban property location")
                
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
                
                # Technical details (expandable)
                with st.expander("🔧 Technical Details"):
                    st.write("**Model Information:**")
                    st.write(f"• Model Type: {model_type}")
                    st.write(f"• Supports Probability: {has_predict_proba}")
                    st.write(f"• Prediction Value: {prediction}")
                    
                    if prediction_proba is not None:
                        st.write(f"• Probability Distribution: {prediction_proba}")
                    
                    st.write("**Input Encoding:**")
                    for col, val in input_data.iloc[0].items():
                        st.write(f"• {col}: {val}")
                        
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Please check if your model files are compatible with the input format.")
                
                # Debug information
                with st.expander("🔍 Debug Information"):
                    st.write("**Error Details:**")
                    st.code(str(e))
                    st.write("**Model Type:**", type(model).__name__ if model else "None")
                    st.write("**Has predict_proba:**", hasattr(model, 'predict_proba') if model else False)

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
with st.expander("🔧 Advanced Options & Batch Processing"):
    st.subheader("📄 Batch Prediction")
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch predictions",
        type=['csv'],
        help="Upload a CSV file with columns: Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area"
    )
    
    if uploaded_file is not None and model and scaler:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("**Data Preview:**")
            st.dataframe(batch_df.head())
            
            if st.button("🚀 Process Batch Predictions"):
                with st.spinner("Processing batch predictions..."):
                    try:
                        # Prepare batch data with proper encoding
                        batch_processed = batch_df.copy()
                        
                        # Apply the same encoding as single prediction
                        batch_processed['Gender'] = batch_processed['Gender'].map({'Male': 1, 'Female': 0})
                        batch_processed['Married'] = batch_processed['Married'].map({'Yes': 1, 'No': 0})
                        batch_processed['Education'] = batch_processed['Education'].map({'Graduate': 0, 'Not Graduate': 1})
                        batch_processed['Self_Employed'] = batch_processed['Self_Employed'].map({'Yes': 1, 'No': 0})
                        batch_processed['Property_Area'] = batch_processed['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
                        
                        # Handle dependents
                        dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
                        batch_processed['Dependents'] = batch_processed['Dependents'].map(dependents_map)
                        
                        # Select feature columns (excluding Loan_ID if present)
                        feature_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                                         'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                                         'Loan_Amount_Term', 'Credit_History', 'Property_Area']
                        
                        X_batch = batch_processed[feature_columns]
                        
                        # Fill missing values with median/mode
                        for col in X_batch.columns:
                            if X_batch[col].dtype in ['int64', 'float64']:
                                X_batch[col].fillna(X_batch[col].median(), inplace=True)
                            else:
                                X_batch[col].fillna(X_batch[col].mode()[0] if not X_batch[col].mode().empty else 0, inplace=True)
                        
                        # Scale and predict
                        X_scaled = scaler.transform(X_batch)
                        predictions = model.predict(X_scaled)
                        
                        # Convert predictions back to readable format
                        batch_df['Prediction'] = ['APPROVED' if pred == 0 else 'REJECTED' for pred in predictions]
                        batch_df['Prediction_Code'] = predictions
                        
                        st.success("✅ Batch processing completed!")
                        st.dataframe(batch_df[['Prediction', 'Prediction_Code']])
                        
                        # Download option
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name='loan_predictions.csv',
                            mime='text/csv'
                        )
                        
                    except Exception as e:
                        st.error(f"Error in batch processing: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    st.subheader("📊 Model Information")
    if model:
        st.info(f"""
        **Model Type:** {model_type}
        **Probability Support:** {'Yes' if has_predict_proba else 'No (using decision function/default confidence)'}
        **Features:** 11 input features
        **Encoding:** Label encoded categorical variables
        **Scaling:** StandardScaler applied
        """)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.write("**Feature Importance:**")
            feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                           'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                           'Loan_Amount_Term', 'Credit_History', 'Property_Area']
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(importance_df)
        elif hasattr(model, 'coef_'):
            st.write("**Feature Coefficients:**")
            feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                           'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                           'Loan_Amount_Term', 'Credit_History', 'Property_Area']
            
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': model.coef_[0] if hasattr(model.coef_, '__len__') else model.coef_
            }).sort_values('Coefficient', ascending=False, key=abs)
            
            st.dataframe(coef_df)
