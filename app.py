import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, PCA, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the saved models and preprocessing objects
with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('if_model.pkl', 'rb') as f:
    if_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

# Feature categories
numerical_features = ['emp_length', 'annual_income', 'debt_to_income', 'total_credit_limit',
                      'total_credit_utilized', 'num_collections_last_12m', 'num_historical_failed_to_pay',
                      'months_since_90d_late', 'current_accounts_delinq', 'total_collection_amount_ever',
                      'current_installment_accounts', 'accounts_opened_24m', 'months_since_last_credit_inquiry',
                      'num_satisfactory_accounts', 'num_accounts_120d_past_due', 'num_accounts_30d_past_due',
                      'num_active_debit_accounts', 'total_debit_limit', 'num_total_cc_accounts',
                      'num_open_cc_accounts', 'num_cc_carrying_balance', 'num_mort_accounts',
                      'account_never_delinq_percent', 'tax_liens', 'public_record_bankrupt',
                      'loan_amount', 'term', 'interest_rate', 'installment']

categorical_features = ['loan_purpose', 'application_type', 'grade', 'sub_grade']

# For simplicity, define possible options for categorical variables
loan_purpose_options = ['Debt Consolidation', 'Credit Card', 'Other']
application_type_options = ['Individual', 'Joint']
grade_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
sub_grade_options = ['A1', 'A2', 'A3', 'A4', 'A5',
                     'B1', 'B2', 'B3', 'B4', 'B5',
                     'C1', 'C2', 'C3', 'C4', 'C5',
                     'D1', 'D2', 'D3', 'D4', 'D5',
                     'E1', 'E2', 'E3', 'E4', 'E5',
                     'F1', 'F2', 'F3', 'F4', 'F5',
                     'G1', 'G2', 'G3', 'G4', 'G5']

def main():
    st.title("Loan Application Prediction System")
    st.write("This app predicts loan approval and detects potential fraud based on your input.")

    st.sidebar.header("Input Loan Application Data")

    # Function to collect user input
    def user_input_features():
        emp_length = st.sidebar.number_input("Employee Length (years)", min_value=0, max_value=50, value=5)
        annual_income = st.sidebar.number_input("Annual Income", min_value=0, value=50000)
        debt_to_income = st.sidebar.number_input("Debt to Income Ratio (%)", min_value=0.0, max_value=100.0, value=20.0)
        total_credit_limit = st.sidebar.number_input("Total Credit Limit", min_value=0, value=20000)
        total_credit_utilized = st.sidebar.number_input("Total Credit Utilized", min_value=0, value=5000)
        num_collections_last_12m = st.sidebar.number_input("Number of Collections Last 12 Months", min_value=0, value=0)
        num_historical_failed_to_pay = st.sidebar.number_input("Number of Historical Failed to Pay", min_value=0, value=0)
        months_since_90d_late = st.sidebar.number_input("Months Since 90 Days Late", min_value=0, value=12)
        current_accounts_delinq = st.sidebar.number_input("Current Accounts Delinquent", min_value=0, value=0)
        total_collection_amount_ever = st.sidebar.number_input("Total Collection Amount Ever", min_value=0, value=0)
        current_installment_accounts = st.sidebar.number_input("Current Installment Accounts", min_value=0, value=1)
        accounts_opened_24m = st.sidebar.number_input("Accounts Opened in Last 24 Months", min_value=0, value=2)
        months_since_last_credit_inquiry = st.sidebar.number_input("Months Since Last Credit Inquiry", min_value=0, value=6)
        num_satisfactory_accounts = st.sidebar.number_input("Number of Satisfactory Accounts", min_value=0, value=5)
        num_accounts_120d_past_due = st.sidebar.number_input("Number of Accounts 120 Days Past Due", min_value=0, value=0)
        num_accounts_30d_past_due = st.sidebar.number_input("Number of Accounts 30 Days Past Due", min_value=0, value=0)
        num_active_debit_accounts = st.sidebar.number_input("Number of Active Debit Accounts", min_value=0, value=2)
        total_debit_limit = st.sidebar.number_input("Total Debit Limit", min_value=0, value=15000)
        num_total_cc_accounts = st.sidebar.number_input("Number of Total Credit Card Accounts", min_value=0, value=3)
        num_open_cc_accounts = st.sidebar.number_input("Number of Open Credit Card Accounts", min_value=0, value=1)
        num_cc_carrying_balance = st.sidebar.number_input("Number of Credit Card Carrying Balance", min_value=0, value=1)
        num_mort_accounts = st.sidebar.number_input("Number of Mortgage Accounts", min_value=0, value=1)
        account_never_delinq_percent = st.sidebar.number_input("Account Never Delinquent Percent (%)", min_value=0.0, max_value=100.0, value=100.0)
        tax_liens = st.sidebar.number_input("Tax Liens", min_value=0, value=0)
        public_record_bankrupt = st.sidebar.number_input("Public Record Bankruptcy", min_value=0, value=0)
        loan_purpose = st.sidebar.selectbox("Loan Purpose", loan_purpose_options)
        application_type = st.sidebar.selectbox("Application Type", application_type_options)
        grade = st.sidebar.selectbox("Grade", grade_options)
        sub_grade = st.sidebar.selectbox("Sub-Grade", sub_grade_options)
        loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=10000)
        term = st.sidebar.selectbox("Term", [36, 60])
        interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
        installment = st.sidebar.number_input("Installment", min_value=0, value=200)

        # Collecting all the inputs in a dictionary
        data = {'emp_length': emp_length,
                'annual_income': annual_income,
                'debt_to_income': debt_to_income,
                'total_credit_limit': total_credit_limit,
                'total_credit_utilized': total_credit_utilized,
                'num_collections_last_12m': num_collections_last_12m,
                'num_historical_failed_to_pay': num_historical_failed_to_pay,
                'months_since_90d_late': months_since_90d_late,
                'current_accounts_delinq': current_accounts_delinq,
                'total_collection_amount_ever': total_collection_amount_ever,
                'current_installment_accounts': current_installment_accounts,
                'accounts_opened_24m': accounts_opened_24m,
                'months_since_last_credit_inquiry': months_since_last_credit_inquiry,
                'num_satisfactory_accounts': num_satisfactory_accounts,
                'num_accounts_120d_past_due': num_accounts_120d_past_due,
                'num_accounts_30d_past_due': num_accounts_30d_past_due,
                'num_active_debit_accounts': num_active_debit_accounts,
                'total_debit_limit': total_debit_limit,
                'num_total_cc_accounts': num_total_cc_accounts,
                'num_open_cc_accounts': num_open_cc_accounts,
                'num_cc_carrying_balance': num_cc_carrying_balance,
                'num_mort_accounts': num_mort_accounts,
                'account_never_delinq_percent': account_never_delinq_percent,
                'tax_liens': tax_liens,
                'public_record_bankrupt': public_record_bankrupt,
                'loan_purpose': loan_purpose,
                'application_type': application_type,
                'grade': grade,
                'sub_grade': sub_grade,
                'loan_amount': loan_amount,
                'term': term,
                'interest_rate': interest_rate,
                'installment': installment}

        # Returning the user inputs as a DataFrame
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    st.subheader("User Input Features")
    st.write(input_df)

    # Preprocessing the user inputs
    input_numerical = input_df[numerical_features]
    input_categorical = input_df[categorical_features]

    input_numerical = imputer.transform(input_numerical)
    input_numerical = scaler.transform(input_numerical)
    input_numerical = pca.transform(input_numerical)

    # One-hot encoding for categorical features
    encoder = OneHotEncoder()
    input_categorical_encoded = encoder.fit_transform(input_categorical)

    # Combine numerical and categorical features
    input_preprocessed = np.concatenate([input_numerical, input_categorical_encoded.toarray()], axis=1)

    # Loan approval prediction using Logistic Regression
    loan_prediction = lr_model.predict(input_preprocessed)
    loan_prediction_prob = lr_model.predict_proba(input_preprocessed)

    # Fraud detection using Isolation Forest
    fraud_prediction = if_model.predict(input_preprocessed)

    # Display results
    st.subheader("Loan Prediction")
    loan_status = "Approved" if loan_prediction == 1 else "Rejected"
    st.write(f"Loan Status: {loan_status}")
    st.write(f"Approval Probability: {loan_prediction_prob[0][1]:.2f}")

    st.subheader("Fraud Detection")
    fraud_status = "No Fraud" if fraud_prediction == 1 else "Potential Fraud Detected"
    st.write(f"Fraud Status: {fraud_status}")

if __name__ == '__main__':
    main()
