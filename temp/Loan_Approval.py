import streamlit as st
import requests

# Streamlit App Title
st.title("Loan Approval Prediction App")
st.write("Enter the required details to check if your loan will be approved.")

# Input Fields
person_age = st.number_input("Person Age", min_value=18, max_value=100, value=30, step=1)
person_income = st.number_input("Person Income (in $)", min_value=0, value=50000, step=1000)
person_emp_exp = st.number_input("Employment Experience (in years)", min_value=0, max_value=50, value=5, step=1)
loan_amnt = st.number_input("Loan Amount (in $)", min_value=0, value=10000, step=1000)
loan_int_rate = st.number_input("Loan Interest Rate (in %)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
loan_percent_income = st.number_input("Loan Percent Income (in %)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
cb_person_cred_hist_length = st.number_input("Credit History Length (in years)", min_value=0, max_value=50, value=10, step=1)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700, step=1)
person_education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "Doctorate"])
person_home_ownership = st.selectbox("Home Ownership", ["OWN", "RENT"])
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL"])
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["Yes", "No"])

# API Call Button
if st.button("Predict Loan Approval"):
    # API Endpoint (Replace with your API URL)
    api_url = "http://127.0.0.1:8000/predict"

    # Payload for API Request
    payload = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "person_education": person_education,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }

    # Call the API
    try:
        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['Prediction']}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.write("---")
st.write("Powered by Streamlit and FastAPI.")
