import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Load the trained model (provide the correct path to your .joblib model file)
model = joblib.load("D:/WorkSpace/EAS 503/Loan Approval Classification/App/FastAPI/final_rf_model.joblib")

# Initialize FastAPI app
app = FastAPI()

# Define the input data structure using Pydantic (replace with actual features)
class UserInput(BaseModel):
    person_age: float
    person_income: float
    person_emp_exp: int
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: int
    person_education: str  # Categorical feature (e.g., 'Bachelor', 'Master', etc.)
    person_home_ownership: str  # Categorical feature (e.g., 'OWN', 'RENT')
    loan_intent: str  # Categorical feature (e.g., 'EDUCATION', 'PERSONAL', etc.)
    previous_loan_defaults_on_file: str  # Categorical feature (e.g., 'Yes', 'No')

# Function to preprocess the input data (e.g., One-Hot Encoding, derived features)
def preprocess_input(data):
    # Calculate derived features (loan_percent_income and others)
    #data['loan_percent_income'] = data['loan_amnt'] / (data['person_income'] + 1e-9)  # Prevent division by zero
    #data['debt_to_income_ratio'] = data['loan_amnt'] / (data['person_income'] + 1e-9)  # Debt-to-income ratio
    #data['interest_installment_ratio'] = data['loan_int_rate'] / (data['loan_percent_income'] + 1e-9)  # Interest/Installment ratio
    
    # One-hot encoding for categorical columns
    categorical_cols = ['person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
    data_encoded = pd.get_dummies(pd.DataFrame([data]), columns=categorical_cols, drop_first=True)
    
    # Ensure input matches the model's expected columns
    required_columns = [
        'person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length', 'credit_score',
        'person_education_Bachelor', 'person_education_Doctorate', 'person_education_High School',
        'person_education_Master', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
        'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL',
        'previous_loan_defaults_on_file_Yes'
    ]
    
    # Add missing columns with 0 if not present in the processed data
    for col in required_columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0  # Add missing columns with 0
    
    # Reorder columns to match the required order
    data_encoded = data_encoded[required_columns]

    return data_encoded

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: UserInput):
    # Convert the input data to a dictionary
    input_dict = input_data.dict()

    # Preprocess and predict
    processed_data = preprocess_input(input_dict)
    prediction = model.predict(processed_data)[0]

    # Return the prediction result
    if prediction == 0:
        return {"Prediction": "Loan Rejected"}
    else:
        return {"Prediction": "Loan Approved"}

# Run the app using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
