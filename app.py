#app.py
import gradio as gr
import joblib
import pandas as pd

# Load the trained Random Forest model
model = joblib.load("loan_prediction_model.pkl")

# Prediction function
def predict_loan(
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_amount_term,
    credit_history,
    property_area
):
    # Convert categorical values into numbers
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0

    dep_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
    dependents = dep_map[dependents]

    education = 0 if education == "Graduate" else 1
    self_employed = 1 if self_employed == "Yes" else 0
    credit_history = 1 if credit_history == "Yes" else 0

    area_map = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }
    property_area = area_map[property_area]

    # Create dataframe
    data = pd.DataFrame([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        property_area
    ]], columns=[
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area"
    ])

    prediction = model.predict(data)[0]

    if prediction == 1:
        return "✅ Loan Approved"
    else:
        return "❌ Loan Not Approved"


# Gradio Interface
demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Radio(["Yes", "No"], label="Married"),
        gr.Dropdown(["0", "1", "2", "3+"], label="Dependents"),
        gr.Radio(["Graduate", "Not Graduate"], label="Education"),
        gr.Radio(["Yes", "No"], label="Self Employed"),
        gr.Number(label="Applicant Income"),
        gr.Number(label="Coapplicant Income"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Amount Term"),
        gr.Radio(["Yes", "No"], label="Credit History"),
        gr.Dropdown(["Rural", "Semiurban", "Urban"], label="Property Area"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Loan Prediction System",
    description="""
Random Forest Machine Learning Model

**Name:** Nikita Ahlawat  
**Class:** MCA
""",
)

demo.launch()
