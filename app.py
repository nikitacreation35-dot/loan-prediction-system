# app.py

import os
import gradio as gr
import joblib

# Load the trained model
model = joblib.load("diabetes_prediction_model.pkl")


def predict_diabetes(pregnancies, glucose, bmi, age):
    """
    Predict diabetes using the trained Decision Tree model.
    """

    # Input must be in the same order used during training
    input_data = [[pregnancies, glucose, bmi, age]]

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        return "Prediction: High Risk of Diabetes (Positive)"
    else:
        return "Prediction: Low Risk of Diabetes (Negative)"


# Create Gradio Interface
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="BMI"),
        gr.Number(label="Age"),
    ],
    outputs=gr.Textbox(label="Assessment Result"),
    title="Diabetes Prediction System",
    description="""Enter the medical metrics to predict diabetes risk using a Decision Tree Machine Learning model.

    Name: Nikita Ahlawat
    Roll No: 252142
    Class: MCA
    """
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    interface.launch(
        server_name="0.0.0.0",
        server_port=port
    )
