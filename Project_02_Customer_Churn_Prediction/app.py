from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "customer_churn_model.joblib"

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None
    risk = None

    form_data = {}

    if request.method == "POST":

        form_data = request.form.to_dict()

        customer_data = {
            "gender": request.form.get("gender"),
            "SeniorCitizen": int(request.form.get("SeniorCitizen")),
            "Partner": request.form.get("Partner"),
            "Dependents": request.form.get("Dependents"),
            "tenure": int(request.form.get("tenure")),
            "PhoneService": request.form.get("PhoneService"),
            "MultipleLines": request.form.get("MultipleLines"),
            "InternetService": request.form.get("InternetService"),
            "OnlineSecurity": request.form.get("OnlineSecurity"),
            "OnlineBackup": request.form.get("OnlineBackup"),
            "DeviceProtection": request.form.get("DeviceProtection"),
            "TechSupport": request.form.get("TechSupport"),
            "StreamingTV": request.form.get("StreamingTV"),
            "StreamingMovies": request.form.get("StreamingMovies"),
            "Contract": request.form.get("Contract"),
            "PaperlessBilling": request.form.get("PaperlessBilling"),
            "PaymentMethod": request.form.get("PaymentMethod"),
            "MonthlyCharges": float(request.form.get("MonthlyCharges")),
            "TotalCharges": float(request.form.get("TotalCharges")),
        }

        input_df = pd.DataFrame([customer_data])

        prediction_value = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]

        probability = round(float(probabilities[1]) * 100, 2)

        if prediction_value == 1:
            prediction = "Likely to Churn"
        else:
            prediction = "Likely to Stay"

        if probability >= 70:
            risk = "High Risk"
        elif probability >= 40:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        risk=risk,
        form_data=form_data,
    )


if __name__ == "__main__":
    app.run(debug=True)