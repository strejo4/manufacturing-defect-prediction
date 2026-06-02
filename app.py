import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load production artifacts
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

app = FastAPI(title="Manufacturing Defect Prediction API")


class MachineData(BaseModel):
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float
    Type_L: int
    Type_M: int


@app.get("/")
def home():
    return {
        "message": "Welcome to the Manufacturing Defect Prediction API. API is up and running."
    }


@app.post("/predict")
def predict(data: MachineData):
    input_df = pd.DataFrame([data.dict()])

    # Ensure same column order used during training
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Only scale numeric columns, not one-hot encoded columns
    numeric_columns = [
        "Air_temperature_K",
        "Process_temperature_K",
        "Rotational_speed_rpm",
        "Torque_Nm",
        "Tool_wear_min"
    ]

    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "failure_probability": float(probability),
        "interpretation": "Potential machine failure" if int(prediction) == 1 else "No failure predicted"
    }