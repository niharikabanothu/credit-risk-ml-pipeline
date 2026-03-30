"""
FastAPI REST API for Credit Risk Prediction.

Endpoints:
    GET  /             — Health check
    GET  /model-info   — Model metadata and performance metrics
    POST /predict      — Single prediction
    POST /predict-batch — Batch predictions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import os

# ---- App Setup ----
app = FastAPI(
    title="Credit Risk Prediction API",
    description="ML-powered credit risk assessment using trained classification models.",
    version="1.0.0",
)

# ---- Load Model ----
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")

model_data = None
if os.path.exists(MODEL_PATH):
    model_data = joblib.load(MODEL_PATH)


# ---- Request/Response Schemas ----
class CreditApplication(BaseModel):
    """Single credit application input."""
    checking_status: str = Field(default="A11", description="Status of existing checking account")
    duration: int = Field(default=24, description="Duration of credit in months")
    credit_history: str = Field(default="A34", description="Credit history")
    purpose: str = Field(default="A43", description="Purpose of the credit")
    credit_amount: int = Field(default=5000, description="Credit amount")
    savings_status: str = Field(default="A61", description="Savings account/bonds")
    employment: str = Field(default="A73", description="Present employment since")
    installment_commitment: int = Field(default=4, description="Installment rate (% of disposable income)")
    personal_status: str = Field(default="A93", description="Personal status and sex")
    other_parties: str = Field(default="A101", description="Other debtors/guarantors")
    residence_since: int = Field(default=2, description="Present residence since")
    property_magnitude: str = Field(default="A121", description="Property type")
    age: int = Field(default=35, description="Age in years")
    other_payment_plans: str = Field(default="A143", description="Other installment plans")
    housing: str = Field(default="A152", description="Housing type")
    existing_credits: int = Field(default=1, description="Number of existing credits at this bank")
    job: str = Field(default="A173", description="Job type")
    num_dependents: int = Field(default=1, description="Number of dependents")
    own_telephone: str = Field(default="A192", description="Telephone registered")
    foreign_worker: str = Field(default="A201", description="Foreign worker status")


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str
    model_version: str


class BatchRequest(BaseModel):
    applications: list[CreditApplication]


# ---- Helper Functions ----
def classify_risk(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    return "High"


# ---- Endpoints ----
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_data is not None,
        "api_version": "1.0.0",
    }


@app.get("/model-info")
def model_info():
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    return {
        "model_name": model_data["model_name"],
        "features": model_data["feature_names"],
        "metrics": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in model_data["metrics"].items()
            if k != "best_params"
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication):
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    try:
        # convert to dataframe
        input_dict = application.model_dump()
        df = pd.DataFrame([input_dict])

        # load feature engineer and transform
        fe_path = os.path.join(MODELS_DIR, "feature_engineer.pkl")
        if os.path.exists(fe_path):
            fe = joblib.load(fe_path)

            from src.data_loader import get_feature_types
            feat_types = {"categorical": [], "numerical": []}
            for col in df.columns:
                if df[col].dtype == "object":
                    feat_types["categorical"].append(col)
                else:
                    feat_types["numerical"].append(col)

            df = fe.create_features(df)
            df = fe.encode_categoricals(df, feat_types)

            # ensure correct feature order
            for feat in fe.feature_names:
                if feat not in df.columns:
                    df[feat] = 0
            df = df[fe.feature_names]

            df_scaled = pd.DataFrame(
                fe.scaler.transform(df), columns=fe.feature_names
            )
        else:
            df_scaled = df

        model = model_data["model"]
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1]

        return PredictionResponse(
            prediction="Bad Credit" if prediction == 1 else "Good Credit",
            probability=round(float(probability), 4),
            risk_level=classify_risk(probability),
            model_version=f"{model_data['model_name']}_v1",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-batch")
def predict_batch(batch: BatchRequest):
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for app_data in batch.applications:
        result = predict(app_data)
        results.append(result)

    return {"predictions": results, "total": len(results)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
