import os
from fastapi import FastAPI, HTTPException
import mlflow
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
import pandas as pd


app = FastAPI(title="ML Factory API")

# ================== CONFIG ==================
MLFLOW_TRACKING_URI = "http://mlflow:5000"

MODEL_NAME = "iris_model"
MODEL_ALIAS = "Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
print("MLflow URI:", os.environ.get("MLFLOW_TRACKING_URI"))
# ================== CACHE ==================
state = {
    "model": None,
    "version": None
}

# ================== LOAD MODEL ==================
def load_production_model():
    """Загружает модель по alias Production (с кешированием)"""
    try:
        alias_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        prod_version = alias_info.version

        if state["model"] is None or prod_version != state["version"]:
            print(f"Loading version {prod_version}...")
            model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            state["model"] = mlflow.pyfunc.load_model(model_uri)
            state["version"] = prod_version

        return state["model"], prod_version

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"MLflow error: {str(e)}")

# ================== INPUT ==================
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# ================== ROUTES ==================
@app.get("/")
def root():
    return {"message": "API is working!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        model, version = load_production_model()

        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)[0]

        return {
            "prediction": int(prediction),
            "model_version": version
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))