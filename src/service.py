from fastapi import FastAPI
from pydantic import BaseModel
import os, pandas as pd, mlflow.pyfunc

MODEL_URI = os.getenv("MODEL_URI", "models:/credit_gam/Production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Credit GAM API", version="1.0.0")
model = mlflow.pyfunc.load_model(MODEL_URI)

class CreditRequest(BaseModel):
    # añade campos reales del dataset según tu entrenamiento
    Age: float | None = None
    CreditAmount: float | None = None
    Duration: float | None = None

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/score")
def score(req: CreditRequest):
    df = pd.DataFrame([req.dict()])
    p = float(model.predict(df)[0])
    decision = "approve" if p < 0.25 else "review"
    return {"prob_default": p, "decision": decision}
