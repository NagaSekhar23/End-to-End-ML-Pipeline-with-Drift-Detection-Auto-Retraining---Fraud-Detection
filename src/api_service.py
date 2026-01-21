from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import uvicorn
from typing import List

app = FastAPI(title="TabNet Fraud Detection API", version="1.0.0")

# Mock TabNet for demo (replace with real model later)
class MockTabNet:
    def predict_proba(self, X):
        # Simulate fraud detection
        base_proba = 0.05
        fraud_score = np.sum(np.abs(X)) / 10
        return np.array([[1-fraud_score, fraud_score]])

model = MockTabNet()

class Transaction(BaseModel):
    features: List[float]

@app.get("/")
def root():
    return {"message": "TabNet Fraud API - Ready!"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "TabNet v1"}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        features = np.array([transaction.features], dtype=np.float32)
        proba = model.predict_proba(features)[0, 1]
        prediction = 1 if proba > 0.5 else 0
        
        return {
            "prediction": int(prediction),
            "fraud_probability": float(proba),
            "risk_level": "HIGH" if proba > 0.7 else "MEDIUM" if proba > 0.3 else "LOW"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
