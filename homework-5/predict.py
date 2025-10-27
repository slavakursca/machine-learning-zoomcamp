import pickle
from typing import Any, Dict, Literal
from pydantic import BaseModel

from fastapi import FastAPI
import uvicorn


class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool


app = FastAPI(title="customer-churn-prediction")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: Dict[str, Any]) -> PredictResponse:
    prob = predict_single(customer)

    return PredictResponse(
        churn_probability=prob,
        churn=prob >= 0.5
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)