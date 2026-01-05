from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_from_url
from onnx_session import ModelSession

app = FastAPI(title="Room Classifier API")

class ImageRequest(BaseModel):
    image_url: str

# Create singleton instance
model_session = ModelSession()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Deeplearning ML Model API",
        "endpoints": {
            "health": "/health",
            "predict": "/classify",
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/classify")
def predict(req: ImageRequest):
    result = predict_from_url(req.image_url, model_session)
    return result