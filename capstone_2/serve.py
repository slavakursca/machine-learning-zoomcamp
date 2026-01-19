"""
FastAPI Inference Service for Astana Apartment Price Prediction
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pickle
import pandas as pd
import numpy as np
from typing import Literal, Optional

from starlette.responses import HTMLResponse

# ============================================================================
# LOAD MODEL
# ============================================================================
try:
    with open("astana_price_model.pkl", "rb") as f:
        model_package = pickle.load(f)

    model = model_package["model"]
    encoder = model_package["encoder"]
    numeric_features = model_package["feature_names"]["numeric"]
    categorical_features = model_package["feature_names"]["categorical"]

    print("✅ Model loaded successfully!")
    print(f"   - Numeric features: {len(numeric_features)}")
    print(f"   - Categorical features: {len(categorical_features)}")
    print(f"   - Test RMSE: ${model_package['metrics']['test']['rmse']:,.2f}")
    print(f"   - Test R²: {model_package['metrics']['test']['r2']:.4f}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Astana Apartment Price Prediction API",
    description="ML-powered real estate price prediction for Astana, Kazakhstan",
    version="1.0.0",
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================
class ApartmentFeatures(BaseModel):
    rooms: int = Field(..., ge=1, le=10)
    area: float = Field(..., gt=0, le=150)
    living_area: float = Field(..., gt=0)
    kitchen_area: float = Field(..., gt=0)
    floor: int = Field(..., ge=-1)
    total_floors: int = Field(..., ge=-1)
    ceiling_height: float = Field(..., gt=0, le=5)
    building_age: int = Field(..., ge=0)

    district: Literal[
        "Yesil", "Almaty", "Saryarka", "Nura", "Baikonur", "Saryshyk"
    ]
    latitude: float = Field(..., ge=51.0, le=51.3)
    longitude: float = Field(..., ge=71.2, le=71.6)

    house_type: Literal["monolithic", "brick", "panel", "other", "unknown"]
    condition: Literal["new", "good", "rough", "needs_repair", "unknown"]

    parking: Literal["private_parking", "public_parking", "garage", "no_parking"]
    furniture: Literal[
        "fully_furnished", "partially_furnished", "unfurnished", "unknown"
    ]
    bathroom_type: Literal["combined", "separate", "multiple", "unknown"]
    bathroom_count: int = Field(..., ge=0, le=5)
    balcony_type: Literal["balcony", "loggia", "multiple", "unknown", "none"]

    security_high: int = Field(..., ge=0, le=1)
    wooden_floor: int = Field(..., ge=0, le=1)
    has_window_grills: int = Field(..., ge=0, le=1)

    floor_relative: float = Field(..., ge=0, le=1)
    living_ratio: float = Field(..., ge=0, le=1)
    kitchen_ratio: float = Field(..., ge=0, le=1)

    @validator("living_area")
    def validate_living_area(cls, v, values):
        if "area" in values and v > values["area"] * 0.95:
            raise ValueError(
                f"living_area ({v}) cannot exceed 95% of total area ({values['area']})"
            )
        return v

    @validator("kitchen_area")
    def validate_kitchen_area(cls, v, values):
        if "area" in values and v > values["area"] * 0.35:
            raise ValueError(
                f"kitchen_area ({v}) cannot exceed 35% of total area ({values['area']})"
            )
        return v

    @validator("floor_relative")
    def validate_floor_relative(cls, v, values):
        if "floor" in values and "total_floors" in values:
            if values["floor"] > 0 and values["total_floors"] > 0:
                expected = values["floor"] / values["total_floors"]
                if abs(v - expected) > 0.01:
                    raise ValueError(f"floor_relative should be {expected:.3f}, got {v}")
        return v


class PredictionResponse(BaseModel):
    predicted_price_usd: float
    price_per_m2: float
    confidence_interval_95: dict
    price_category: str
    input_features: dict


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def prepare_features(data: ApartmentFeatures) -> np.ndarray:
    """Convert input data to model-ready features"""

    # Numeric features
    numeric_values = [[getattr(data, feat) for feat in numeric_features]]
    numeric_arr = np.array(numeric_values)

    # Categorical features
    categorical_dict = {feat: [getattr(data, feat)] for feat in categorical_features}
    categorical_df = pd.DataFrame(categorical_dict)

    # Encode categorical
    categorical_encoded = encoder.transform(categorical_df)
    if hasattr(categorical_encoded, "toarray"):
        categorical_encoded = categorical_encoded.toarray()

    # Combine numeric + categorical
    X = np.concatenate([numeric_arr, categorical_encoded], axis=1)
    return X


def categorize_price(price: float) -> str:
    if price < 50000:
        return "budget"
    elif price < 80000:
        return "mid-range"
    elif price < 110000:
        return "premium"
    else:
        return "luxury"


def calculate_confidence_interval(prediction_log: float, margin: float = 0.095) -> dict:
    prediction = np.expm1(prediction_log)
    lower = prediction * (1 - margin)
    upper = prediction * (1 + margin)
    return {
        "lower": round(float(lower), 2),  # ✅ ensure Python float
        "upper": round(float(upper), 2),
        "margin_percent": round(float(margin * 100), 1),
    }


# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/")
async def root():
    return {
        "message": "Astana Apartment Price Prediction API",
        "version": "1.0.0",
        "model": "XGBoost",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "form": "/form",
            "model/info": "/model/info",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "test_metrics": {
            "rmse": f"${model_package['metrics']['test']['rmse']:,.2f}",
            "r2": f"{model_package['metrics']['test']['r2']:.4f}",
            "mape": f"{model_package['metrics']['test']['mape']:.2f}%",
        },
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(apartment: ApartmentFeatures):
    try:
        X = prepare_features(apartment)

        # Make prediction
        prediction_log = model.predict(X)[0]
        predicted_price = float(np.expm1(prediction_log))  # ✅ convert NumPy scalar to float

        price_per_m2 = float(predicted_price / apartment.area)
        confidence_interval = calculate_confidence_interval(prediction_log)
        price_category = categorize_price(predicted_price)

        return PredictionResponse(
            predicted_price_usd=round(predicted_price, 2),
            price_per_m2=round(price_per_m2, 2),
            confidence_interval_95=confidence_interval,
            price_category=price_category,
            input_features=apartment.dict(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model/info")
async def model_info():
    return {
        "model_type": model_package["metadata"]["model_type"],
        "optimization_metric": model_package["metadata"]["optimization_metric"],
        "training_samples": model_package["metadata"]["training_samples"],
        "features_count": model_package["metadata"]["features_count"],
        "performance": {
            "train": {
                "rmse": f"${model_package['metrics']['train']['rmse']:,.2f}",
                "r2": f"{model_package['metrics']['train']['r2']:.4f}",
                "mape": f"{model_package['metrics']['train']['mape']:.2f}%",
            },
            "validation": {
                "rmse": f"${model_package['metrics']['validation']['rmse']:,.2f}",
                "r2": f"{model_package['metrics']['validation']['r2']:.4f}",
                "mape": f"{model_package['metrics']['validation']['mape']:.2f}%",
            },
            "test": {
                "rmse": f"${model_package['metrics']['test']['rmse']:,.2f}",
                "r2": f"{model_package['metrics']['test']['r2']:.4f}",
                "mape": f"{model_package['metrics']['test']['mape']:.2f}%",
            },
        },
        "hyperparameters": model_package["hyperparameters"],
    }


@app.get("/form", response_class=HTMLResponse)
async def form():
    """Render prediction form"""
    html_file = Path("form.html")
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="Form template not found")

    return HTMLResponse(content=html_file.read_text())


# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9696)
