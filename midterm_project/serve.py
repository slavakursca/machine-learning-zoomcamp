from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from predict import ModelPredictor
import uvicorn
from typing import Literal

# Initialize FastAPI app
app = FastAPI(title="ML Model API", version="1.0.0")

# Initialize predictor
predictor = ModelPredictor("midterm_model.bin")

# Define request model with validation
class PredictionRequest(BaseModel):
    person_age: int = Field(..., ge=20, le=144, description="Age of applicant (20-144)")
    person_gender: Literal["male", "female"] = Field(..., description="Gender of applicant")
    person_education: Literal["High School", "Bachelor", "Master", "PhD", "Associate"] = Field(
        ..., description="Highest education level"
    )
    person_income: float = Field(..., gt=0, description="Annual income (must be positive)")
    person_emp_exp: int = Field(..., ge=0, description="Years of employment experience")
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"] = Field(
        ..., description="Home ownership status"
    )
    loan_amnt: float = Field(..., gt=0, description="Loan amount requested (must be positive)")
    loan_intent: Literal["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"] = Field(
        ..., description="Purpose of the loan"
    )
    loan_int_rate: float = Field(..., ge=0, le=100, description="Interest rate (0-100%)")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Loan to income ratio (0-1)")
    cb_person_cred_hist_length: int = Field(..., ge=0, description="Credit history length in years")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    previous_loan_defaults_on_file: Literal[0, 1] = Field(
        ..., description="Prior loan defaults (0=No, 1=Yes)"
    )
    
    @field_validator('person_gender')
    @classmethod
    def validate_gender(cls, v):
        v = v.lower()
        if v not in ['male', 'female']:
            raise ValueError('person_gender must be "male" or "female"')
        return v
    
    @field_validator('person_education')
    @classmethod
    def validate_education(cls, v):
        valid_education = ['High School', 'Bachelor', 'Master', 'PhD', 'Associate']
        if v not in valid_education:
            raise ValueError(f'person_education must be one of: {", ".join(valid_education)}')
        return v
    
    @field_validator('person_home_ownership')
    @classmethod
    def validate_home_ownership(cls, v):
        v = v.upper()
        valid_ownership = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
        if v not in valid_ownership:
            raise ValueError(f'person_home_ownership must be one of: {", ".join(valid_ownership)}')
        return v
    
    @field_validator('loan_intent')
    @classmethod
    def validate_loan_intent(cls, v):
        v = v.upper()
        valid_intents = ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']
        if v not in valid_intents:
            raise ValueError(f'loan_intent must be one of: {", ".join(valid_intents)}')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "person_age": 35,
                "person_gender": "male",
                "person_education": "Bachelor",
                "person_income": 75000.0,
                "person_emp_exp": 10,
                "person_home_ownership": "RENT",
                "loan_amnt": 15000.0,
                "loan_intent": "EDUCATION",
                "loan_int_rate": 5.5,
                "loan_percent_income": 0.2,
                "cb_person_cred_hist_length": 8,
                "credit_score": 720,
                "previous_loan_defaults_on_file": 0
            }
        }

# Define response model
class PredictionResponse(BaseModel):
    prediction: bool
    probability: float
    threshold: float
    prediction_class: int

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Model API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert request to dictionary
        features = request.model_dump()
        
        # Get prediction
        result = predictor.predict(features)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
