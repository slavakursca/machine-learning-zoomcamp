import pickle
import numpy as np

class ModelPredictor:
    def __init__(self, model_file: str = "midterm_model.bin"):
        self.model = None
        self.dv = None
        self.threshold = None
        self.model_file = model_file
        self.load_model()
    
    def load_model(self):
        """Load the model, DictVectorizer, and threshold from disk"""
        try:
            # Load Random Forest model
            with open(self.model_file, 'rb') as f_in:
                random_forest_model, random_forest_dv, random_forest_threshold = pickle.load(f_in)
                self.model = random_forest_model
                self.dv = random_forest_dv
                self.threshold = random_forest_threshold
            
        except FileNotFoundError as e:
            raise Exception(f"Model file not found: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading model components: {str(e)}")
    
    def predict(self, features: dict) -> dict:
        try:
            # Transform features using DictVectorizer
            X = self.dv.transform([features])
            
            # Get prediction probability
            y_pred_proba = self.model.predict_proba(X)[0, 1]
            
            # Apply threshold to get binary prediction
            y_pred = (y_pred_proba >= self.threshold)
            
            return {
                "prediction": bool(y_pred),
                "probability": float(y_pred_proba),
                "threshold": float(self.threshold),
                "prediction_class": int(y_pred)
            }
                
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
