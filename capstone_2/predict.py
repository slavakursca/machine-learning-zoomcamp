"""
Prediction Script for Astana Apartment Price Model

This script loads the trained model and provides predictions for new apartments.
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, Union


class ApartmentPricePredictor:
    """Predictor class for apartment price predictions"""

    def __init__(self, model_file: str = "astana_price_model.pkl"):
        """
        Initialize the predictor

        Args:
            model_file: Path to the saved model package
        """
        self.model = None
        self.encoder = None
        self.numeric_features = None
        self.categorical_features = None
        self.model_file = model_file
        self.metadata = None

        self.load_model()

    def load_model(self):
        """Load the model package from disk"""
        try:
            with open(self.model_file, 'rb') as f:
                package = pickle.load(f)

            self.model = package['model']
            self.encoder = package['encoder']
            self.numeric_features = package['feature_names']['numeric']
            self.categorical_features = package['feature_names']['categorical']
            self.metadata = package.get('metadata', {})

            print("✅ Model loaded successfully!")
            print(f"   Model type: {self.metadata.get('model_type', 'Unknown')}")
            print(f"   Features: {self.metadata.get('features_count', 'Unknown')}")

        except FileNotFoundError:
            raise Exception(f"Model file not found: {self.model_file}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def predict(self, features: Dict[str, Union[int, float, str]]) -> Dict:
        """
        Predict apartment price

        Args:
            features: Dictionary containing apartment features

        Returns:
            Dictionary with prediction results

        Example:
            features = {
                'rooms': 2,
                'area': 65.4,
                'living_area': 46.6,
                'kitchen_area': 10.6,
                'floor': 7,
                'total_floors': 12,
                'building_age': 1,
                'ceiling_height': 3.0,
                'latitude': 51.1694,
                'longitude': 71.4491,
                'has_window_grills': 0,
                'security_high': 1,
                'bathroom_count': 2,
                'wooden_floor': 0,
                'floor_relative': 0.583333,
                'living_ratio': 0.712992,
                'kitchen_ratio': 0.162080,
                'house_type': 'monolithic',
                'parking': 'public_parking',
                'furniture': 'unfurnished',
                'district': 'Yesil',
                'bathroom_type': 'multiple',
                'balcony_type': 'multiple',
                'condition': 'good'
            }
        """
        try:
            # Validate required features
            self._validate_features(features)

            # Prepare numeric features
            numeric_values = [features.get(feat, 0) for feat in self.numeric_features]

            # Prepare categorical features for encoding
            categorical_dict = {feat: features.get(feat, 'unknown')
                                for feat in self.categorical_features}
            categorical_df = pd.DataFrame([categorical_dict])

            # Encode categorical features
            categorical_encoded = self.encoder.transform(categorical_df)

            # Combine features
            X = np.concatenate([numeric_values, categorical_encoded[0]])
            X = X.reshape(1, -1)

            # Get prediction (in log space)
            y_pred_log = self.model.predict(X)[0]

            # Transform back to original scale
            predicted_price = np.expm1(y_pred_log)

            # Calculate price per m²
            area = features.get('area', 1)
            price_per_m2 = predicted_price / area if area > 0 else 0

            return {
                "predicted_price_usd": float(predicted_price),
                "price_per_m2": float(price_per_m2),
                "confidence_interval_95": self._get_confidence_interval(predicted_price),
                "price_category": self._categorize_price(predicted_price),
                "area": float(area)
            }

        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")

    def predict_batch(self, features_list: list) -> list:
        """
        Predict prices for multiple apartments

        Args:
            features_list: List of feature dictionaries

        Returns:
            List of prediction results
        """
        results = []
        for features in features_list:
            try:
                result = self.predict(features)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        return results

    def _validate_features(self, features: Dict):
        """Validate that required features are present"""
        required_features = set(self.numeric_features + self.categorical_features)
        provided_features = set(features.keys())

        missing = required_features - provided_features
        if missing:
            print(f"⚠️  Warning: Missing features will use defaults: {missing}")

    def _get_confidence_interval(self, price: float, margin: float = 0.095) -> Dict:
        """
        Calculate approximate confidence interval based on model MAPE

        Args:
            price: Predicted price
            margin: Error margin (default 9.5% from test MAPE)

        Returns:
            Dictionary with lower and upper bounds
        """
        lower = price * (1 - margin)
        upper = price * (1 + margin)

        return {
            "lower": float(lower),
            "upper": float(upper),
            "margin_percent": margin * 100
        }

    def _categorize_price(self, price: float) -> str:
        """Categorize price into market segments"""
        if price < 50000:
            return "budget"
        elif price < 80000:
            return "mid_range"
        elif price < 110000:
            return "premium"
        else:
            return "luxury"

    def get_feature_importance(self, top_n: int = 10) -> Dict:
        """
        Get top N most important features

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary with feature names and importance scores
        """
        try:
            importance = self.model.feature_importances_

            # Get feature names (numeric + encoded categorical)
            feature_names = (
                    self.numeric_features +
                    list(self.encoder.get_feature_names_out(self.categorical_features))
            )

            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)

            return importance_df.set_index('feature')['importance'].to_dict()

        except Exception as e:
            return {"error": f"Could not retrieve feature importance: {str(e)}"}


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🏠 ASTANA APARTMENT PRICE PREDICTOR")
    print("=" * 70)

    # Initialize predictor
    predictor = ApartmentPricePredictor("astana_price_model.pkl")

    # Example 1: Single prediction
    print("\n" + "=" * 70)
    print("📊 EXAMPLE 1: Single Apartment Prediction")
    print("=" * 70)

    example_apartment = {"rooms":2,"area":65.4,"living_area":46.7,"kitchen_area":10.6,"floor":7,"total_floors":12,"ceiling_height":3,"building_age":1,"district":"Yesil","latitude":51.1694,"longitude":71.4491,"house_type":"monolithic","condition":"good","parking":"public_parking","furniture":"unfurnished","bathroom_type":"multiple","bathroom_count":2,"balcony_type":"multiple","security_high":1,"wooden_floor":0,"has_window_grills":0,"floor_relative":0.583333,"living_ratio":0.714067,"kitchen_ratio":0.16208}

    result = predictor.predict(example_apartment)

    print("\n📍 Apartment Details:")
    print(f"   Rooms: {example_apartment['rooms']}")
    print(f"   Area: {example_apartment['area']} m²")
    print(f"   Floor: {example_apartment['floor']}/{example_apartment['total_floors']}")
    print(f"   District: {example_apartment['district']}")
    print(f"   Type: {example_apartment['house_type']}")

    print("\n💰 Prediction Results:")
    print(f"   Predicted Price: ${result['predicted_price_usd']:,.2f}")
    print(f"   Price per m²: ${result['price_per_m2']:,.2f}")
    print(f"   Price Category: {result['price_category']}")
    print(f"\n   95% Confidence Interval:")
    print(f"     Lower: ${result['confidence_interval_95']['lower']:,.2f}")
    print(f"     Upper: ${result['confidence_interval_95']['upper']:,.2f}")

    # Example 2: Feature importance
    print("\n" + "=" * 70)
    print("📈 TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 70)

    importance = predictor.get_feature_importance(top_n=10)
    for i, (feature, score) in enumerate(importance.items(), 1):
        print(f"{i:2d}. {feature:30s} {score:.4f}")

    # Example 3: Batch prediction
    print("\n" + "=" * 70)
    print("📦 EXAMPLE 2: Batch Prediction")
    print("=" * 70)

    apartments = [
        {**example_apartment, 'area': 45, 'rooms': 1, 'district': 'Saryarka'},
        {**example_apartment, 'area': 85, 'rooms': 3, 'district': 'Almaty'},
        {**example_apartment, 'area': 120, 'rooms': 4, 'district': 'Yesil'}
    ]

    batch_results = predictor.predict_batch(apartments)

    print("\nBatch Predictions:")
    for i, result in enumerate(batch_results, 1):
        if 'error' not in result:
            print(f"\n  {i}. {apartments[i - 1]['rooms']}-room, "
                  f"{apartments[i - 1]['area']}m², "
                  f"{apartments[i - 1]['district']}")
            print(f"     Price: ${result['predicted_price_usd']:,.2f} "
                  f"({result['price_category']})")
        else:
            print(f"\n  {i}. Error: {result['error']}")

    print("\n" + "=" * 70)
    print("✅ Prediction complete!")
    print("=" * 70)