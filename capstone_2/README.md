# 🏠 Astana Apartment Price Prediction - ML Project

![Python](https://img.shields.io/badge/python-3.9+-blue)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

This project implements an end-to-end machine learning pipeline for predicting apartment prices in Astana, Kazakhstan, using real estate data scraped from krisha.kz (dataset was taken from Kaggle).

## Features

- Comprehensive data cleaning and feature engineering pipeline
- Multiple gradient boosting models (XGBoost, LightGBM, CatBoost) comparison
- Hyperparameter optimization with Optuna
- Dual-metric optimization (RMSE and MAPE)
- Production-ready REST API with FastAPI
- Docker and Kubernetes deployment support
- Detailed error analysis by price segments

The model predicts apartment prices in USD based on property characteristics, location, and building features.

---

## 🚀 Quick Start

```bash
# Clone repository
git clone git@github.com:slavakursca/machine-learning-zoomcamp.git
cd capstone_2

# Run with Docker Compose (recommended)
docker-compose up -d

# Test the API
curl http://localhost:9696/health

# Try interactive form
open http://localhost:9696/form
```

**API will be available at:** `http://localhost:9696`

---

## 📑 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset Description](#-dataset-description)
- [Data Cleaning & Feature Engineering](#-data-cleaning--feature-engineering)
- [Modeling Approach](#-modeling-approach)
- [Model Performance](#-model-performance)
- [Project Structure](#%EF%B8%8F-project-structure)
- [Development Setup](#-development-setup)
- [Deployment & API Usage](#-deployment--api-usage)
- [API Documentation](#-api-documentation)
- [Performance Metrics](#-performance-metrics)
- [Limitations](#%EF%B8%8F-limitations--considerations)
- [Future Improvements](#-roadmap)

---

## 🧩 Problem Statement

Real estate pricing is complex and influenced by multiple factors. Accurate price predictions help:

- **Buyers** - Make informed purchase decisions
- **Sellers** - Set competitive listing prices
- **Agents** - Provide data-driven valuations
- **Investors** - Identify undervalued opportunities

**Objective:**
- Predict apartment sale price in USD
- Minimize prediction error across different price ranges
- Provide interpretable results with feature importance

This project demonstrates a complete ML workflow from raw data to production-ready API.

---

## 📊 Dataset Description

- **Source:** https://www.kaggle.com/datasets/muraraimbekov/astana-real-estate-2025
- **Initial Rows:** 18,388 listings
- **Final Rows:** 15,556 (after cleaning)
- **Features:** 28 features after engineering
- **Target:** `price_usd` (apartment sale price)

### Feature Categories (After Cleaning)

**Physical Attributes:**
- `rooms` - Number of rooms
- `area` - Total area (m²)
- `living_area` - Living space (m²)
- `kitchen_area` - Kitchen space (m²)
- `floor` - Floor number
- `total_floors` - Building height
- `ceiling_height` - Ceiling height (m)

**Building Characteristics:**
- `house_type` - Construction type (brick, monolithic, panel, other)
- `building_age` - Years since construction
- `building_stage` - Under construction flag

**Location:**
- `district` - City district (Yesil, Nura, Almaty, Saryarka, Saryshyk, Baikonur)
- `latitude` - Geographic coordinate
- `longitude` - Geographic coordinate

**Amenities & Features:**
- `parking` - Parking availability
- `furniture` - Furnishing status
- `condition` - Renovation condition
- `bathroom_count` - Number of bathrooms
- `bathroom_type` - Bathroom configuration
- `balcony_type` - Balcony/loggia type
- `wooden_floor` - Premium flooring indicator
- `security_high` - Security features
- `has_window_grills` - Security bars

**Engineered Features:**
- `floor_relative` - floor / total_floors
- `living_ratio` - living_area / area
- `kitchen_ratio` - kitchen_area / area
- `price_per_m2` - Price per square meter

---

## 🔍 Data Cleaning & Feature Engineering

### Data Quality Issues Addressed

**1. Missing Values:**
- `ceiling_height` - Filled with 2.7m (median), 3.0m for luxury (>$1,300/m²)
- `floor` / `total_floors` - Filled with -1 for missing floor information
- `living_area` - Imputed using mean ratio (71.3% of total area)
- `kitchen_area` - Imputed using mean ratio (20.4% of total area)

**2. Outliers Removed:**
- Rows with >9 missing values (high data quality threshold)
- Area > 150 m² (extreme outliers)
- Price per m² > $3,500 (exclude super-luxury/unrealistic pricing)
- Ceiling height > 5m (likely data entry errors)

**3. Data Validation:**
- Fixed invalid kitchen_area (>35% of total area)
- Fixed invalid living_area (>95% of total area)
- Verified floor ≤ total_floors consistency

### Feature Engineering

**Categorical Encoding:**
- `house_type` - Mapped to English: monolithic, brick, panel, other, unknown
- `condition` - Mapped: new, good, rough, needs_repair, unknown
- `bathroom_type` - Extracted from text: combined, separate, multiple
- `balcony_type` - Categorized: balcony, loggia, multiple, unknown
- `parking` - Mapped: private_parking, public_parking, garage, no_parking
- `furniture` - Mapped: fully_furnished, partially_furnished, unfurnished, unknown
- `district` - Translated to English names

**Binary Features:**
- `wooden_floor` - Premium flooring indicator
- `security_high` - High security features (security, alarm, concierge)
- `has_window_grills` - Window security bars (high danger area)
- `building_stage` - Under construction flag

**Derived Features:**
- `bathroom_count` - Extracted count (0, 1, 2+)
- `building_age` - 2026 - year_built

---

## 🤖 Modeling Approach

### 1. Data Split Strategy

- **Training:** 70% (10,889 samples)
- **Validation:** 15% (2,333 samples)  
- **Test:** 15% (2,334 samples)

**Preprocessing:**
- Target transformation: `log1p(price_usd)` for stable training
- Numeric features: Used as-is (no scaling for tree models)
- Categorical features: One-hot encoded for XGBoost/LightGBM, native encoding for CatBoost

### 2. Baseline Model Comparison

Three gradient boosting algorithms evaluated:

| Model | Test RMSE | Test R² | Test MAPE | Training Approach |
|-------|-----------|---------|-----------|-------------------|
| **XGBoost** | **$14,905** | **0.9175** | **9.48%** | 1000 rounds, early stopping |
| LightGBM | $15,341 | 0.9126 | 9.67% | 920 rounds (early stopped) |
| CatBoost | $15,259 | 0.9135 | 9.97% | 999 rounds, native categorical |

**Winner: XGBoost** selected for best RMSE and balanced performance.

### 3. Hyperparameter Optimization

**Framework:** Optuna with 100 trials per model per metric

**Dual-Metric Strategy:**
- Optimize separately for RMSE (financial accuracy)
- Optimize separately for MAPE (percentage error)

**Search Spaces:**

<details>
<summary>XGBoost Hyperparameter Ranges</summary>

- n_estimators: [500, 2500]
- learning_rate: [0.01, 0.1] (log scale)
- max_depth: [4, 10]
- min_child_weight: [1, 7]
- subsample: [0.6, 1.0]
- colsample_bytree: [0.6, 1.0]
- reg_alpha: [1e-8, 10.0] (log scale)
- reg_lambda: [1e-8, 10.0] (log scale)
- gamma: [1e-8, 1.0] (log scale)

</details>

<details>
<summary>LightGBM Hyperparameter Ranges</summary>

- n_estimators: [500, 2500]
- learning_rate: [0.01, 0.1] (log scale)
- max_depth: [4, 12]
- num_leaves: [20, 100]
- min_child_samples: [10, 50]

</details>

<details>
<summary>CatBoost Hyperparameter Ranges</summary>

- iterations: [500, 2500]
- learning_rate: [0.01, 0.1] (log scale)
- depth: [4, 10]
- l2_leaf_reg: [1e-8, 10.0] (log scale)

</details>

---

## 📈 Model Performance

### 🎯 Final Model: XGBoost (RMSE-Optimized)

**Test Set Performance:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | $14,781.55 | Average prediction error is ~$14.8k |
| **R²** | 0.9189 | Model explains 91.9% of price variance |
| **MAPE** | 9.27% | Typical error is 9.3% of actual price |

**Optimal Hyperparameters:**
```python
optimal_params = {
    "colsample_bytree": 0.6174243195963651,
    "gamma": 0.00029194695300892537,
    "learning_rate": 0.025072744653519028,
    "max_depth": 8,
    "min_child_weight": 3,
    "n_estimators": 2000,
    "reg_alpha": 0.017313707755127073,
    "reg_lambda": 0.07549915003331413,
    "subsample": 0.8978305041846237,
    "random_state": 42,
    "tree_method": 'hist',
    "early_stopping_rounds": 50
}
```

**Error Analysis by Price Segment:**

| Price Range | Mean Absolute Error | Mean Absolute % Error |
|-------------|---------------------|-----------------------|
| < $50k | $3,399 | 9.33% |
| $50-80k | $5,328 | 8.29% |
| $80-110k | $8,681 | 9.39% |
| > $110k | $19,316 | 10.81% |

**Key Insights:**
- Consistent performance across most price ranges
- Slightly higher percentage error for luxury segment (>$110k)
- Strong R² indicates excellent explanatory power
- MAPE under 10% indicates production-ready accuracy

### All Optimized Models Comparison

| Model | Optimized For | RMSE | R² | MAPE (%) |
|-------|---------------|------|-----|----------|
| XGBoost_RMSE | RMSE | 14,965.94 | 0.9168 | 9.34 |
| **XGBoost_MAPE** | **MAPE** | **14,757.85** | **0.9191** | **9.25** |
| LightGBM_RMSE | RMSE | 15,124.21 | 0.9151 | 9.59 |
| LightGBM_MAPE | MAPE | 15,036.86 | 0.9160 | 9.23 |
| CatBoost_RMSE | RMSE | 14,817.24 | 0.9185 | 9.53 |
| CatBoost_MAPE | MAPE | 14,815.89 | 0.9185 | 9.55 |

<details>
<summary>📋 Model Selection Rationale</summary>

**Why XGBoost?**

1. **Best Overall Performance:** Lowest RMSE ($14,781) with highest R² (0.9189)
2. **Robust Generalization:** Consistent performance across train/validation/test sets
3. **Feature Interactions:** Excellent at capturing complex non-linear relationships
4. **Production Proven:** Widely used in real estate price prediction industry
5. **Interpretability:** Strong support for feature importance analysis

**Why Not Others?**
- **LightGBM:** Slightly higher RMSE, though faster training time
- **CatBoost:** Better native categorical handling, but marginally higher MAPE
- **Linear Models:** Underfitted on this dataset (tested but not shown, R² < 0.75)

</details>

---

## 🗂️ Project Structure

```
astana-apartment-prediction/
├── dataset/
│   ├── astana_apartments.csv           # Raw scraped data
│   └── astana_apartments_ready.csv     # Cleaned dataset
├── eda-cleaning.ipynb                  # Data exploration & cleaning
├── train-model.ipynb                   # Model training & optimization
├── serve.py                            # FastAPI application
├── midterm_model.bin                   # Trained XGBoost model (15 MB)
├── Dockerfile                          # Container configuration
├── docker-compose.yml                  # Docker Compose setup
├── deployment.yaml                     # Kubernetes deployment config
├── service.yaml                        # Kubernetes service config
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## 🛠️ Development Setup

Use this section if you want to train models or experiment with the notebooks.

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Run `eda-cleaning.ipynb` to:
- Load raw data from `dataset/astana_apartments.csv`
- Clean and validate features
- Engineer new features
- Save to `dataset/astana_apartments_ready.csv`

### 3. Model Training

Run `train-model.ipynb` to:
- Load cleaned dataset
- Train baseline models (XGBoost, LightGBM, CatBoost)
- Run Optuna hyperparameter optimization
- Compare all model variants
- Save best model to `midterm_model.bin`

---

## 🚀 Deployment & API Usage

### Option 1: Docker Compose (Recommended)

```bash
# Start service
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop service
docker-compose down
```

### Option 2: Docker (Manual)

```bash
# Build image
docker build -t astana-price-api .

# Run container
docker run -p 9696:9696 astana-price-api

# Run with custom port
docker run -p 8080:9696 astana-price-api
```

### Option 3: Local Python

```bash
# Activate environment
source venv/bin/activate

# Start server
uvicorn serve:app --host 0.0.0.0 --port 9696

# Or with reload for development
uvicorn serve:app --reload --port 9696
```

### Option 4: Kubernetes

Deploy to Kubernetes cluster for high availability and scalability.

**Requirements:**
- Kubernetes cluster (minikube, kind, or cloud provider)
- kubectl configured
- Docker installed

#### Local Kubernetes with Minikube

**Step 1: Install & Start Minikube**

```bash
# macOS
brew install minikube

# Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Windows (Chocolatey)
choco install minikube

# Start cluster
minikube start

# Verify
kubectl cluster-info
kubectl get nodes
```

**Step 2: Build Docker Image**

```bash
# Use minikube's Docker daemon
eval $(minikube docker-env)

# Build image
docker build -t astana-price-api:latest .

# Verify
docker images | grep astana-price-api
```

**Step 3: Deploy Application**

```bash
# Apply configurations
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check status
kubectl get deployments
kubectl get pods
kubectl get services

# Wait for pod to be ready
kubectl wait --for=condition=ready pod -l app=astana-price-api --timeout=60s
```

**Step 4: Access the Service**

```bash
# Get service URL (minikube)
minikube service astana-price-api --url

# Or use port forwarding
kubectl port-forward service/astana-price-api 9696:80

# Access at: http://localhost:9696
```

**Useful Commands:**

```bash
# View logs
kubectl logs -l app=astana-price-api -f

# Describe pod for debugging
kubectl describe pod -l app=astana-price-api

# Scale deployment
kubectl scale deployment astana-price-api --replicas=3

# Delete deployment
kubectl delete -f deployment.yaml
kubectl delete -f service.yaml
```

#### Production Kubernetes (Cloud)

For AWS EKS, GCP GKE, or Azure AKS:

**Step 1: Build and Push to Registry**

```bash
# Tag for your registry
docker tag astana-price-api:latest <your-registry>/astana-price-api:v1.0.0

# Push to registry
docker push <your-registry>/astana-price-api:v1.0.0
```

**Step 2: Update deployment.yaml**

```yaml
spec:
  containers:
  - name: astana-price-api
    image: <your-registry>/astana-price-api:v1.0.0
    imagePullPolicy: Always  # Change to Always for cloud
```

**Step 3: Deploy**

```bash
# Apply configurations
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Get external IP (may take a few minutes)
kubectl get service astana-price-api -w
```

---

## 📌 API Documentation

### Available Endpoints

- **GET `/health`** — Health check endpoint
- **POST `/predict`** — ML inference endpoint (returns price prediction)
- **GET `/form`** — Interactive HTML form for manual testing
- **GET `/docs`** — Auto-generated Swagger UI documentation
- **GET `/model/info`** — Model metadata and feature information

### Testing the API

**Using curl:**

```bash
# Health check
curl http://localhost:9696/health

# Make prediction
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rooms": 2,
    "area": 65.4,
    "living_area": 46.7,
    "kitchen_area": 10.6,
    "floor": 7,
    "total_floors": 12,
    "ceiling_height": 3,
    "building_age": 1,
    "district": "Yesil",
    "latitude": 51.1694,
    "longitude": 71.4491,
    "house_type": "monolithic",
    "condition": "good",
    "parking": "public_parking",
    "furniture": "unfurnished",
    "bathroom_type": "multiple",
    "bathroom_count": 2,
    "balcony_type": "multiple",
    "security_high": 1,
    "wooden_floor": 0,
    "has_window_grills": 0,
    "floor_relative": 0.583333,
    "living_ratio": 0.714067,
    "kitchen_ratio": 0.16208
  }'
```

**Using Python:**

```python
import requests

# Prediction request
data = {
    "rooms": 2,
    "area": 65.4,
    "living_area": 46.7,
    "kitchen_area": 10.6,
    "floor": 7,
    "total_floors": 12,
    "ceiling_height": 3,
    "building_age": 1,
    "district": "Yesil",
    "latitude": 51.1694,
    "longitude": 71.4491,
    "house_type": "monolithic",
    "condition": "good",
    "parking": "public_parking",
    "furniture": "unfurnished",
    "bathroom_type": "multiple",
    "bathroom_count": 2,
    "balcony_type": "multiple",
    "security_high": 1,
    "wooden_floor": 0,
    "has_window_grills": 0,
    "floor_relative": 0.583333,
    "living_ratio": 0.714067,
    "kitchen_ratio": 0.16208
}

response = requests.post(
    "http://localhost:9696/predict",
    json=data
)

print(response.json())
```

**Interactive Testing:**

1. **HTML Form:** Open `http://localhost:9696/form` in your browser
2. **Swagger UI:** Visit `http://localhost:9696/docs` for interactive API documentation

### Request Example

```json
{
  "rooms": 2,
  "area": 65.4,
  "living_area": 46.7,
  "kitchen_area": 10.6,
  "floor": 7,
  "total_floors": 12,
  "ceiling_height": 3,
  "building_age": 1,
  "district": "Yesil",
  "latitude": 51.1694,
  "longitude": 71.4491,
  "house_type": "monolithic",
  "condition": "good",
  "parking": "public_parking",
  "furniture": "unfurnished",
  "bathroom_type": "multiple",
  "bathroom_count": 2,
  "balcony_type": "multiple",
  "security_high": 1,
  "wooden_floor": 0,
  "has_window_grills": 0,
  "floor_relative": 0.583333,
  "living_ratio": 0.714067,
  "kitchen_ratio": 0.16208
}
```

### Response Example

```json
{
  "predicted_price_usd": 71796.32,
  "price_per_m2": 1097.8,
  "confidence_interval_95": {
    "lower": 64975.67,
    "upper": 78616.98,
    "margin_percent": 9.5
  },
  "price_category": "mid-range"
}
```

---

## ⚡ Performance Metrics

- **Model Size:** ~15 MB (serialized with pickle)
- **Inference Time:** ~10-15ms per prediction (single request)
- **Memory Usage:** ~200 MB (includes FastAPI application overhead)
- **Throughput:** ~100 requests/second (single container, no load balancer)
- **Cold Start:** ~2-3 seconds (model loading time)

**Optimization Notes:**
- Model is loaded once at startup and kept in memory
- No GPU required (CPU inference is sufficient)
- Scales horizontally with Kubernetes replicas

---

## ⚠️ Limitations & Considerations

### Data Limitations
- **Temporal Scope:** Training data from January 2026 snapshot only
- **Geographic Scope:** Limited to Astana, Kazakhstan; not generalizable to other cities
- **Missing Features:** ~15% of listings had imputed values for amenity features
- **Class Imbalance:** Fewer luxury apartments (>$110k) in training set

### Model Limitations
- **No Temporal Features:** Does not account for market trends or seasonality
- **External Factors Excluded:** Macro-economic indicators (interest rates, inflation, GDP growth)
- **Luxury Segment Error:** Higher MAPE (10.8%) for apartments >$110k
- **Static Predictions:** Does not incorporate real-time market conditions
- **Feature Dependencies:** Requires all 28 features; cannot handle partial inputs

### Operational Considerations
- **Drift Detection:** No automated monitoring for data/concept drift
- **Retraining Needed:** Model should be retrained quarterly as market evolves
- **Input Validation:** Production deployment requires robust validation middleware
- **Geographic Boundaries:** Coordinates outside Astana may produce unreliable predictions
- **New Construction:** May underestimate prices for areas with rapid development

---

## 🎯 Future Improvements

**Model Enhancements:**
- [ ] Time-series features (listing duration, price changes)
- [ ] External data (proximity to metro, schools, parks)
- [ ] Ensemble stacking of top models
- [ ] Deep learning with embeddings for text descriptions
- [ ] Automated retraining pipeline

**Production Deployment:**
- [ ] Model versioning with MLflow
- [ ] A/B testing framework
- [ ] Real-time price monitoring dashboard

**Feature Engineering:**
- [ ] Neighborhood price trends
- [ ] Developer reputation scores
- [ ] Walkability scores
- [ ] Public transport accessibility metrics

---

## 📚 Key Technologies

- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, xgboost, lightgbm, catboost
- **Optimization:** optuna
- **API Framework:** FastAPI, uvicorn
- **Containerization:** Docker, Docker Compose
- **Orchestration:** Kubernetes
- **Visualization:** matplotlib, seaborn (in notebooks)

---

## 🤝 Contributing

This is an educational project demonstrating ML best practices. Suggestions for improvements welcome!

---

## 📄 License

MIT License - feel free to use for learning and non-commercial purposes.

---

## 📧 Contact

For questions, feedback, or collaboration opportunities, please open an issue on GitHub.

---

**Project Status:** ✅ Complete - Model trained, evaluated, and deployed

**Last Updated:** January 2026

**Version:** 1.0.0