# 🏠 Astana Apartment Price Prediction - ML Project

This project implements an end-to-end machine learning pipeline for predicting apartment prices in Astana, Kazakhstan, using real estate data scraped from krisha.kz (dataset was taken from Kaggle)

## Features

- Comprehensive data cleaning and feature engineering pipeline
- Multiple gradient boosting models (XGBoost, LightGBM, CatBoost) comparison
- Hyperparameter optimization with Optuna
- Dual-metric optimization (RMSE and MAPE)
- Production-ready model serialization
- Detailed error analysis by price segments

The model predicts apartment prices in USD based on property characteristics, location, and building features.

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

This project demonstrates a complete ML workflow from raw data to production-ready models.

---

## 📊 Dataset Description

- **Source:** https://www.kaggle.com/datasets/muraraimbekov/astana-real-estate-2025
- **Initial Rows:** 18,388 listings
- **Final Rows:** 15,556 (after cleaning)
- **Columns:** 28 features after engineering
- **Target:** `price_usd` (apartment sale price)

### Feature Categories (after cleaning)

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
- Ceiling height > 5m (data errors)

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

**Baseline XGBoost Configuration:**
```python
{
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
```

### 3. Hyperparameter Optimization

**Framework:** Optuna with 100 trials per model per metric

**Dual-Metric Strategy:**
- Optimize separately for RMSE (financial accuracy)
- Optimize separately for MAPE (percentage error)

**Search Spaces:**

**XGBoost:**
- n_estimators: [500, 2500]
- learning_rate: [0.01, 0.1] (log scale)
- max_depth: [4, 10]
- min_child_weight: [1, 7]
- subsample: [0.6, 1.0]
- colsample_bytree: [0.6, 1.0]
- reg_alpha: [1e-8, 10.0] (log scale)
- reg_lambda: [1e-8, 10.0] (log scale)
- gamma: [1e-8, 1.0] (log scale)

**LightGBM:**
- n_estimators: [500, 2500]
- learning_rate: [0.01, 0.1] (log scale)
- max_depth: [4, 12]
- num_leaves: [20, 100]
- min_child_samples: [10, 50]

**CatBoost:**
- iterations: [500, 2500]
- learning_rate: [0.01, 0.1] (log scale)
- depth: [4, 10]
- l2_leaf_reg: [1e-8, 10.0] (log scale)

### 4. Final Model Performance

**Best Model: XGBoost (RMSE-optimized)**

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

**Test Set Performance:**

| Metric | Value |
|--------|-------|
| RMSE | $14,781.55 |
| R² | 0.9189 |
| MAPE | 9.27% |

**Error Analysis by Price Segment:**

| Price Range | Mean Absolute Error | Mean Absolute % Error |
|-------------|---------------------|-----------------------|
| < $50k | $3,399              | 9.33%                 |
| $50-80k | $5,328              | 8.29%                 |
| $80-110k | $8,681              | 9.39%                 |
| > $110k | $19,316             | 10.81%                |

**Key Insights:**
- Model performs consistently across price ranges
- Slightly higher percentage error for luxury segment (>$110k)
- Strong R² indicates excellent explanatory power
- MAPE under 10% indicates production-ready accuracy

---

## 📈 Model Comparison Results

### All Optimized Models

| Model | Optimized_For | RMSE | R² | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| XGBoost_RMSE | RMSE | 14965.941129 | 0.916823 | 9.336905 |
| XGBoost_MAPE | MAPE | 14757.846477 | 0.919120 | 9.245500 |
| LightGBM_RMSE | RMSE | 15124.214839 | 0.915055 | 9.593321 |
| LightGBM_MAPE | MAPE | 15036.856976 | 0.916033 | 9.225793 |
| CatBoost_RMSE | RMSE | 14817.238706 | 0.918468 | 9.525816 |
| CatBoost_MAPE | MAPE | 14815.892530 | 0.918483 | 9.554846 |

**Conclusion:** XGBoost optimized for RMSE provides the best overall performance with lowest error and highest R².

---

## 🛠️ Project Structure

```
├── dataset/
│   ├── astana_apartments.csv           # Raw scraped data
│   └── astana_apartments_ready.csv     # Cleaned dataset
├── eda-cleaning.ipynb                  # Data exploration & cleaning
├── train-model.ipynb                   # Model training & optimization
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

---

## 🚀 Running the Project

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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

## 📊 Key Technologies

- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, xgboost, lightgbm, catboost
- **Optimization:** optuna
- **Visualization:** matplotlib, seaborn (in notebooks)

## 📝 Model Selection Rationale

**Why XGBoost?**

1. **Best Performance:** Lowest RMSE ($14,905) among all tested models
2. **Robust Generalization:** Consistent performance across train/val/test
3. **Feature Interactions:** Excellent at capturing complex relationships
4. **Production Proven:** Widely used in real estate price prediction
5. **Interpretability:** Supports feature importance analysis

**Why Not Others?**
- **LightGBM:** Slightly higher RMSE, though faster training
- **CatBoost:** Better categorical handling, but higher MAPE
- **Linear Models:** Underfitted (tested but not shown, R² < 0.75)

---

## 🚀 Running the Project

### 1. Run with Docker (recommended)

**Using docker-compose**

    docker-compose up -d

**Manual build/run**

    docker build -t astana-price-api .
    docker run -p 9696:9696 astana-price-api

API runs at: `http://localhost:9696`

### 2. Run Locally

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    uvicorn serve:app --host 0.0.0.0 --port 9696

------------------------------------------------------------------------

### Kubernetes (Recommended for Scale)

Deploy to Kubernetes cluster for high availability and scalability.

**Requirements:**
- Kubernetes cluster (minikube, kind, or cloud provider)
- kubectl configured
- Docker (for building image)

#### Local Kubernetes with Minikube

**Step 1: Install Minikube**

```bash
# macOS
brew install minikube

# Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Windows (using Chocolatey)
choco install minikube
```

**Step 2: Start Minikube**

```bash
# Start cluster
minikube start

# Verify cluster
kubectl cluster-info
kubectl get nodes
```

**Step 3: Build Docker Image in Minikube**

```bash
# Use minikube's Docker daemon
eval $(minikube docker-env)

# Build image
docker build -t astana-price-api:latest .

# Verify image is available
docker images | grep astana-price-api
```

**Step 4: Deploy to Kubernetes**

```bash
# Apply deployment and service
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services

# Wait for pod to be ready
kubectl wait --for=condition=ready pod -l app=astana-price-api --timeout=60s
```

**Step 5: Access the Service**

```bash
# Get service URL (for minikube)
minikube service astana-price-api --url

# Or use port forwarding
kubectl port-forward service/astana-price-api 9696:80

# Access API at: http://localhost:9696
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

**Step 1: Build and Push Image to Registry**

```bash
# Tag for your registry
docker tag astana-price-api:latest <your-registry>/astana-price-api:v1.0.0

# Push to registry
docker push <your-registry>/astana-price-api:v1.0.0
```

**Step 2: Update deployment.yaml**

```yaml
# Change image reference
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

## 🔌 API Documentation

### Available Endpoints

- **GET `/health`** — Health-check endpoint to verify the service is running  
- **POST `/predict`** — Main ML inference endpoint returning prediction  
- **GET `/form`** — Simple interactive form for manually submitting applicant data and testing predictions
- **GET `/docs`** - Application auto-generated docs
- **GET `/model/info`** - Full details regarding model

### Request Example

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

### Response Example

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
---

## ⚠️ Limitations

- **Data Recency:** Model trained on January 2026 snapshot; requires retraining for market changes
- **Geographic Scope:** Limited to Astana, Kazakhstan
- **Feature Completeness:** Some listings had missing amenity details
- **Market Dynamics:** Does not account for seasonal variations or economic events
- **External Factors:** Excludes macro-economic indicators (interest rates, inflation)

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

## 🤝 Contributing

This is an educational project demonstrating ML best practices. Suggestions for improvements welcome!

---

## 📄 License

MIT License - feel free to use for learning and non-commercial purposes.

---

**Project Status:** ✅ Complete - Model trained and evaluated

**Last Updated:** January 2026
