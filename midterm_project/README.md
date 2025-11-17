# 📘 Loan Default Probability Prediction --- Machine Learning API

This project implements an end-to-end loan default risk prediction
system.

## Features

-   Full ML training pipeline (preprocessing → model selection →
    threshold tuning → serialization)
-   FastAPI inference service exposing a `/predict` endpoint
-   Dockerized deployment for reproducible and portable serving
-   Clean REST interface returning both probability and binary
    classification

The model predicts the probability that a borrower will default, based
on demographic, financial, and credit history features.

------------------------------------------------------------------------

## 🧩 Problem Statement

Financial institutions must determine which loan applicants are likely
to default. Incorrect decisions lead to:

-   High financial losses (false negatives)\
-   Missed business opportunities (false positives)

Objective:

-   Predict default probability\
-   Convert probability into a binary decision using an optimized
    threshold\
-   Serve results in real time via an API

This project demonstrates a production-ready approach to credit risk
scoring.

------------------------------------------------------------------------

## 📊 Dataset Description

-   **Rows:** 45,000\
-   **Columns:** 14\
-   **Target:** `loan_status` (1 = Default, 0 = Paid)

Includes demographic attributes, financial indicators, and credit
history features.

### Feature Summary

  -----------------------------------------------------------------------------
  Feature                          Type                 Notes
  -------------------------------- -------------------- -----------------------
  person_age                       int                  20--144 (outliers
                                                        addressed)

  person_gender                    categorical          male/female

  person_education                 categorical          HS, Bachelor, Master,
                                                        PhD

  person_income                    float                annual income

  person_emp_exp                   int                  years of employment

  person_home_ownership            categorical          RENT, OWN, MORTGAGE

  loan_amnt                        float                loan request amount

  loan_intent                      categorical          EDUCATION, MEDICAL,
                                                        VENTURE, etc.

  loan_int_rate                    float                interest rate

  loan_percent_income              float                loan amount / income

  cb_person_cred_hist_length       int                  credit history length

  credit_score                     int                  credit score

  previous_loan_defaults_on_file   categorical → binary Yes/No converted to 1/0
  -----------------------------------------------------------------------------

Dataset link: [Kaggle
Dataset](https://www.kaggle.com/datasets/sumit12100012/loan-approval-classification)

------------------------------------------------------------------------

## 🔎 Exploratory Data Analysis (EDA)

### Key Observations

-   Age distribution concentrated 25--55; outliers above 100 removed
-   Income heavily right-tailed; extreme values filtered
-   Credit score strongly negative correlated with default
-   `loan_percent_income` is one of strongest predictors
-   Prior defaults drastically increase default risk
-   Loan intents well balanced

### Data Cleaning

-   Removed unrealistic age and income
-   Converted Yes/No to binary
-   Verified no severe class imbalance (~60/40)
-   One-hot encoded categorical features

------------------------------------------------------------------------

## 🤖 Modeling Approach

### 1. Data Split

-   60% Train
-   20% Validation
-   20% Test

### 2. Models Evaluated

| Model                 | Pros                                     | Cons                             |
|-----------------------|------------------------------------------|----------------------------------|
| Logistic Regression   | Interpretable                            | Underfits nonlinear relations    |
| Decision Tree         | Captures interactions                    | Overfits easily                  |
| Random Forest (Selected) | Best accuracy, robust, handles mixed features | Larger model size                |


**Random Forest chosen** for stable performance and good probability
calibration.

### 3. Threshold Optimization

A sweep from 0.1--0.8 determined the F1-maximizing threshold.
Stored along with:

-   RandomForest model
-   DictVectorizer
-   Final probability cutoff

Saved in: `midterm_model.bin`

### 4. Final Performance (Example)

| Metric     | Score |
|------------|-------|
| Accuracy   | ~0.82 |
| Precision  | ~0.78 |
| Recall     | ~0.75 |
| F1-Score   | ~0.76 |
| AUC-ROC    | ~0.86 |

------------------------------------------------------------------------

## 🏛️ API Service Architecture

```
    graph LR
        A[Client Sends JSON Request] --> B[FastAPI Endpoint /predict]
        B --> C[ModelPredictor Class (predict.py)]
        C --> D[Load Model + DictVectorizer]
        C --> E[Compute Probability]
        E --> F[Apply Tuned Threshold]
        F --> G[Return JSON Response]
```

------------------------------------------------------------------------

## 🚀 Running the Project

### 1. Run with Docker (recommended)

**Using docker-compose**

    docker-compose up -d

**Manual build/run**

    docker build -t loan-default-api .
    docker run -p 9696:9696 loan-default-api

API runs at: `http://localhost:9696`

### 2. Run Locally

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    uvicorn serve:app --host 0.0.0.0 --port 9696

------------------------------------------------------------------------

## 🔌 API Documentation

### Endpoints

  Method   Endpoint   Description
  -------- ---------- ---------------------
  GET      /          Welcome message
  GET      /health    Health check
  POST     /predict   Generate prediction

### Request Example

    {
      "person_age": 35,
      "person_gender": "male",
      "person_education": "Bachelor",
      "person_income": 75000,
      "person_emp_exp": 10,
      "person_home_ownership": "RENT",
      "loan_amnt": 15000,
      "loan_intent": "EDUCATION",
      "loan_int_rate": 5.5,
      "loan_percent_income": 0.2,
      "cb_person_cred_hist_length": 8,
      "credit_score": 720,
      "previous_loan_defaults_on_file": 0
    }

### Response Example

    {
      "prediction": false,
      "probability": 0.1234,
      "threshold": 0.34,
      "prediction_class": 0
    }

------------------------------------------------------------------------

## ⚠️ Limitations

-   Trained on public dataset → may not reflect real-world patterns
-   No fairness/bias review
-   No CI/CD retraining automation

------------------------------------------------------------------------

## 🛠 Next Steps

-   Feature importance stability checks
-   Stronger Pydantic input schemas
-   Cloud deployment (AWS ECS/EC2, Render, Railway)
