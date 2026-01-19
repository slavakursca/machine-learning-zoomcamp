"""
Train Final XGBoost Model for Astana Apartment Price Prediction

This script trains the production model using optimal hyperparameters
discovered through Optuna optimization.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

print("=" * 70)
print("TRAINING FINAL XGBOOST MODEL")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\nLoading dataset...")
df = pd.read_csv('./dataset/astana_apartments_ready.csv')
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

# ============================================================================
# 2. DEFINE FEATURES
# ============================================================================
target = 'price_usd'

numeric_features = [
    'rooms', 'area', 'living_area', 'kitchen_area', 'floor', 'total_floors',
    'building_age', 'ceiling_height', 'latitude', 'longitude',
    'has_window_grills', 'security_high', 'bathroom_count', 'wooden_floor',
    'floor_relative', 'living_ratio', 'kitchen_ratio'
]

categorical_features = [
    'house_type', 'parking', 'furniture', 'district',
    'bathroom_type', 'balcony_type', 'condition'
]

print(f"\nFeatures:")
print(f"  - Numeric: {len(numeric_features)}")
print(f"  - Categorical: {len(categorical_features)}")

# ============================================================================
# 3. ENCODE CATEGORICAL FEATURES
# ============================================================================
print("\nOne-hot encoding categorical features...")
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_encoded = ohe.fit_transform(df[categorical_features])
cat_encoded_df = pd.DataFrame(
    cat_encoded,
    columns=ohe.get_feature_names_out(categorical_features)
)

# Combine numeric and encoded categorical features
X = pd.concat([
    df[numeric_features].reset_index(drop=True),
    cat_encoded_df.reset_index(drop=True)
], axis=1)

# Apply log transformation to target
y = np.log1p(df[target])

print(f"Final feature count: {X.shape[1]}")

# ============================================================================
# 4. SPLIT DATA
# ============================================================================
print("\nSplitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"  - Train: {X_train.shape[0]:,} samples")
print(f"  - Validation: {X_val.shape[0]:,} samples")
print(f"  - Test: {X_test.shape[0]:,} samples")

# ============================================================================
# 5. TRAIN FINAL MODEL WITH OPTIMAL HYPERPARAMETERS
# ============================================================================
print("\nTraining XGBoost with optimal hyperparameters...")
print("-" * 70)

# Optimal parameters from Optuna tuning (RMSE-optimized)
# optimal_params = {
#     'n_estimators': 1750,
#     'learning_rate': 0.036292477535162564,
#     'max_depth': 6,
#     'min_child_weight': 5,
#     'subsample': 0.9442231627495341,
#     'colsample_bytree': 0.6075430935454081,
#     'reg_alpha': 0.0074516636909294385,
#     'reg_lambda': 8.478313903382128e-07,
#     'gamma': 2.3516099974979833e-05,
#     'random_state': 42,
#     'tree_method': 'hist',
#     'early_stopping_rounds': 50
# }
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

print("\nOptimal Hyperparameters:")
for key, value in optimal_params.items():
    if key != 'early_stopping_rounds':
        print(f"  {key:20s}: {value}")

model = xgb.XGBRegressor(**optimal_params)

# Train with validation monitoring
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

print(f"\nTraining complete! Best iteration: {model.best_iteration}")

# ============================================================================
# 6. EVALUATE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)


def evaluate_model(model, X, y, dataset_name):
    """Evaluate model on a dataset"""
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n{dataset_name} Set:")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


train_metrics = evaluate_model(model, X_train, y_train, "Train")
val_metrics = evaluate_model(model, X_val, y_val, "Validation")
test_metrics = evaluate_model(model, X_test, y_test, "Test")

# ============================================================================
# 7. ERROR ANALYSIS BY PRICE SEGMENT
# ============================================================================
print("\n" + "=" * 70)
print("ERROR ANALYSIS BY PRICE SEGMENT (Test Set)")
print("=" * 70)

y_pred_test = np.expm1(model.predict(X_test))
y_test_orig = np.expm1(y_test)

bins = [0, 50000, 80000, 110000, np.inf]
labels = ['<50k', '50–80k', '80–110k', '>110k']

df_err = pd.DataFrame({
    'price': y_test_orig,
    'pred': y_pred_test
})

df_err['bin'] = pd.cut(df_err['price'], bins=bins, labels=labels)
df_err['abs_err'] = np.abs(df_err['price'] - df_err['pred'])

bin_summary = (
    df_err
    .groupby('bin', observed=True)
    .agg(
        count=('abs_err', 'count'),
        mean_abs_error=('abs_err', 'mean'),
        mean_abs_pct_error=(
            'abs_err',
            lambda x: np.mean(x / df_err.loc[x.index, 'price']) * 100
        )
    )
)

print("\n" + bin_summary.to_string())

# ============================================================================
# 8. SAVE MODEL AND ARTIFACTS
# ============================================================================
print("\n" + "=" * 70)
print("SAVING MODEL AND ARTIFACTS")
print("=" * 70)

# Save complete model package
model_package = {
    'model': model,
    'encoder': ohe,
    'feature_names': {
        'numeric': numeric_features,
        'categorical': categorical_features
    },
    'hyperparameters': optimal_params,
    'metrics': {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    },
    'metadata': {
        'model_type': 'XGBoost',
        'optimization_metric': 'RMSE',
        'training_samples': len(X_train),
        'features_count': X.shape[1]
    }
}

with open('astana_price_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("\nModel saved to: astana_price_model.pkl")
print("\nPackage contents:")
print("  - Trained XGBoost model")
print("  - OneHotEncoder for categorical features")
print("  - Feature names and types")
print("  - Optimal hyperparameters")
print("  - Performance metrics")
print("  - Training metadata")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\nFinal Model Performance (Test Set):")
print(f"   RMSE: ${test_metrics['rmse']:,.2f}")
print(f"   R²:   {test_metrics['r2']:.4f}")
print(f"   MAPE: {test_metrics['mape']:.2f}%")
print("\nModel ready for deployment!")
print("=" * 70)