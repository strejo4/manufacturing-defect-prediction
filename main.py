from src.data.preprocess import (
    load_data,
    clean_data,
    encode_features,
    split_data,
    scale_data,
    clean_column_names
)
from src.models.train import (
    train_model,
    train_random_forest,
    evaluate_model,
    evaluate_thresholds,
    plot_feature_importance,
    train_xgboost,
    evaluate_cross_validation,
    plot_learning_curve
)

# Load
df = load_data("data/raw/ai4i2020.csv")

# Clean
df = clean_data(df)

# Encode
df = encode_features(df)
df = clean_column_names(df)

#debug
print(df.head())

# Split
X_train, X_test, y_train, y_test = split_data(df)

# Scale
X_train, X_test, scaler = scale_data(X_train, X_test)

print("Pipeline ready:")
print(X_train.shape, X_test.shape)

print("\n--- Logistic Regression ---")
model = train_model(X_train, y_train)

evaluate_model(model, X_test, y_test)
evaluate_thresholds(model, X_test, y_test)

print("\n--- Random Forest ---")

rf_model = train_random_forest(X_train, y_train)

evaluate_model(rf_model, X_test, y_test)
evaluate_thresholds(rf_model, X_test, y_test)

# Plot feature importance
plot_feature_importance(rf_model, X_train)

print("\n--- XGBoost ---")

xgb_model = train_xgboost(X_train, y_train)

evaluate_model(xgb_model, X_test, y_test)
evaluate_thresholds(xgb_model, X_test, y_test)

#Cross-validation
print("\n--- Cross Validation (XGBoost) ---")
evaluate_cross_validation(xgb_model, X_train, y_train)

#Learning curve
print("\n--- Learning Curve (XGBoost) ---")
plot_learning_curve(xgb_model, X_train, y_train)

