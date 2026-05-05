# Model Training - Baseline
#Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score
)
from sklearn.model_selection import cross_val_score, learning_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


#Train the baseline model - Logistic Regression
def train_model(X_train, y_train):
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000
    )

    model.fit(X_train, y_train)
    return model

# Evaluate model performance 
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def evaluate_thresholds(model, X_test, y_test):
    """Evaluate different classification thresholds."""

    y_probs = model.predict_proba(X_test)[:, 1]

    thresholds = [0.3, 0.5, 0.7, 0.9]

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"\nThreshold: {t}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")


## Train Model with Random Forest
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model

# Plot Feature Importance for tree-based models

def plot_feature_importance(model, X_train):
    
    importances = model.feature_importances_
    feature_names = X_train.columns

    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(df_importance)

    #plot
    plt.figure(figsize=(8,5))
    plt.barh(df_importance["feature"], df_importance["importance"])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance - Random Forest")
    plt.savefig("outputs/feature_importance.png")
    plt.close()

# Train Model with XGBoost

def train_xgboost(X_train, y_train):

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=28,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    return model

# Cross-validation
def evaluate_cross_validation(model, X_train, y_train):
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring='recall'
    )

    print("\nCross-Validation Recall Scores:", scores)
    print("Mean Recall:", scores.mean())

# Plot Learning Curve
def plot_learning_curve(model, X_train, y_train):
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=5,
        scoring='recall',
        train_sizes=np.linspace(0.1, 1.0, 5)
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, label="Train Recall")
    plt.plot(train_sizes, val_mean, label="Validation Recall")
    plt.xlabel("Training Size")
    plt.ylabel("Recall")
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig("outputs/learning_curve.png")
    plt.close()