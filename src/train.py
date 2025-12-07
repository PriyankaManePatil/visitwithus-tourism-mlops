import os
import json
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

from huggingface_hub import HfApi

# Read dataset/model repo IDs from environment variables set in GitHub Actions
DATASET_REPO_ID = os.environ.get(
    "HF_DATASET",
    "priyankamane10/visitwithus-tourism-dataset"  # fallback
)
MODEL_REPO_ID = os.environ.get(
    "HF_MODEL",
    "priyankamane10/visitwithus-tourism-model"    # fallback
)

def load_data():
    """Load processed train and test splits from Hugging Face dataset repo."""
    train_url = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main/processed/train.csv"
    test_url = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main/processed/test.csv"

    print(f"Loading train from: {train_url}")
    print(f"Loading test from: {test_url}")

    train = pd.read_csv(train_url)
    test = pd.read_csv(test_url)

    X_train = train.drop(columns=["ProdTaken"])
    y_train = train["ProdTaken"]

    X_test = test.drop(columns=["ProdTaken"])
    y_test = test["ProdTaken"]

    print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)
    return X_train, X_test, y_train, y_test


def build_pipeline(X_train):
    """Build preprocessing + model pipeline and grid search object."""
    # Identify columns
    categorical_cols = [
        "TypeofContact",
        "Occupation",
        "Gender",
        "ProductPitched",
        "MaritalStatus",
        "Designation",
    ]

    numeric_cols = [
        "Age",
        "CityTier",
        "DurationOfPitch",
        "NumberOfPersonVisiting",
        "NumberOfFollowups",
        "PreferredPropertyStar",
        "NumberOfTrips",
        "Passport",
        "PitchSatisfactionScore",
        "OwnCar",
        "NumberOfChildrenVisiting",
        "MonthlyIncome",
    ]

    # Safety check: only keep columns that exist
    categorical_cols = [c for c in categorical_cols if c in X_train.columns]
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]

    print("Categorical columns:", categorical_cols)
    print("Numeric columns:", numeric_cols)

    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numerical", StandardScaler(), numeric_cols),
        ]
    )

    clf = RandomForestClassifier(random_state=42)

    pipe = Pipeline(
        steps=[
            ("preprocessing", preprocess),
            ("clf", clf),
        ]
    )

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20],
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    return grid


def train_and_evaluate():
    """Train the model, evaluate, log metrics, save + upload best model."""
    X_train, X_test, y_train, y_test = load_data()
    grid = build_pipeline(X_train)

    print("Starting grid search training...")
    grid.fit(X_train, y_train)
    print("Grid search completed.")
    print("Best params:", grid.best_params_)

    # Evaluation
    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("ROC-AUC:", roc_auc)
    print("Confusion Matrix:\n", cm)

    # Prepare models folder
    os.makedirs("models", exist_ok=True)

    # Save best model
    model_path = "models/best_model.pkl"
    joblib.dump(grid.best_estimator_, model_path)
    print(f"Saved best model to {model_path}")

    # Save experiment log
    log_data = {
        "best_params": grid.best_params_,
        "roc_auc": float(roc_auc),
        "classification_report": clf_report,
        "confusion_matrix": cm,
    }

    log_path = "models/experiment_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)

    print(f"Saved experiment log to {log_path}")

    # Upload to Hugging Face model hub
    api = HfApi()

    print(f"Uploading best_model.pkl to model repo: {MODEL_REPO_ID}")
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_model.pkl",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="Upload best model from CI pipeline",
    )

    print(f"Uploading experiment_log.json to model repo: {MODEL_REPO_ID}")
    api.upload_file(
        path_or_fileobj=log_path,
        path_in_repo="experiment_log.json",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="Upload experiment log from CI pipeline",
    )

    print("Model training and upload completed successfully.")


if __name__ == "__main__":
    train_and_evaluate()
