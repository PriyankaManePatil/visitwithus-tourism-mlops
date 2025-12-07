import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Read dataset repo ID from environment variable (set in GitHub Actions)
DATASET_REPO_ID = os.environ.get(
    "HF_DATASET",
    "priyankamane10/visitwithus-tourism-dataset"  # fallback if env not set
)

def main():
    # 1. Load raw dataset directly from Hugging Face
    #    Make sure the filename matches what you uploaded (e.g., raw_tourism.csv)
    raw_filename = "tourism.csv"  # change if your raw file name is different

    raw_url = (
        f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main/{raw_filename}"
    )

    print(f"Loading raw data from: {raw_url}")
    df = pd.read_csv(raw_url)

    # 2. Drop unnecessary columns
    cols_to_drop = ["Unnamed: 0", "CustomerID"]
    df = df.drop(columns=cols_to_drop)

    # 3. Split into features and target
    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]

    # 4. Train–test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train = X_train.copy()
    train["ProdTaken"] = y_train

    test = X_test.copy()
    test["ProdTaken"] = y_test

    # Ensure local data folder exists
    os.makedirs("data", exist_ok=True)

    train_path = "data/train.csv"
    test_path = "data/test.csv"

    # 5. Save locally
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Saved train data to {train_path} with shape {train.shape}")
    print(f"Saved test data to {test_path} with shape {test.shape}")

    # 6. Upload processed train/test back to Hugging Face dataset repo
    api = HfApi()

    print(f"Uploading train.csv to {DATASET_REPO_ID} at processed/train.csv")
    api.upload_file(
        path_or_fileobj=train_path,
        path_in_repo="processed/train.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message="Upload processed training data from CI pipeline"
    )

    print(f"Uploading test.csv to {DATASET_REPO_ID} at processed/test.csv")
    api.upload_file(
        path_or_fileobj=test_path,
        path_in_repo="processed/test.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message="Upload processed testing data from CI pipeline"
    )

    print("Data preparation and upload completed successfully.")

if __name__ == "__main__":
    main()
