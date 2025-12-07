import os
from huggingface_hub import upload_folder, HfApi

SPACE_REPO_ID = os.environ.get(
    "HF_SPACE",
    "priyankamane10/visitwithus-tourism-app"  # fallback
)

def deploy():
    api = HfApi()

    print(f"Deploying app folder to Hugging Face Space: {SPACE_REPO_ID}")

    upload_folder(
        folder_path="app",
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        path_in_repo=".",  # root of space
        commit_message="Automated deployment from GitHub Actions CI/CD pipeline"
    )

    print("Deployment successful.")

if __name__ == "__main__":
    deploy()
