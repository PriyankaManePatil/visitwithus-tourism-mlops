from huggingface_hub import upload_folder

SPACE_REPO = "priyankamane10/visitwithus-tourism-app"

upload_folder(
    folder_path="app",
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo=".",
    commit_message="Deploying Streamlit App via Automation"
)

print("Deployment completed.")
