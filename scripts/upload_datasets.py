import os
import sys
from huggingface_hub import HfApi

def upload():
    token = os.environ.get("HF_TOKEN")
    if not token:
        # Prompt user if token is not in env
        token = input("Please enter your Hugging Face write token: ").strip()
    if not token:
        print("Error: No token provided.")
        sys.exit(1)
        
    repo_id = "rahulhirur/TheTravellingSalesman-Model"
    folder_path = os.path.join("misc", "data")
    
    if not os.path.exists(folder_path):
        print(f"Error: Local directory {folder_path} not found.")
        sys.exit(1)
        
    print(f"Uploading {folder_path} to HF Repository {repo_id}/data ...")
    try:
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=folder_path,
            path_in_repo="data",
            repo_id=repo_id,
            repo_type="model"
        )
        print("\n[SUCCESS] Datasets uploaded successfully to Hugging Face Model Repository!")
    except Exception as e:
        print(f"\n[FAILURE] Upload failed: {e}")

if __name__ == "__main__":
    upload()
