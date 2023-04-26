import argparse
from huggingface_hub import login, create_repo, HfApi

login(token="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")
api = HfApi()
def push(repo_id, local_dir):
    create_repo(repo_id, private=False, exist_ok=True)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model"
    )


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo_id", default=None, type=str, required=True, help="repo_id"
    )
    parser.add_argument(
        "--model_dir", default=None, type=str, required=True, help="converted model dir to upload"
    )

    args = parser.parse_args()
    push(args.repo_id, args.model_dir)