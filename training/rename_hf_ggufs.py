"""
Rename the 3 GGUF files in Efso/gemma-4-E4B-it-GR-v2 to follow the
gemma-4-e4b-it-gr-v2 naming convention. Server-side — no re-upload needed.
"""
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationDelete

load_dotenv()

REPO_ID = "Efso/gemma-4-E4B-it-GR-v2"
TOKEN = os.getenv("HF_TOKEN")

RENAMES = [
    ("gemma4gr-e4b-v2-q4_k_m.gguf", "gemma-4-e4b-it-gr-v2-Q4_K_M.gguf"),
    ("gemma4gr-e4b-v2-q8_0.gguf",   "gemma-4-e4b-it-gr-v2-Q8_0.gguf"),
    ("gemma4gr-e4b-v2-mmproj.gguf", "gemma-4-e4b-it-gr-v2-mmproj.gguf"),
]

api = HfApi(token=TOKEN)

for old, new in RENAMES:
    print(f"  {old} -> {new}")
    api.create_commit(
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"Rename {old} to {new}",
        operations=[
            CommitOperationCopy(src_path_in_repo=old, path_in_repo=new),
            CommitOperationDelete(path_in_repo=old),
        ],
    )
    print(f"  OK")

print("\nAll renames complete.")
