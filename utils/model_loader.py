import os
from huggingface_hub import hf_hub_download
import config


def get_model_path():
    local_path = os.path.join(os.path.dirname(__file__), config.MODEL_FILENAME)
    if os.path.exists(local_path):
        return local_path
    return hf_hub_download(repo_id=config.HF_REPO_ID, filename=config.MODEL_FILENAME)