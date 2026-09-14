import json
import os
import torch

# Mod these as variables as needed/wanted. These are the items used for training
WINDOW_SIZE = (512, 512)
STRIDE = 256
PIXEL_THRESHOLD = 0.25
SPHERICITY_THRESHOLD = 0.65
CELL_AREA_THRESHOLD = 60 # This mostly meant for removing noise, but it can be used for other things, too
BATCH_SIZE = 16

SCALING = 0.75
USE_SCALING = False

# Set HF_TOKEN env variable to bypass rate limits. That being said, with rate limits this should be a 6-ish second download on good internet
HF_REPO_ID = "skyle123/Adipolabeler"
MODEL_FILENAME = "unet_ft_adipocytes.pth"

def _detect_batch_size():
    if torch.cuda.is_available():
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_memory_gb >= 8:
            return 32
        if total_memory_gb >= 4:
            return 16
        return 8
    cpu_count = os.cpu_count() or 4
    return max(1, min(8, cpu_count // 2))
 
 
def _configure_cpu_threads():
    if not torch.cuda.is_available():
        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(max(1, cpu_count - 1))
 
 
_configure_cpu_threads()
BATCH_SIZE = _detect_batch_size()
 
_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "user_settings.json")
 
if os.path.exists(_SETTINGS_PATH):
    with open(_SETTINGS_PATH) as f:
        _user_settings = json.load(f)
    SCALING = _user_settings.get("scaling", SCALING)
    USE_SCALING = _user_settings.get("use_scaling", USE_SCALING)
    SPHERICITY_THRESHOLD = _user_settings.get("sphericity_threshold", SPHERICITY_THRESHOLD)
    CELL_AREA_THRESHOLD = _user_settings.get("cell_area_threshold", CELL_AREA_THRESHOLD)
    if _user_settings.get("batch_size", 0):
        BATCH_SIZE = int(_user_settings["batch_size"])
