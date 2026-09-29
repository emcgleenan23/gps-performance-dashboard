import os
import yaml

def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # optional override, e.g. when the drive is mounted somewhere else
    if os.environ.get("GPS_DATA_ROOT"):
        cfg["data"]["root"] = os.environ["GPS_DATA_ROOT"]
    return cfg
