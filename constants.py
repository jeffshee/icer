import os
import json
from pprint import pprint
from argparse import Namespace

CURRENT_CONFIG_VERSION = 1


def load_config():
    if os.path.isfile("config.json"):
        # Read Config
        with open("config.json", "r") as f:
            config = json.load(f)
            if config.get("CONFIG_VERSION", -1) < CURRENT_CONFIG_VERSION:
                print("[Config] Incompatible configuration. A new one will be generated.")
                return False, None
            print("[Config] Current configuration")
            pprint(config)
            return True, config
    print("[Config] Configuration not found. A new one will be generated.")
    return False, None


def generate_config():
    config = dict()
    config["CONFIG_VERSION"] = CURRENT_CONFIG_VERSION  # For checking the version of config.json
    # Run mode
    config["RUN_CAPTURE_FACE"] = True
    config["RUN_EMOTION_RECOGNITION"] = True
    config["RUN_TRANSCRIPT"] = True
    config["RUN_OVERLAY"] = True

    # Settings
    config["NUM_GPU"] = 1
    config["TF_LOG_LEVEL"] = 2
    config["DEBUG"] = False

    # Write Default config
    with open("config.json", "w") as f:
        pprint(config)
        json.dump(config, f, indent=3)
    return config


ret, config = load_config()
if not ret:
    config = generate_config()

# Append config to namespace
constants = Namespace(**config)
