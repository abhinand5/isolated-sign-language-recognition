import os
import sys
import argparse
import datetime
import wandb

from src.islr.config import load_config

# from src.src.islr.logging import get_logger
from src.islr.train import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISLR CLI")
    parser.add_argument("mode", type=str, help="Mode of operation - train or inference")
    args = parser.parse_args()

    MODE = args.mode

    if MODE == "train":
        config_dict = load_config("./conf/train_config.yml")
    elif MODE == "inference":
        config_dict = load_config("./conf/inference_config.yml")
    else:
        raise ValueError(
            f"Invalid value set for MODE - [{MODE}] \
            Possible values are 'train' and 'inference' "
        )

    secrets = load_config("./conf/secrets.yml")
    wandb.login(key=secrets["WANDB_API_KEY"])

    # now = datetime.datetime.now()
    # timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
    # log_file_path = os.path.join(
    #     config_dict["general"]["LOG_DIR"], f"applog-{timestamp}.log"
    # )
    # logger = get_logger(__name__, log_file_path)

    run_training(config_dict)
