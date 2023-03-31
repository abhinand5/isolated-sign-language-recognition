import os
import numpy as np

from src.islr.data_processing.load import load_csv
from src.islr.data_processing.feature_extraction import get_all_feature_stats
from src.islr.training.trainer import train_model
from src.islr.training.cross_validation import get_fold_idx_map
from src.islr.logging import get_logger

logger = get_logger(__name__)


def run_training(config_dict):
    general_conf = config_dict["general"]
    data_conf = config_dict["data"]
    train_conf = config_dict["train"]
    model_conf = config_dict["model"]

    # Load the dataframe and get fold indices
    df = load_csv(csv_path=os.path.join(general_conf["BASE_DATA_DIR"], "train.csv"))
    logger.info(f"Loaded train.csv | Length={len(df)}")

    logger.info("Getting fold indices map ...")
    fold_ds_idx_map = get_fold_idx_map(
        df=df, k_folds=train_conf["K_FOLDS"], force_lh=False, seed=general_conf["SEED"]
    )

    logger.info("Loading Numpy data ... ")
    data = {}
    data["X"] = np.load(os.path.join(general_conf["BASE_DATA_DIR"], "X.npy"))
    data["y"] = np.load(os.path.join(general_conf["BASE_DATA_DIR"], "y.npy"))
    data["NON_EMPTY_FRAME_IDXS"] = np.load(
        os.path.join(general_conf["BASE_DATA_DIR"], "NON_EMPTY_FRAME_IDXS.npy")
    )
    data["config"] = data_conf
    logger.info("Successfully loaded Numpy data!")

    logger.info(f"Computing feature statistics ...")
    lips_stats, pose_stats, hand_stats = get_all_feature_stats(
        data["X"], data_conf["N_DIMS"]
    )
    logger.info("Successfully computed feature statistics!")

    data["feature_stats"] = {
        "LIPS_MEAN": lips_stats["mean"],
        "LIPS_STD": lips_stats["std"],
        "POSE_MEAN": pose_stats["mean"],
        "POSE_STD": pose_stats["std"],
        "LEFT_HANDS_MEAN": hand_stats["lh_mean"],
        "LEFT_HANDS_STD": hand_stats["lh_std"],
        "RIGHT_HANDS_MEAN": hand_stats["rh_mean"],
        "RIGHT_HANDS_STD": hand_stats["rh_std"],
    }

    logger.info(
        f"Begin model training for\n\texp={general_conf['EXP_ID']} \
                \n\tmodel_config={model_conf} \
                \n\ttrain_config={train_conf}"
    )

    histories = train_model(
        exp_id=general_conf["EXP_ID"],
        data=data,
        model_config=model_conf,
        train_config=train_conf,
        fold_ds_idx_map=fold_ds_idx_map,
        num_classes=data_conf["NUM_CLASSES"],
    )

    return histories
