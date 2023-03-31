import os
import numpy as np
import tensorflow as tf

from src.islr.data_processing.load import load_csv
from src.islr.training.cross_validation import get_fold_idx_map, get_val_of_fold
from src.islr.models.custom_transformer import get_model
from src.islr.data_processing.feature_extraction import get_all_feature_stats
from src.islr.config import METRICS_LIST
from src.islr.logging import get_logger

def run_eval_on_oof(config_dict, fold_num, weights_path):
    general_conf = config_dict["general"]
    data_conf = config_dict["data"]
    train_conf = config_dict["train"]
    model_conf = config_dict["model"]

    logger = get_logger(__name__)

    # Load the dataframe and get fold indices
    df = load_csv(csv_path=os.path.join(general_conf["BASE_DATA_DIR"], "train.csv"))
    logger.info(f"Loaded train.csv | Length={len(df)}")

    logger.info("Getting fold indices map ...")
    fold_ds_idx_map = get_fold_idx_map(
        df=df, k_folds=train_conf["K_FOLDS"], force_lh=False, seed=general_conf["SEED"]
    )

    logger.info("Loading Numpy data ... ")
    X = np.load(os.path.join(general_conf["BASE_DATA_DIR"], "X.npy"))
    y = np.load(os.path.join(general_conf["BASE_DATA_DIR"], "y.npy"))
    NON_EMPTY_FRAME_IDXS = np.load(
        os.path.join(general_conf["BASE_DATA_DIR"], "NON_EMPTY_FRAME_IDXS.npy")
    )
    logger.info("Successfully loaded Numpy data!")

    logger.info(f"Loading validation data of fold {fold_num} ...")
    X_val, y_val, NON_EMPTY_FRAME_IDXS_VAL = get_val_of_fold(
        X, y, fold_num, fold_ds_idx_map, NON_EMPTY_FRAME_IDXS
    )
    logger.info(f"Loaded validation data of fold {fold_num}!")

    logger.info(f"Computing feature statistics ...")
    lips_stats, pose_stats, hand_stats = get_all_feature_stats(X, data_conf["N_DIMS"])

    feature_stats = {
        "LIPS_MEAN": lips_stats["mean"],
        "LIPS_STD": lips_stats["std"],
        "POSE_MEAN": pose_stats["mean"],
        "POSE_STD": pose_stats["std"],
        "LEFT_HANDS_MEAN": hand_stats["lh_mean"],
        "LEFT_HANDS_STD": hand_stats["lh_std"],
        "RIGHT_HANDS_MEAN": hand_stats["rh_mean"],
        "RIGHT_HANDS_STD": hand_stats["rh_std"],
    }
    logger.info("Successfully computed feature statistics!")

    logger.info("Loading the Custom Transformer Model ...")
    model = get_model(data_conf, model_conf, feature_stats)
    logger.info(f"Loading the Custom Transformer Model weights from {weights_path}...")
    model.load_weights(weights_path)

    logger.info("Loaded the model. Starting Evaluation!")    

    metrics = model.evaluate(
        {"frames": X_val, "non_empty_frame_idxs": NON_EMPTY_FRAME_IDXS_VAL},
        y_val,
        verbose=1,
    )

    return dict(zip(METRICS_LIST, metrics))
