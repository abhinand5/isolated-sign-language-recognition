import os
import numpy as np

from src.islr.data_processing.load import load_csv
from src.islr.data_processing.feature_extraction import get_all_feature_stats
from src.islr.training.trainer import train_model
from src.islr.training.cross_validation import get_fold_idx_map
from src.islr.logging import get_logger
from src.islr.utils import seed_it_all

logger = get_logger(__name__)


def run_training(config_dict, dry_run=False):
    general_conf = config_dict["general"]
    data_conf = config_dict["data"]
    train_conf = config_dict["train"]
    model_conf = config_dict["model"]

    seed_it_all(general_conf['SEED'])

    os.environ['LABEL_SMOOTHING'] = str(model_conf['LABEL_SMOOTHING'])

    # Load the dataframe and get fold indices
    df = load_csv(csv_path=os.path.join(general_conf["BASE_DATA_DIR"], "train.csv"))
    logger.info(f"Loaded train.csv | Length={len(df)}")

    logger.info("Getting fold indices map ...")
    is_single_fold = True if train_conf["K_FOLDS"] == 1 and train_conf["TRAIN_ON_ALL_DATA"] else False
    fold_ds_idx_map = get_fold_idx_map(
        df=df,
        k_folds=train_conf["K_FOLDS"],
        force_lh=False,
        seed=general_conf["SEED"],
        all_data=train_conf["TRAIN_ON_ALL_DATA"],
        val_ratio=train_conf["VAL_RATIO"],
    )

    logger.info("Loading Numpy data ... ")
    data = {}
    data["X"] = np.load(os.path.join(general_conf["BASE_DATA_DIR"], "X.npy"))
    data["y"] = np.load(os.path.join(general_conf["BASE_DATA_DIR"], "y.npy"))
    data["NON_EMPTY_FRAME_IDXS"] = np.load(
        os.path.join(general_conf["BASE_DATA_DIR"], "NON_EMPTY_FRAME_IDXS.npy")
    )

    if train_conf['APPLY_AUGMENTATION']:
        logger.info("Enabled Augmentation... Setting X_aug ...")
        data["X_aug"] = np.load(os.path.join(general_conf["AUGMENT_DATA_DIR"], "X.npy"))
    else:
        data["X_aug"] = None

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
        verbose=general_conf["VERBOSE"],
        train_on_all_data=is_single_fold,
        dry_run=dry_run,
    )

    return histories
