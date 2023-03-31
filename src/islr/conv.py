import os
import numpy as np
import tensorflow as tf

from src.islr.models.custom_transformer import get_model
from src.islr.data_processing.feature_extraction import get_all_feature_stats
from src.islr.logging import get_logger
from src.islr.models.tf_lite import convert_to_tflite


def run_tflite_conversion(config_dict, weights_path, **kwargs):
    general_conf = config_dict["general"]
    data_conf = config_dict["data"]
    # train_conf = config_dict["train"]
    model_conf = config_dict["model"]

    logger = get_logger(__name__)

    # Load the dataframe and get fold indices
    logger.info("Loading Numpy data [X.npy] ... ")
    X = np.load(os.path.join(general_conf["BASE_DATA_DIR"], "X.npy"))
    logger.info("Successfully loaded Numpy data!")

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
    logger.info(
        f"Loading the Custom Transformer Model [Keras] weights from {weights_path}..."
    )
    model.load_weights(weights_path)

    logger.info("Loaded the model. Starting Conversion ...")

    logger.info(f"Beginning conversion with the following settings: \n\t{kwargs}")
    convert_to_tflite(
        name=kwargs["name"],
        tflite_keras_model=model,
        model_dir=kwargs["dest_dir"],
        quantize_model=kwargs["quantize"],
        quant_method=kwargs["quant_method"],
    )

    logger.info(" END ")
