import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from wandb.keras import WandbCallback, WandbModelCheckpoint

from src.islr.data_processing.load import load_csv
from src.islr.data_processing.feature_extraction import get_all_feature_stats
from src.islr.training.cross_validation import get_fold_idx_map, get_fold_data
from src.islr.logging import get_logger
from src.islr.models.custom_transformer import get_model
from src.islr.training.utils import get_loss_fn, get_train_batch_all_signs
from src.islr.data_processing.feature_extraction import extra_features
from src.islr.training.distiller import Distiller, SaveBestStudent
from src.islr.models.callbacks import WeightDecayCallback

logger = get_logger(__name__)


def run_distill_training(config_dict):
    general_conf = config_dict["general"]
    data_conf = config_dict["data"]
    train_conf = config_dict["train"]
    model_conf = config_dict["model"]
    kd_conf = config_dict["knowledge_distillation"]

    # Load the dataframe and get fold indices
    df = load_csv(csv_path=os.path.join(general_conf["BASE_DATA_DIR"], "train.csv"))
    logger.info(f"Loaded train.csv | Length={len(df)}")

    logger.info("Getting fold indices map ...")
    fold_ds_idx_map = get_fold_idx_map(
        df=df, k_folds=kd_conf["K_FOLDS"], force_lh=False, seed=general_conf["SEED"]
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
                \n\tkd_config={kd_conf}"
    )

    tf.keras.backend.clear_session()

    teacher_model = get_model(data_conf, model_conf, data["feature_stats"])
    teacher_model.load_weights(
        os.path.join(model_conf["MODEL_DIR"], kd_conf["TEACHER_WEIGHTS_PATH"])
    )

    student_model = get_model(data_conf, kd_conf, data["feature_stats"])

    logger.info("Successfully created TEACHER and STUDENT models!")
    logger.info(f"Teacher Model Total Params: {teacher_model.count_params()}")
    logger.info(f"Student Model Total Params: {student_model.count_params()}")

    loss_fn = get_loss_fn(kd_conf)

    logger.info("Validating the Teacher Model ...")
    (
        X_train,
        y_train,
        non_empty_frame_idxs_train,
        X_val,
        y_val,
        non_empty_frame_idxs_val,
    ) = get_fold_data(
        data["X"],
        data["y"],
        kd_conf["TEACHER_FOLD_NUM"],
        fold_ds_idx_map,
        data["NON_EMPTY_FRAME_IDXS"],
    )

    eval_result = teacher_model.evaluate(
        {"frames": X_val, "non_empty_frame_idxs": non_empty_frame_idxs_val},
        y_val,
        verbose=1,
    )
    logger.info(f"Evaluation Result:\n{eval_result}")
    logger.info(f"Teacher evaluation complete!")

    logger.info("Creating the Distiller ...")

    distiller = Distiller(student=student_model, teacher=teacher_model)

    distiller.compile(
        # optimizer=tfa.optimizers.AdamW(
        #     learning_rate=kd_conf["INIT_LR"],
        #     weight_decay=kd_conf["WT_DECAY"],
        #     clipnorm=1.0,
        # ),
        optimizer=tf.keras.optimizers.Adam(learning_rate=kd_conf['INIT_LR']),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        student_loss_fn=loss_fn,
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=kd_conf["ALPHA"],
        temperature=kd_conf["TEMPERATURE"],
    )

    wandb.init(
        entity="abhinand",
        project="Google-ISLR-kaggle",
        job_type="train",
        name=f"distill_{kd_conf['MODEL_ID']}",
    )

    logger.info("TRAINING BEGIN: KNOWLEDGE DISTILLATION")

    distiller.fit(
        x=get_train_batch_all_signs(
            X_train,
            y_train,
            non_empty_frame_idxs_train,
            train_conf["BATCH_ALL_SIGNS_N"],
            data_conf,
            extra_features.N_COLS,
        ),
        validation_data=(
            {"frames": X_val, "non_empty_frame_idxs": non_empty_frame_idxs_val},
            y_val,
        ),
        steps_per_epoch=len(X_train)
        // (data_conf["NUM_CLASSES"] * train_conf["BATCH_ALL_SIGNS_N"]),
        batch_size=train_conf["BATCH_SIZE"],
        epochs=kd_conf["STUDENT_TRAIN_EPOCHS"],
        verbose=general_conf["VERBOSE"],
        callbacks=[
            # WeightDecayCallback(wd_ratio=kd_conf["WT_DECAY"]),
            WandbCallback(save_model=False, monitor="val_acc", mode="max"),
            SaveBestStudent(
                filepath=f"{kd_conf['MODEL_DIR']}/{kd_conf['MODEL_ID']}_distill.h5",
                verbose=1,
                save_best_only=True,
            ),
        ],
    )

    logger.info("TRAINING END: KNOWLEDGE DISTILLATION")
