import os
import gc
import sys
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback, WandbModelCheckpoint

from src.islr.utils import print_shape_dtype
from src.islr.models.custom_transformer import get_model
from src.islr.training.utils import get_train_batch_all_signs
from src.islr.models.callbacks import WeightDecayCallback
from src.islr.data_processing.feature_extraction import extra_features
from src.islr.logging import get_logger

logger = get_logger(__name__)


def train_model(
    exp_id,
    data,
    model_config,
    train_config,
    fold_ds_idx_map,
    num_classes=250,
    verbose=1,
    train_on_all_data=False,
    dry_run=False,
):
    # X, y, NON_EMPTY_FRAME_IDXS = data.values()
    X = data["X"]
    y = data["y"]
    X_aug = data["X_aug"]
    NON_EMPTY_FRAME_IDXS = data["NON_EMPTY_FRAME_IDXS"]
    data_config = data["config"]

    histories = []

    if not os.path.isdir(model_config["MODEL_DIR"]):
        os.makedirs(model_config["MODEL_DIR"])

    for fold_num, fold_idxs in fold_ds_idx_map.items():
        logger.info(f"Cur Fold -> {fold_num}")
        if train_on_all_data or fold_num in train_config["FOLDS_TO_TRAIN"]:
            logger.info(f"================ FOLD {fold_num} - START =================")

            X_val, y_val = X[fold_idxs["val"]], y[fold_idxs["val"]]
            X_train, y_train = X[fold_idxs["train"]], y[fold_idxs["train"]]

            NON_EMPTY_FRAME_IDXS_TRAIN = NON_EMPTY_FRAME_IDXS[fold_idxs["train"]]
            NON_EMPTY_FRAME_IDXS_VAL = NON_EMPTY_FRAME_IDXS[fold_idxs["val"]]

            print_shape_dtype(
                [X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN],
                ["X_train", "y_train", "NON_EMPTY_FRAME_IDXS_TRAIN"],
            )
            print_shape_dtype(
                [X_val, y_train, NON_EMPTY_FRAME_IDXS_VAL],
                ["X_val", "y_val", "NON_EMPTY_FRAME_IDXS_VAL"],
            )

            # Clear all models in GPU
            tf.keras.backend.clear_session()

            # Get new fresh model
            # first_decay_steps = (
            #     X_train.shape[0] // train_config.BATCH_SIZE
            # ) * train_config.N_WARMUP_EPOCHS
            # logger.info(f"first_decay_steps={first_decay_steps}")
            model = get_model(data_config, model_config, data["feature_stats"])

            logger.info(f"Loaded Model!")
            logger.info(f"Total Model Params = {model.count_params()}")

            if dry_run:
                logger.info(f"[DRY RUN] Exiting... ")
                sys.exit(0)

            wandb.init(
                entity="abhinand",
                project="Google-ISLR-kaggle",
                job_type="train",
                name=f"{exp_id}_fold{fold_num}",
            )

            if train_config["LOAD_MODELS"]:
                if (
                    train_config["LOAD_MODELS_MAP"]
                    and train_config["LOAD_MODELS_MAP"][fold_num]
                ):
                    artifact_dir = wandb.use_artifact(
                        train_config["LOAD_MODELS_MAP"][fold_num]
                    ).download()
                    model.load_weights(artifact_dir)
                else:
                    logger.info(f"FAILED LOADING MODEL FOR FOLD {fold_num}")
                    break

            # Sanity Check
            # model.summary()

            # early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            #     patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=VERBOSE, monitor=CB_MONITOR)

            # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            #                                 filepath=os.path.join(wandb.run.dir, "best-model.h5"),
            #                                 save_weights_only=True,
            #                                 monitor='val_acc',
            #                                 mode='max',
            #                                 save_best_only=True)

            # Actual Training

            if not train_on_all_data:
                validation_data = (
                    {"frames": X_val, "non_empty_frame_idxs": NON_EMPTY_FRAME_IDXS_VAL},
                    y_val,
                )
            else:
                validation_data = None
            
            if X_aug is not None:
                logger.info(f"Data Augmentation has been set ... aug_prob = {train_config['AUG_PROB']}")

            history = model.fit(
                x=get_train_batch_all_signs(
                    X_train,
                    y_train,
                    NON_EMPTY_FRAME_IDXS_TRAIN,
                    train_config["BATCH_ALL_SIGNS_N"],
                    data_config,
                    extra_features.N_COLS,
                    X_aug=X_aug,
                    aug_prob=train_config["AUG_PROB"],
                ),
                steps_per_epoch=len(X_train)
                // (num_classes * train_config["BATCH_ALL_SIGNS_N"]),
                epochs=train_config["N_EPOCHS"],
                # Only used for validation data since training data is a generator
                batch_size=train_config["BATCH_SIZE"],
                validation_data=validation_data,
                callbacks=[
                    WeightDecayCallback(wd_ratio=model_config["WT_DECAY"]),
                    # early_stopping_cb,
                    WandbCallback(
                        save_model=False,
                        monitor=model_config["CB_MONITOR"],
                        mode=model_config["CB_MONITOR_MODE"],
                    ),
                    WandbModelCheckpoint(
                        f"{model_config['MODEL_DIR']}/{exp_id}-model-fold{fold_num}-best.h5",
                        monitor=model_config["CB_MONITOR"],
                        save_best_only=True,
                        save_weights_only=True,
                        mode=model_config["CB_MONITOR_MODE"],
                    ),
                ],
                verbose=verbose,
            )

            histories.append(history)
            # Save

            model.load_weights(f"{model_config['MODEL_DIR']}/{exp_id}-model-fold{fold_num}-best.h5")
            score = model.evaluate(
                {"frames": X_val, "non_empty_frame_idxs": NON_EMPTY_FRAME_IDXS_VAL},
                y_val,
                verbose=0,
            )

            wandb.log({"val acc": score[1]})
            # model.save_weights(
            #     os.path.join(
            #         model_config["MODEL_DIR"], f"islr_model__fold_{fold_num}__v2.h5"
            #     )
            # )

            gc.collect()
            del (
                model,
                X_train,
                y_train,
                X_val,
                y_val,
                NON_EMPTY_FRAME_IDXS_TRAIN,
                NON_EMPTY_FRAME_IDXS_VAL,
            )
            gc.collect()

            logger.info(f"... Logging best model artifact in WANDB ...")
            artifact = wandb.Artifact(
                f"model-gislr_{exp_id}_fold{fold_num}", type="model"
            )
            rounded_score = f"{round(score[1], 4)}".replace(".", "")
            artifact.add_file(
                f"{model_config['MODEL_DIR']}/{exp_id}-model-fold{fold_num}-best.h5",
                f"model-gislr_{exp_id}_fold{fold_num}_best_{rounded_score}.h5",
            )
            wandb.log_artifact(artifact)

            wandb.join()
            logger.info(f"================ FOLD {fold_num} - END =================\n\n")
        else:
            gc.collect()
            logger.info(f"\tSkipping FOLD {fold_num}")

    return histories
