import os
import sys
import argparse
import datetime
import wandb

from src.islr.config import load_config
from src.islr.logging import get_logger
from src.islr.datagen import run_datagen
from src.islr.train import run_training
from src.islr.distill import run_distill_training
from src.islr.eval import run_eval_on_oof
from src.islr.conv import run_tflite_conversion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISLR CLI")
    subparsers = parser.add_subparsers(
        help="Mode of operation - [train, distill, eval, tflite-convert]", dest="mode"
    )

    data_parser = subparsers.add_parser("datagen")
    data_parser.add_argument(
        "--save",
        dest="save_data",
        # type=bool,
        action="store_true",
        help="Setting this flag will save the data in the location set in config file",
        # required=False,
        # default=False,
    )

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--dry-run",
        dest="dry_run",
        # type=bool,
        action="store_true",
        help="Dry run exits just before model.fit()",
        # required=False,
        # default=False,
    )
    train_parser.add_argument(
        "--save-feature-stats",
        dest="save_feature_stats",
        # type=bool,
        action="store_true",
        help="Saves the feature stats (used for normalization and preprocessing) in a pickle dump",
        # required=False,
        # default=False,
    )

    distill_parser = subparsers.add_parser("distill")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument(
        "--fold-num",
        dest="fold_num",
        type=int,
        help="The OOF fold on which to evaluate the model",
        required=True,
    )
    eval_parser.add_argument(
        "--weights-path",
        dest="weights_path",
        type=str,
        help="The weights (*.h5) of the model to load and evaluate",
        required=True,
    )

    convert_parser = subparsers.add_parser("tflite-convert")
    convert_parser.add_argument(
        "--input",
        type=str,
        help="Path of the Keras model which has to be converted (*.h5)",
        required=True,
    )
    convert_parser.add_argument(
        "--dest-dir",
        dest="dest_dir",
        type=str,
        help="Destination directory where the converted tflite model has to be saved",
        required=False,
    )
    convert_parser.add_argument(
        "--quantize",
        type=bool,
        help="Where or not to quantize the model as part of conversion",
        required=False,
        default=False,
    )
    convert_parser.add_argument(
        "--quantize-method",
        dest="quantize_method",
        choices=["dynamic", "float16"],
        help="Quantization method - dynamic is default",
        required=False,
        default="dynamic",
    )

    args = parser.parse_args()

    secrets = load_config("./conf/secrets.yml")

    logger = get_logger(__name__)

    if args.mode == "datagen":
        config_dict = load_config("./conf/train_config.yml")
        save_data = args.save_data
        run_datagen(config_dict, save_data=save_data)

    elif args.mode == "train":
        config_dict = load_config("./conf/train_config.yml")
        dry_run = args.dry_run
        save_feature_stats = args.save_feature_stats
        if not dry_run:
            wandb.login(key=secrets["WANDB_API_KEY"])
        run_training(config_dict, dry_run, save_feature_stats)

    elif args.mode == "eval":
        wandb.init(mode="disabled")
        config_dict = load_config("./conf/eval_config.yml")

        fold_num = args.fold_num
        weights_path = args.weights_path

        metrics = run_eval_on_oof(config_dict, fold_num, weights_path)
        logger.info(f"Evaluation Result:\n{metrics}")

    elif args.mode == "distill":
        config_dict = load_config("./conf/train_config.yml")
        wandb.login(key=secrets["WANDB_API_KEY"])

        run_distill_training(config_dict)

    elif args.mode == "tflite-convert":
        config_dict = load_config("./conf/train_config.yml")
        source_model = args.input
        dest_dir = args.dest_dir
        quantize = args.quantize
        quant_method = args.quantize_method

        if dest_dir == None:
            dest_dir = config_dict["model"]["MODEL_DIR"]

        name = os.path.basename(source_model).split(".")[0]

        run_tflite_conversion(
            config_dict=config_dict,
            weights_path=source_model,
            name=name,
            dest_dir=dest_dir,
            quantize=quantize,
            quant_method=quant_method,
        )

    elif args.mode == "inference":
        wandb.init(mode="disabled")
        config_dict = load_config("./conf/inference_config.yml")

    else:
        raise ValueError(
            f"Invalid value set for MODE - [{args.mode}] \
            Possible values are 'train' and 'inference' "
        )
