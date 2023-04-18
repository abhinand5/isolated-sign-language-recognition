import os

from src.islr.data_processing.preprocessing import PreprocessLayer
from src.islr.data_processing.load import load_csv, get_np_data_from_df
from src.islr.utils import print_shape_dtype
from src.islr.logging import get_logger
from src.islr.data_processing.feature_extraction import extra_features

# Best to run this on Kaggle because of the huge dataset size
def run_datagen(config_dict, save_data=True):
    general_conf = config_dict["general"]
    data_conf = config_dict["data"]

    logger = get_logger(__name__)

    df = load_csv(csv_path=os.path.join(general_conf["BASE_DATA_DIR"], "train.csv"))
    df['file_path'] = df['path'].apply(lambda path: os.path.join(general_conf["BASE_DATA_DIR"], path))
    df['sign_ord'] = df['sign'].astype('category').cat.codes

    logger.info(f"Loaded train.csv | Length={len(df)}")
    logger.info(f"Train DF INFO: \n {df.info()}")

    data_conf['N_SAMPLES'] = len(df)
    data_conf['N_COLS'] = extra_features.LANDMARK_IDXS0.size

    preprocess_layer = PreprocessLayer(
        input_size=data_conf["INPUT_SIZE"], n_dims=data_conf["N_DIMS"]
    )

    logger.info("Running Dataloader and Pre-Processor!")

    X, y, non_empty_frame_idxs = get_np_data_from_df(
        data_conf, preprocess_layer, df, save=save_data
    )

    logger.info("Successfully Loaded Data!")

    print_shape_dtype([X, y, non_empty_frame_idxs], ["X", "y", "non_empty_frame_idxs"])
