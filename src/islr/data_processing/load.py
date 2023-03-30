import numpy as np
import pandas as pd
from tqdm import tqdm


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df


def load_relevant_data_subset(pq_path, rows_per_frame):
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / rows_per_frame)
    data = data.values.reshape(n_frames, rows_per_frame, len(data_columns))
    return data.astype(np.float32)


# Get the full dataset
def get_np_data_from_df(data_config, train_df, save=True):
    # Create arrays to save data
    X = np.zeros(
        [
            data_config["N_SAMPLES"],
            data_config["INPUT_SIZE"],
            data_config["N_COLS"],
            data_config["N_DIMS"],
        ],
        dtype=np.float32,
    )
    y = np.zeros([data_config["N_SAMPLES"]], dtype=np.int32)
    NON_EMPTY_FRAME_IDXS = np.full(
        [data_config["N_SAMPLES"], data_config["INPUT_SIZE"]], -1, dtype=np.float32
    )

    for row_idx, (file_path, sign_ord) in enumerate(
        tqdm(train_df[["file_path", "sign_ord"]].values)
    ):
        if row_idx % 5000 == 0:
            print(f"Generated {row_idx}/{data_config['N_SAMPLES']}")

        data, non_empty_frame_idxs = load_parquet_data(file_path)
        X[row_idx] = data
        y[row_idx] = sign_ord
        NON_EMPTY_FRAME_IDXS[row_idx] = non_empty_frame_idxs
        if np.isnan(data).sum() > 0:
            print(row_idx)
            return data

    if save:
        # Save X/y
        np.save("X.npy", X)
        np.save("y.npy", y)
        np.save("NON_EMPTY_FRAME_IDXS.npy", NON_EMPTY_FRAME_IDXS)

    return X, y, NON_EMPTY_FRAME_IDXS


def load_parquet_data(file_path, preprocess_layer):
    # Load Raw Data
    data = load_relevant_data_subset(file_path)
    # Process Data Using Tensorflow
    data = preprocess_layer(data)

    return data
