import numpy as np
import tensorflow as tf
from src.islr.models.utils import SparseCategoricalCrossentropyLS

# Custom sampler to get a batch containing N times all signs
def get_train_batch_all_signs(X, y, non_empty_frame_idxs, n_signs, data_config, n_cols):
    # Arrays to store batch in
    X_batch = np.zeros(
        [data_config['NUM_CLASSES'] * n_signs, data_config['INPUT_SIZE'], n_cols, data_config['N_DIMS']],
        dtype=np.float32,
    )
    y_batch = np.arange(
        0, data_config['NUM_CLASSES'], step=1 / n_signs, dtype=np.float32
    ).astype(np.int64)
    non_empty_frame_idxs_batch = np.zeros(
        [data_config['NUM_CLASSES'] * n_signs, data_config['INPUT_SIZE']], dtype=np.float32
    )

    # Dictionary mapping ordinally encoded sign to corresponding sample indices
    CLASS2IDXS = {}
    for i in range(data_config['NUM_CLASSES']):
        CLASS2IDXS[i] = np.argwhere(y == i).squeeze().astype(np.int32)

    while True:
        # Fill batch arrays
        for i in range(data_config['NUM_CLASSES']):
            idxs = np.random.choice(CLASS2IDXS[i], n_signs)
            X_batch[i * n_signs : (i + 1) * n_signs] = X[idxs]
            non_empty_frame_idxs_batch[
                i * n_signs : (i + 1) * n_signs
            ] = non_empty_frame_idxs[idxs]

        yield {
            "frames": X_batch,
            "non_empty_frame_idxs": non_empty_frame_idxs_batch,
        }, y_batch


def get_loss_fn(model_config):
    if model_config["LABEL_SMOOTHING"] is not None:
        loss_fn = SparseCategoricalCrossentropyLS
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    return loss_fn