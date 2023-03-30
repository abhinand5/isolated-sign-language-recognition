import numpy as np
from tqdm import tqdm


def get_feature_stats(X, feature_idxs, n_dims):
    feature_mean_x = np.zeros([feature_idxs.size], dtype=np.float32)
    feature_mean_y = np.zeros([feature_idxs.size], dtype=np.float32)
    feature_std_x = np.zeros([feature_idxs.size], dtype=np.float32)
    feature_std_y = np.zeros([feature_idxs.size], dtype=np.float32)

    for col, ll in enumerate(
        tqdm(
            np.transpose(X[:, :, feature_idxs], [2, 3, 0, 1]).reshape(
                [feature_idxs.size, n_dims, -1]
            )
        )
    ):
        for dim, l in enumerate(ll):
            v = l[np.nonzero(l)]
            if dim == 0:  # X
                feature_mean_x[col] = v.mean()
                feature_std_x[col] = v.std()
            if dim == 1:  # Y
                feature_mean_y[col] = v.mean()
                feature_std_y[col] = v.std()

    feature_mean = np.array([feature_mean_x, feature_mean_y]).T
    feature_std = np.array([feature_std_x, feature_std_y]).T

    return {"mean": feature_mean, "std": feature_std}


def get_hand_feature_stats(X, hand_idxs, left_hand_idxs, right_hand_idxs, n_dims):
    # LEFT HAND
    left_hands_mean_x = np.zeros([left_hand_idxs.size], dtype=np.float32)
    left_hands_mean_y = np.zeros([left_hand_idxs.size], dtype=np.float32)
    left_hands_std_x = np.zeros([left_hand_idxs.size], dtype=np.float32)
    left_hands_std_y = np.zeros([left_hand_idxs.size], dtype=np.float32)
    # RIGHT HAND
    right_hands_mean_x = np.zeros([right_hand_idxs.size], dtype=np.float32)
    right_hands_mean_y = np.zeros([right_hand_idxs.size], dtype=np.float32)
    right_hands_std_x = np.zeros([right_hand_idxs.size], dtype=np.float32)
    right_hands_std_y = np.zeros([right_hand_idxs.size], dtype=np.float32)

    # fig, axes = plt.subplots(3, 1, figsize=(15, N_DIMS*6))

    for col, ll in enumerate(
        tqdm(
            np.transpose(X[:, :, hand_idxs], [2, 3, 0, 1]).reshape(
                [hand_idxs.size, n_dims, -1]
            )
        )
    ):
        for dim, l in enumerate(ll):
            v = l[np.nonzero(l)]
            if dim == 0:  # X
                if col < right_hand_idxs.size:  # LEFT HAND
                    left_hands_mean_x[col] = v.mean()
                    left_hands_std_x[col] = v.std()
                else:
                    right_hands_mean_x[col - left_hand_idxs.size] = v.mean()
                    right_hands_std_x[col - left_hand_idxs.size] = v.std()
            if dim == 1:  # Y
                if col < right_hand_idxs.size:  # LEFT HAND
                    left_hands_mean_y[col] = v.mean()
                    left_hands_std_y[col] = v.std()
                else:  # RIGHT HAND
                    right_hands_mean_y[col - left_hand_idxs.size] = v.mean()
                    right_hands_std_y[col - left_hand_idxs.size] = v.std()

    left_hands_mean = np.array([left_hands_mean_x, left_hands_mean_y]).T
    left_hands_std = np.array([left_hands_std_x, left_hands_std_y]).T
    right_hands_mean = np.array([right_hands_mean_x, right_hands_mean_y]).T
    right_hands_std = np.array([right_hands_std_x, right_hands_std_y]).T

    return {
        "lh_mean": left_hands_mean,
        "lh_std": left_hands_std,
        "rh_mean": right_hands_mean,
        "rh_std": right_hands_std,
    }


def get_all_feature_stats(X, n_dims):
    lips_stats = get_feature_stats(X, extra_features.LIPS_IDXS, n_dims)
    pose_stats = get_feature_stats(X, extra_features.POSE_IDXS, n_dims)
    hand_stats = get_hand_feature_stats(
        X,
        extra_features.HAND_IDXS,
        extra_features.LEFT_HAND_IDXS,
        extra_features.RIGHT_HAND_IDXS,
        n_dims,
    )

    return lips_stats, pose_stats, hand_stats


class ExtraFeatures:
    USE_TYPES = ["left_hand", "pose", "right_hand"]
    START_IDX = 468
    LIPS_IDXS0 = np.array(
        [
            61,
            185,
            40,
            39,
            37,
            0,
            267,
            269,
            270,
            409,
            291,
            146,
            91,
            181,
            84,
            17,
            314,
            405,
            321,
            375,
            78,
            191,
            80,
            81,
            82,
            13,
            312,
            311,
            310,
            415,
            95,
            88,
            178,
            87,
            14,
            317,
            402,
            318,
            324,
            308,
        ]
    )
    # Landmark indices in original data
    LEFT_HAND_IDXS0 = np.arange(468, 489)
    RIGHT_HAND_IDXS0 = np.arange(522, 543)
    POSE_IDXS0 = np.arange(502, 512)
    LANDMARK_IDXS0 = np.concatenate(
        (LIPS_IDXS0, LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0, POSE_IDXS0)
    )
    HAND_IDXS0 = np.concatenate((LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0), axis=0)
    N_COLS = LANDMARK_IDXS0.size
    # Landmark indices in processed data
    LIPS_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, LIPS_IDXS0)).squeeze()
    LEFT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, LEFT_HAND_IDXS0)).squeeze()
    RIGHT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, RIGHT_HAND_IDXS0)).squeeze()
    HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, HAND_IDXS0)).squeeze()
    POSE_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, POSE_IDXS0)).squeeze()

    LIPS_START = 0
    LEFT_HAND_START = LIPS_IDXS.size
    RIGHT_HAND_START = LEFT_HAND_START + LEFT_HAND_IDXS.size
    POSE_START = RIGHT_HAND_START + RIGHT_HAND_IDXS.size

    # All left handers in the dataset (useful for splitting the data)
    LH_SIGNERS = [16069, 32319, 36257, 22343, 27610, 61333, 34503, 55372, 37055]


extra_features = ExtraFeatures()
