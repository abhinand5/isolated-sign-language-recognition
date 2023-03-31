from sklearn.model_selection import StratifiedGroupKFold
from src.islr.utils import print_shape_dtype


def get_fold_idx_map(df, k_folds, force_lh=True, lh_signers=[], seed=42):
    while True:
        sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        _fold_ds_idx_map = {
            i: {"train": t_idxs, "val": v_idxs}
            for i, (t_idxs, v_idxs) in enumerate(
                sgkf.split(df.index, df.sign, df.participant_id)
            )
        }

        # Ensure only one left hander in every val group
        if force_lh:
            if all(
                [
                    len(
                        set(
                            df.iloc[_idxs["val"]]["participant_id"].unique()
                        ).intersection(set(lh_signers))
                    )
                    >= 1
                    for _idxs in _fold_ds_idx_map.values()
                ]
            ):
                return _fold_ds_idx_map
            else:
                print(".", end="")
        else:
            return _fold_ds_idx_map


def get_val_of_fold(X, y, fold_num, fold_ds_idx_map, non_empty_frame_idxs):
    fold_idxs = fold_ds_idx_map[fold_num]
    X_val, y_val = X[fold_idxs["val"]], y[fold_idxs["val"]]
    non_empty_frame_idxs_val = non_empty_frame_idxs[fold_idxs["val"]]

    print_shape_dtype(
        [X_val, y_val, non_empty_frame_idxs_val],
        ["X_val", "y_val", "NON_EMPTY_FRAME_IDXS_VAL"],
    )

    return X_val, y_val, non_empty_frame_idxs_val
