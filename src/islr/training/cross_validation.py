from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from src.islr.utils import print_shape_dtype


def get_fold_idx_map(df, k_folds, force_lh=True, lh_signers=[], val_ratio=0.1, all_data=False, seed=42):
    if k_folds == 1 and all_data:
        return {0: {"train": df.index.tolist(), "val": df.index.tolist()}}
    elif k_folds == 1:
        splitter = GroupShuffleSplit(test_size=val_ratio, n_splits=2, random_state=seed)
        participant_ids = df['participant_id'].values
        train_idxs, val_idxs = next(splitter.split(df.index, df.sign, groups=participant_ids))
        
        return{0: {"train": train_idxs, "val": val_idxs}}

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

def get_fold_data(X, y, fold_num, fold_ds_idx_map, non_empty_frame_idxs):
    fold_idxs = fold_ds_idx_map[fold_num]
    X_train, y_train = X[fold_idxs["train"]], y[fold_idxs["train"]]
    X_val, y_val = X[fold_idxs["val"]], y[fold_idxs["val"]]
    
    non_empty_frame_idxs_train = non_empty_frame_idxs[fold_idxs["train"]]
    non_empty_frame_idxs_val = non_empty_frame_idxs[fold_idxs["val"]]

    print_shape_dtype(
        [X_train, y_train, non_empty_frame_idxs_train],
        ["X_train", "y_train", "NON_EMPTY_FRAME_IDXS_TRAIN"],
    )

    print_shape_dtype(
        [X_val, y_val, non_empty_frame_idxs_val],
        ["X_val", "y_val", "NON_EMPTY_FRAME_IDXS_VAL"],
    )

    return X_train, y_train, non_empty_frame_idxs_train, X_val, y_val, non_empty_frame_idxs_val
