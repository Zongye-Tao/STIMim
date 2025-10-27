import os

import h5py
import numpy as np


def window_truncate(feature_vectors, seq_len, sliding_len=None):
    """ Generate time series samples, truncating windows from time-series data with a given sequence length.
    Parameters
    ----------
    feature_vectors: time series data, len(shape)=2, [total_length, feature_num]
    seq_len: sequence length
    sliding_len: size of the sliding window
    """
    sliding_len = seq_len if sliding_len is None else sliding_len
    total_len = feature_vectors.shape[0]
    start_indices = np.asarray(range(total_len // sliding_len)) * sliding_len
    if total_len - start_indices[-1] * sliding_len < seq_len:  # remove the last one if left length is not enough
        start_indices = start_indices[:-1]
    sample_collector = []
    for idx in start_indices:
        sample_collector.append(feature_vectors[idx: idx + seq_len])
    return np.asarray(sample_collector).astype('float32')


def random_mask(vector, artificial_missing_rate):
    """generate indices for random mask"""
    assert len(vector.shape) == 1
    indices = np.where(~np.isnan(vector))[0].tolist()
    indices = np.random.choice(indices, int(len(indices) * artificial_missing_rate))
    return indices


def add_artificial_mask(X, artificial_missing_rate, set_name):
    """Add artificial missing values.
    Parameters
    ----------
    X: feature vectors
    artificial_missing_rate: rate of artificial missing values that are going to be create
    set_name: dataset name, train/val/test
    """
    sample_num, seq_len, feature_num = X.shape
    if set_name == "train":
        # if this is train set, we don't need add artificial missing values right now.
        # If we want to apply MIT during training, dataloader will randomly mask out some values to generate X_hat

        # calculate empirical mean for model GRU-D, refer to paper
        mask = (~np.isnan(X)).astype(np.float32)
        X_filledWith0 = np.nan_to_num(X)
        empirical_mean_for_GRUD = np.sum(mask * X_filledWith0, axis=(0, 1)) / np.sum(
            mask, axis=(0, 1)
        )
        data_dict = {
            "X": X,
            "empirical_mean_for_GRUD": empirical_mean_for_GRUD,
        }
    else:
        # if this is val/test set, then we need to add artificial missing values right now,
        # because we need they are fixed
        X = X.reshape(-1)
        indices_for_holdout = random_mask(X, artificial_missing_rate)
        X_hat = np.copy(X)
        X_hat[indices_for_holdout] = np.nan  # X_hat contains artificial missing values
        missing_mask = (~np.isnan(X_hat)).astype(np.float32)
        # indicating_mask contains masks indicating artificial missing values
        indicating_mask = ((~np.isnan(X_hat)) ^ (~np.isnan(X))).astype(np.float32)

        data_dict = {
            "X": X.reshape([sample_num, seq_len, feature_num]),
            "X_hat": X_hat.reshape([sample_num, seq_len, feature_num]),
            "missing_mask": missing_mask.reshape([sample_num, seq_len, feature_num]),
            "indicating_mask": indicating_mask.reshape(
                [sample_num, seq_len, feature_num]
            ),
        }

    return data_dict


import numpy as np

def _time_dilate_bool(mask_bool, k=3):
    """
    Light 1D dilation along the time axis to connect isolated True points
    into short segments. k<=1 means no dilation.
    mask_bool: bool array of shape [N, T, F]
    """
    if k <= 1:
        return mask_bool
    pad = k // 2
    m = mask_bool.astype(np.uint8)
    out = np.zeros_like(m)
    # Max-pooling¨Clike effect along the time axis
    for offset in range(-pad, pad + 1):
        sl_from = slice(max(0, -offset), min(m.shape[1], m.shape[1] - offset))
        sl_to   = slice(max(0,  offset), min(m.shape[1], m.shape[1] + offset))
        out[:, sl_to, :] |= m[:, sl_from, :]
    return out.astype(bool)


def add_structural_mask(
    X, artificial_missing_rate, set_name,
    seed=None,
    # Column (feature) heterogeneity
    feature_group_frac=0.1061,
    feature_bias_ratio=3.374,
    # Time bias
    time_block_len_frac=0.0417,
    n_time_blocks=1,
    time_bias_ratio=6.152,
    # Cross boost
    inter_boost=3.0,
    # Block missing
    block_len_t_frac=0.1,    #
    block_len_f_frac=0.2,    # 
    n_blocks=3,              # 
    # Misc
    time_dilate_k=1,
):
    
    N, T, F = X.shape
    rng = np.random.default_rng(seed)

    if set_name == "train":
        mask = (~np.isnan(X)).astype(np.float32)
        X0 = np.nan_to_num(X)
        grud_mean = np.sum(mask * X0, axis=(0, 1)) / np.sum(mask, axis=(0, 1))
        return {"X": X, "empirical_mean_for_GRUD": grud_mean}

    # ----------------------
    # Step 1: structured missing (weighted sampling)
    # ----------------------
    valid = ~np.isnan(X)
    n_valid = int(valid.sum())
    n_holdout = int(round(n_valid * artificial_missing_rate))

    # feature bias
    k_feat = max(1, int(round(F * feature_group_frac)))
    biased_features = np.sort(rng.choice(np.arange(F), size=k_feat, replace=False))
    wf = np.ones(F); wf[biased_features] *= feature_bias_ratio

    # time bias
    L = max(1, int(round(T * time_block_len_frac)))
    blocks = []
    for _ in range(max(1, int(n_time_blocks))):
        start = 0 if T == L else rng.integers(0, T - L + 1)
        blocks.append((start, start+L))
    wt = np.ones(T)
    for (s, e) in blocks:
        wt[s:e] *= time_bias_ratio

    # assemble weights
    w = valid.astype(np.float64)
    w *= wt[None, :, None]
    w *= wf[None, None, :]
    inter_mask = (wt > 1.0)[None, :, None] & np.isin(np.arange(F), biased_features)[None, None, :]
    w *= np.where(inter_mask, inter_boost, 1.0)

    def _weighted_sample(mask_weights, k):
        flat = mask_weights.ravel()
        idx = np.flatnonzero(flat > 0)
        if k <= 0 or idx.size == 0:
            return np.zeros_like(mask_weights, dtype=bool)
        p = flat[idx].astype(np.float64); p /= p.sum()
        chosen = rng.choice(idx, size=min(k, idx.size), replace=False, p=p)
        out = np.zeros_like(flat, dtype=bool); out[chosen] = True
        return out.reshape(mask_weights.shape)

    holdout_struct = _weighted_sample(w, n_holdout)   # [N, T, F]

    # optional dilation
    if time_dilate_k > 1:
        holdout_struct = _time_dilate_bool(holdout_struct, k=time_dilate_k)

    # ----------------------
    # Step 2: block missing
    # ----------------------
    holdout_block = np.zeros((N, T, F), dtype=bool)
    block_len_t = max(1, int(T * block_len_t_frac))
    block_len_f = max(1, int(F * block_len_f_frac))

    for i in range(N):
        for _ in range(n_blocks):
            t0 = rng.integers(0, T - block_len_t + 1)
            f0 = rng.integers(0, F - block_len_f + 1)
            holdout_block[i, t0:t0+block_len_t, f0:f0+block_len_f] = True

    # ----------------------
    # Step 3: merge + calibrate
    # ----------------------
    holdout = holdout_struct | holdout_block
    cur = int(holdout.sum())

    if cur > n_holdout:
        idx_true = np.flatnonzero(holdout)
        keep = rng.choice(idx_true, size=n_holdout, replace=False)
        new_flat = np.zeros_like(holdout.ravel(), dtype=bool)
        new_flat[keep] = True
        holdout = new_flat.reshape(holdout.shape)
    elif cur < n_holdout:
        need = n_holdout - cur
        leftover_w = w * (~holdout).astype(np.float64)
        add = _weighted_sample(leftover_w, need)
        holdout |= add

    # ----------------------
    # Step 4: apply mask
    # ----------------------
    X_hat = np.array(X, copy=True)
    X_hat[holdout] = np.nan
    missing_mask = (~np.isnan(X_hat)).astype(np.float32)
    indicating_mask = ((~np.isnan(X_hat)) ^ (~np.isnan(X))).astype(np.float32)

    return {
        "X": X.astype(np.float32),
        "X_hat": X_hat.astype(np.float32),
        "missing_mask": missing_mask,
        "indicating_mask": indicating_mask,
    }


    
    

def saving_into_h5(saving_dir, data_dict, classification_dataset, mask_mode="structural"):
    """Save data into h5 file.
    Parameters
    ----------
    saving_dir: path of saving dir
    data_dict: data dictionary containing train/val/test sets
    classification_dataset: boolean, if this is a classification dataset
    """

    def save_each_set(handle, name, data):
        single_set = handle.create_group(name)
        if classification_dataset:
            single_set.create_dataset("labels", data=data["labels"].astype(int))
        single_set.create_dataset("X", data=data["X"].astype(np.float32))
        if name in ["val", "test"]:
            single_set.create_dataset("X_hat", data=data["X_hat"].astype(np.float32))
            single_set.create_dataset(
                "missing_mask", data=data["missing_mask"].astype(np.float32)
            )
            single_set.create_dataset(
                "indicating_mask", data=data["indicating_mask"].astype(np.float32)
            )

    filename = f"datasets_{mask_mode}.h5"
    saving_path = os.path.join(saving_dir, filename)
    
    with h5py.File(saving_path, "w") as hf:
        hf.create_dataset(
            "empirical_mean_for_GRUD",
            data=data_dict["train"]["empirical_mean_for_GRUD"],
        )
        save_each_set(hf, "train", data_dict["train"])
        save_each_set(hf, "val", data_dict["val"])
        save_each_set(hf, "test", data_dict["test"])


