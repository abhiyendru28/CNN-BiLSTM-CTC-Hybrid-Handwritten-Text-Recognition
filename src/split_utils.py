import os
import logging
import numpy as np

log = logging.getLogger(__name__)


def _validate_indices(train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, total_size: int) -> None:
    all_idx = np.concatenate([train_idx, val_idx, test_idx]).astype(np.int64)

    if all_idx.size != total_size:
        raise ValueError(f"Split size mismatch: {all_idx.size} indices for dataset size {total_size}.")

    unique_idx = np.unique(all_idx)
    if unique_idx.size != total_size:
        raise ValueError("Split indices overlap or contain duplicates.")

    if unique_idx.size > 0 and (unique_idx[0] < 0 or unique_idx[-1] >= total_size):
        raise ValueError("Split indices contain out-of-range values.")


def load_or_create_split_indices(
    total_size: int,
    split_path: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    strict: bool = True,
):
    if total_size <= 0:
        raise ValueError("Dataset is empty, cannot create split indices.")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if os.path.exists(split_path):
        data = np.load(split_path)
        keys = set(data.files)
        required = {"train_idx", "val_idx", "test_idx"}

        if required.issubset(keys):
            train_idx = data["train_idx"].astype(np.int64)
            val_idx = data["val_idx"].astype(np.int64)
            test_idx = data["test_idx"].astype(np.int64)
            try:
                _validate_indices(train_idx, val_idx, test_idx, total_size)
                return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
            except ValueError as exc:
                if strict:
                    raise
                log.warning(
                    "Split manifest %s failed validation (%s). Recreating split.", split_path, str(exc)
                )

        if strict:
            raise ValueError(
                f"Split manifest {split_path} is not in replication format "
                f"(missing train_idx/val_idx/test_idx)."
            )

    # Create deterministic split and freeze it on disk.
    rng = np.random.default_rng(seed)
    all_idx = np.arange(total_size, dtype=np.int64)
    rng.shuffle(all_idx)

    train_end = int(total_size * train_ratio)
    val_end = train_end + int(total_size * val_ratio)

    train_idx = all_idx[:train_end]
    val_idx = all_idx[train_end:val_end]
    test_idx = all_idx[val_end:]

    _validate_indices(train_idx, val_idx, test_idx, total_size)

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    np.savez(split_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()