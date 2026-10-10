"""Sequential samples and evaluation context helpers for FPMC."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _column(config, setting, default):
    return config.get(setting, default)


def _ordered_groups(frame, config):
    user_col = _column(config, "UID_NAME", "user")
    item_col = _column(config, "IID_NAME", "item")
    time_col = config.get("TIME_NAME", config.get("TID_NAME", "timestamp"))
    if time_col in frame.columns:
        frame = frame.sort_values([user_col, time_col], kind="stable")
    else:
        # Preserve frame order when the selected dataset has no timestamp.
        frame = frame.copy()
    return user_col, item_col, frame.groupby(user_col, sort=False)


def last_item_by_user(frame, config):
    """Return each user's latest observed item as integer IDs."""
    user_col, item_col, groups = _ordered_groups(frame, config)
    return {int(user): int(group[item_col].iloc[-1]) for user, group in groups if len(group)}


def make_fpmc_samples(frame, config, seed=None):
    """Create (user, previous, positive, negative) BPR transitions.

    A negative is sampled uniformly from items the user did not interact with
    in the current training split. `num_ng` controls negatives per transition.
    """
    user_col, item_col, groups = _ordered_groups(frame, config)
    item_num = int(config["item_num"])
    num_ng = int(config.get("num_ng", 1))
    rng = np.random.default_rng(config.get("seed", 42) if seed is None else seed)
    sequences = {
        int(user): [int(item) for item in group[item_col].tolist()]
        for user, group in groups
    }
    positives = {user: set(items) for user, items in sequences.items()}
    samples = []
    for user, items in sequences.items():
        for previous, positive in zip(items[:-1], items[1:]):
            available = item_num - len(positives[user])
            if available <= 0:
                continue
            for _ in range(num_ng):
                negative = int(rng.integers(item_num))
                while negative in positives[user]:
                    negative = int(rng.integers(item_num))
                samples.append((user, previous, positive, negative))
    rng.shuffle(samples)
    return FPMCDataset(samples)


def build_fpmc_loader(frame, config, shuffle=True, seed=None):
    """Build the four-field loader expected by FPMC.calc_loss."""
    dataset = make_fpmc_samples(frame, config, seed=seed)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle and len(dataset) > 0,
        num_workers=int(config.get("num_workers", 0)),
    )


def set_fpmc_eval_context(config, history_frame):
    """Set the last item visible before the current validation/test targets."""
    config["fpmc_eval_prev"] = last_item_by_user(history_frame, config)
    return config


class FPMCDataset(Dataset):
    def __init__(self, samples):
        self.samples = torch.as_tensor(samples, dtype=torch.long)
        if self.samples.numel() == 0:
            self.samples = self.samples.reshape(0, 4)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        return tuple(self.samples[index].unbind())
