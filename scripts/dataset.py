from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class RecipeDataset(Dataset):
    """
    Token-level next-token dataset with cache-aware token loading and train/test split.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        block_size: int,
        split: str = "train",
        train_ratio: float = 0.95,
        cache_tokens: Optional[str | Path] = None,
        cache_meta: Optional[str | Path] = None,
        force_rebuild: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size

        data_path = str(data_path)

        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if not (0.0 < train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1")

        default_cache_tokens = data_path.replace(".txt", ".tokens.npy")
        default_cache_meta = data_path.replace(".txt", ".tokens.meta.json")

        self.cache_tokens = str(cache_tokens) if cache_tokens else default_cache_tokens
        self.cache_meta = str(cache_meta) if cache_meta else default_cache_meta

        all_tokens = self._load_or_build_tokens(data_path, force_rebuild)
        self.total_tokens = len(all_tokens)

        split_idx = int(self.total_tokens * train_ratio)
        if split == "train":
            self.tokens = all_tokens[:split_idx]
        else:
            self.tokens = all_tokens[split_idx:]

        min_needed = self.block_size + 1
        if len(self.tokens) < min_needed:
            raise ValueError(
                f"Split '{split}' has {len(self.tokens)} tokens; need at least {min_needed}. "
                "Increase data size, lower block_size, or adjust train_ratio."
            )

    def _load_or_build_tokens(self, data_path: str, force_rebuild: bool):
        if (not force_rebuild) and self._is_cache_valid(data_path):
            return np.load(self.cache_tokens, mmap_mode="r")

        all_tokens = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = self.tokenizer.encode(line.strip())
                all_tokens.extend(tokens)

        tokens = np.array(all_tokens, dtype=np.int32)
        self._save_cache(tokens, data_path)
        return tokens

    def _is_cache_valid(self, data_path: str) -> bool:
        if not (os.path.exists(self.cache_tokens) and os.path.exists(self.cache_meta)):
            return False

        try:
            with open(self.cache_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False

        if not os.path.exists(data_path):
            return False

        checks = [
            meta.get("source_text") == data_path,
            meta.get("source_mtime") == os.path.getmtime(data_path),
            meta.get("source_size") == os.path.getsize(data_path),
            meta.get("tokenizer_vocab_size") == int(self.tokenizer.get_piece_size()),
            meta.get("dtype") == "int32",
        ]
        return all(checks)

    def _save_cache(self, tokens: np.ndarray, data_path: str) -> None:
        np.save(self.cache_tokens, tokens)
        meta = {
            "dtype": str(tokens.dtype),
            "num_tokens": int(len(tokens)),
            "source_text": data_path,
            "source_mtime": os.path.getmtime(data_path),
            "source_size": os.path.getsize(data_path),
            "tokenizer_vocab_size": int(self.tokenizer.get_piece_size()),
        }
        with open(self.cache_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        chunk = self.tokens[start : start + self.block_size + 1]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def create_train_test_dataloaders(
    data_path: str | Path,
    tokenizer,
    block_size: int = 256,
    batch_size: int = 64,
    train_ratio: float = 0.95,
    cache_tokens: Optional[str | Path] = None,
    cache_meta: Optional[str | Path] = None,
    force_rebuild: bool = False,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> Tuple[RecipeDataset, RecipeDataset, DataLoader, DataLoader]:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_dataset = RecipeDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        block_size=block_size,
        split="train",
        train_ratio=train_ratio,
        cache_tokens=cache_tokens,
        cache_meta=cache_meta,
        force_rebuild=force_rebuild,
    )

    test_dataset = RecipeDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        block_size=block_size,
        split="test",
        train_ratio=train_ratio,
        cache_tokens=cache_tokens,
        cache_meta=cache_meta,
        force_rebuild=False,  # avoid duplicate rebuild
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    return train_dataset, test_dataset, train_loader, test_loader
