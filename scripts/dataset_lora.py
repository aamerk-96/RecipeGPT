import random
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sentencepiece as spm

TITLE_TOKEN_ID = 6
PAD_ID = 0


class FineTuneDataset(Dataset):
    def __init__(self, recipes, tokenizer, block_size=256):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []

        for recipe in recipes:
            result = self._build_example(recipe.strip())
            if result is not None:
                self.examples.append(result)

        print(f"Kept {len(self.examples)}, skipped {len(recipes) - len(self.examples)}")

    def _build_example(self, recipe):
        # Step 1: Rearrange — put ingredients first
        ingredients = self._extract(recipe, "<|ingredients|>", "<|directions|>")
        title = self._extract(recipe, "<|title|>", "<|ingredients|>")
        directions = self._extract(recipe, "<|directions|>", "<|endofrecipe|>")
        reordered = f"<|ingredients|> {ingredients} <|title|> {title} <|directions|> {directions} <|endofrecipe|>"

        # Step 2: Tokenize
        tokens = self.tokenizer.encode(reordered)
        if len(tokens) > self.block_size:
            return None  # Too long, skip

        # Step 3: Find where completion starts
        title_idx = tokens.index(TITLE_TOKEN_ID)

        # Step 4: Build loss mask — 0 for prompt, 1 for completion, 0 for padding
        mask = [0.0] * title_idx + [1.0] * (len(tokens) - title_idx)
        mask += [0.0] * (self.block_size - len(tokens))

        # Step 5: Pad tokens
        tokens += [PAD_ID] * (self.block_size - len(tokens))

        # Step 6: Create input/target/mask tensors
        tokens = torch.tensor(tokens, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.float)

        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        loss_mask = mask[1:]

        return input_ids, target_ids, loss_mask

    def _extract(self, recipe, start_token, end_token):
        start = recipe.find(start_token) + len(start_token)
        end = recipe.find(end_token)
        return recipe[start:end].strip()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def create_finetune_dataloaders(data_path, tokenizer, block_size=256, batch_size=32, sample_size=100000, seed=42):
    with open(data_path, "r", encoding="utf-8") as f:
        recipes = [line.strip() for line in f if line.strip()]

    # Sample and shuffle
    rng = random.Random(seed)
    recipes = rng.sample(recipes, min(sample_size, len(recipes)))

    # Split at recipe level
    split_idx = int(len(recipes) * 0.95)
    train_dataset = FineTuneDataset(recipes[:split_idx], tokenizer, block_size)
    val_dataset = FineTuneDataset(recipes[split_idx:], tokenizer, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader


# Quick test 

sp = spm.SentencePieceProcessor()
sp.load(r"D:\ML\ak_GPT\data\recipe_tokenizer.model")

train_loader, val_loader = create_finetune_dataloaders(
    data_path=r"D:\ML\ak_GPT\data\processed_recipes.txt",
    tokenizer=sp,
    block_size=256,
    batch_size=32,
)

# Check one example
x, y, mask = train_loader.dataset[0]
print(f"input_ids shape: {x.shape}")
print(f"target_ids shape: {y.shape}")
print(f"loss_mask shape: {mask.shape}")
print(f"Prompt tokens (mask=0): {int((mask == 0).sum())}")
print(f"Completion tokens (mask=1): {int((mask == 1).sum())}")

# Decode to verify the reordering looks right
full_tokens = torch.cat([x[:1], y])  # reconstruct full sequence
print(sp.decode(full_tokens.tolist()))