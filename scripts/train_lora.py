from __future__ import annotations

import math
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import wandb
from torch.nn import functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.finetune_dataset import create_finetune_dataloaders
from scripts.lora import inject_lora
from scripts.model import RecipeGPT, build_causal_mask

CHECKPOINTS_DIRNAME = "checkpoints"
PRETRAINED_DIRNAME = "pretrained"
FINETUNED_DIRNAME = "finetuned"
BASE_CHECKPOINT_NAME = "best_recipegpt_4HEADS.pt"


def train() -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoints_dir = root / CHECKPOINTS_DIRNAME
    pretrained_dir = checkpoints_dir / PRETRAINED_DIRNAME
    finetuned_dir = checkpoints_dir / FINETUNED_DIRNAME
    checkpoint_path = pretrained_dir / BASE_CHECKPOINT_NAME
    save_path = finetuned_dir / f"{checkpoint_path.stem}_lora.pt"
    tokenizer_path = root / "data" / "recipe_tokenizer.model"
    data_path = root / "data" / "processed_recipes.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr, weight_decay, epochs, batch_size, grad_clip = 1e-4, 0.01, 3, 32, 1.0

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model = RecipeGPT(
        block_size=config["block_size"],
        vocab_size=checkpoint["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        n_layers=config["n_layers"],
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = inject_lora(model, r=8, alpha=16).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {trainable:,} trainable / {total:,} total")

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    train_loader, val_loader = create_finetune_dataloaders(data_path, sp, config["block_size"], batch_size)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, int(0.03 * total_steps))
    causal_mask = build_causal_mask(config["block_size"], device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_val, step = float("inf"), 0

    wandb.init(project="RecipeGPT", config={"lr": lr, "epochs": epochs, "batch_size": batch_size, "weight_decay": weight_decay})
    for epoch in range(epochs):
        model.train()
        for inputs, targets, loss_mask in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)
            batch_mask = causal_mask[:, :, : inputs.size(1), : inputs.size(1)]
            progress = min(step / max(1, total_steps - warmup_steps), 1.0)
            curr_lr = lr * (step + 1) / warmup_steps if step < warmup_steps else 0.5 * (1 + math.cos(math.pi * progress)) * lr
            optimizer.param_groups[0]["lr"] = curr_lr
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, _ = model(inputs, mask=batch_mask)
                vocab_size = logits.size(-1)
                per_token_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction="none")
                loss = (per_token_loss * loss_mask.view(-1)).sum() / loss_mask.sum()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            wandb.log({"train_loss": loss.item(), "lr": curr_lr}, step=step)

            if step % 200 == 0:
                model.eval()
                losses = []
                with torch.no_grad():
                    for batch_idx, (inputs, targets, loss_mask) in enumerate(val_loader):
                        if batch_idx >= 30:
                            break
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        loss_mask = loss_mask.to(device, non_blocking=True)
                        batch_mask = causal_mask[:, :, : inputs.size(1), : inputs.size(1)]
                        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            logits, _ = model(inputs, mask=batch_mask)
                            per_token_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
                            val_loss = (per_token_loss * loss_mask.view(-1)).sum() / loss_mask.sum()
                        losses.append(val_loss.item())
                mean_val = sum(losses) / len(losses)
                wandb.log({"val_loss": mean_val}, step=step)
                print(f"step {step} val_loss {mean_val:.4f}")
                if mean_val < best_val:
                    best_val = mean_val
                    torch.save({"lora_state_dict": model.state_dict(), "config": config, "vocab_size": checkpoint["vocab_size"]}, save_path)
                model.train()

        prompt = "<|ingredients|> 2 cups flour | 1 cup sugar | 3 eggs | 1 tsp vanilla | 1/2 cup butter"
        sample = torch.tensor([sp.encode(prompt)], dtype=torch.long, device=device)
        text = sp.decode(model.generate(sample, max_new_tokens=200, temp=0.8)[0].tolist())
        print(f"epoch {epoch + 1}: {text}")

    wandb.finish()


if __name__ == "__main__":
    train()