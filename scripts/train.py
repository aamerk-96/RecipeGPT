from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import math
import time

import sentencepiece as spm
import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.dataset import create_train_test_dataloaders
from scripts.model import RecipeGPT, build_causal_mask

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"

DATA_PATH = DATA_DIR / "processed_recipes.txt"
TOKENIZER_PATH = DATA_DIR / "recipe_tokenizer.model"
CHECKPOINT_NAME = "recipegpt.pt"

BLOCK_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
TRAIN_RATIO = 0.95
NUM_WORKERS = 0
GRAD_CLIP = 1.0
EVAL_EVERY = 200
EVAL_BATCHES = 30
LOG_EVERY = 50
SEED = 42

D_MODEL = 768
NUM_HEADS = 8
N_LAYERS = 6
DROPOUT = 0.1

FORCE_REBUILD_CACHE = False
DEVICE = "cuda"  # "auto", "cpu", "cuda"

SAMPLE_PROMPT = "<|startofrecipe|> <|title|>"
SAMPLE_TOKENS = 120
SAMPLE_TEMP = 0.9

USE_MIXED_PRECISION = True
AMP_DTYPE = torch.bfloat16
GRAD_ACCUM_STEPS = 4

MIN_LR = 3e-5
WARMUP_RATIO = 0.03

USE_WANDB = True
WANDB_PROJECT = "RecipeGPT"
WANDB_ENTITY = "aamerk4716-n-a-org"
WANDB_MODE = "online"
WANDB_RUN_NAME = ""  # set this per run, e.g. "baseline-bs32-lr3e4"


def build_config_dict(vocab_size: int) -> dict:
    return {
        "data_path": str(DATA_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "block_size": BLOCK_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "train_ratio": TRAIN_RATIO,
        "num_workers": NUM_WORKERS,
        "grad_clip": GRAD_CLIP,
        "eval_every": EVAL_EVERY,
        "eval_batches": EVAL_BATCHES,
        "log_every": LOG_EVERY,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "n_layers": N_LAYERS,
        "dropout": DROPOUT,
        "seed": SEED,
        "device": DEVICE,
        "use_mixed_precision": USE_MIXED_PRECISION,
        "amp_dtype": str(AMP_DTYPE).replace("torch.", ""),
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "min_lr": MIN_LR,
        "warmup_ratio": WARMUP_RATIO,
        "sample_prompt": SAMPLE_PROMPT,
        "sample_tokens": SAMPLE_TOKENS,
        "sample_temp": SAMPLE_TEMP,
        "vocab_size": vocab_size,
    }


def get_learning_rate(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return LEARNING_RATE * float(step + 1) / float(max(1, warmup_steps))

    if total_steps <= warmup_steps:
        return MIN_LR

    decay_progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return MIN_LR + cosine * (LEARNING_RATE - MIN_LR)


def load_wandb() -> Any:
    import importlib

    try:
        return importlib.import_module("wandb")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "WANDB tracking is enabled but package 'wandb' is not installed. Run: pip install wandb"
        ) from exc


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device_flag: str) -> torch.device:
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda passed but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(
    model: RecipeGPT,
    dataloader,
    device: torch.device,
    max_batches: int,
    criterion: nn.Module,
    mask: torch.Tensor,
    use_amp: bool,
) -> float:
    model.eval()
    losses: list[float] = []

    for batch_idx, (x, y) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=AMP_DTYPE,
            enabled=use_amp,
        ):
            logits, _ = model(x, mask=mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())

    if not losses:
        return float("nan")

    return float(sum(losses) / len(losses))


def train() -> None:
    set_seed(SEED)

    device = pick_device(DEVICE)
    print(f"Using device: {device}")

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {TOKENIZER_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training text not found: {DATA_PATH}")

    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    vocab_size = int(tokenizer.get_piece_size())
    config = build_config_dict(vocab_size)

    wandb = None
    if USE_WANDB:
        wandb = load_wandb()
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            mode=WANDB_MODE,
            name=WANDB_RUN_NAME or None,
            config=config,
            dir=str(ROOT),
        )

    train_dataset, test_dataset, train_loader, test_loader = create_train_test_dataloaders(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        force_rebuild=FORCE_REBUILD_CACHE,
        num_workers=NUM_WORKERS,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Vocab size from tokenizer: {vocab_size}")

    model = RecipeGPT(
        block_size=BLOCK_SIZE,
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters (total): {total_params:,} ({total_params / 1_000_000:.2f}M)")
    print(
        f"Model parameters (trainable): {trainable_params:,} ({trainable_params / 1_000_000:.2f}M)"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    use_amp = USE_MIXED_PRECISION and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    updates_per_epoch = max(1, math.ceil(len(train_loader) / GRAD_ACCUM_STEPS))
    total_updates = max(1, updates_per_epoch * EPOCHS)
    warmup_steps = max(1, int(total_updates * WARMUP_RATIO))

    train_mask = build_causal_mask(BLOCK_SIZE, device=device)
    eval_mask = train_mask

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_eval_loss = float("inf")
    global_step = 0  # optimizer update steps
    total_tokens_seen = 0
    tokens_since_log = 0
    run_start_time = time.perf_counter()
    log_start_time = run_start_time

    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            epoch_tokens = 0
            epoch_start_time = time.perf_counter()

            for step, (x, y) in enumerate(train_loader, start=1):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                batch_tokens = x.numel()
                total_tokens_seen += batch_tokens
                tokens_since_log += batch_tokens
                epoch_tokens += batch_tokens

                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=AMP_DTYPE,
                    enabled=use_amp,
                ):
                    logits, _ = model(x, mask=train_mask)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    loss = loss / GRAD_ACCUM_STEPS

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                should_step = step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader)
                loss_value = float(loss.item()) * GRAD_ACCUM_STEPS
                running_loss += loss_value
                if should_step:
                    current_lr = get_learning_rate(global_step, total_updates, warmup_steps)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr

                    if GRAD_CLIP > 0:
                        if use_amp:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                if should_step and global_step % LOG_EVERY == 0:
                    avg_train_loss = running_loss / LOG_EVERY
                    log_elapsed = max(time.perf_counter() - log_start_time, 1e-8)
                    run_elapsed = max(time.perf_counter() - run_start_time, 1e-8)
                    tokens_per_sec = tokens_since_log / log_elapsed
                    avg_tokens_per_sec = total_tokens_seen / run_elapsed
                    print(
                        f"epoch={epoch} micro_step={step}/{len(train_loader)} "
                        f"update_step={global_step} lr={current_lr:.6f} train_loss={avg_train_loss:.4f} "
                        f"tok/s={tokens_per_sec:,.0f} avg_tok/s={avg_tokens_per_sec:,.0f}"
                    )
                    if USE_WANDB and wandb is not None:
                        wandb.log(
                            {
                                "train/loss": avg_train_loss,
                                "train/lr": current_lr,
                                "train/tokens_per_sec": tokens_per_sec,
                                "train/avg_tokens_per_sec": avg_tokens_per_sec,
                                "train/tokens_seen": total_tokens_seen,
                                "train/epoch": epoch,
                                "train/micro_step": step,
                                "global_step": global_step,
                            },
                            step=global_step,
                        )
                    running_loss = 0.0
                    tokens_since_log = 0
                    log_start_time = time.perf_counter()

                if should_step and global_step % EVAL_EVERY == 0:
                    eval_loss = evaluate(
                        model=model,
                        dataloader=test_loader,
                        device=device,
                        max_batches=EVAL_BATCHES,
                        criterion=criterion,
                        mask=eval_mask,
                        use_amp=use_amp,
                    )
                    print(f"eval global_step={global_step} val_loss={eval_loss:.4f}")
                    if USE_WANDB and wandb is not None:
                        wandb.log(
                            {
                                "val/loss": eval_loss,
                                "train/epoch": epoch,
                                "global_step": global_step,
                            },
                            step=global_step,
                        )

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_path = CHECKPOINT_DIR / f"best_{CHECKPOINT_NAME}"
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "epoch": epoch,
                                "global_step": global_step,
                                "val_loss": eval_loss,
                                "config": config,
                                "vocab_size": vocab_size,
                            },
                            best_path,
                        )
                        print(f"Saved new best checkpoint: {best_path}")

            epoch_eval_loss = evaluate(
                model=model,
                dataloader=test_loader,
                device=device,
                max_batches=EVAL_BATCHES,
                criterion=criterion,
                mask=eval_mask,
                use_amp=use_amp,
            )
            epoch_elapsed = max(time.perf_counter() - epoch_start_time, 1e-8)
            epoch_tokens_per_sec = epoch_tokens / epoch_elapsed
            print(f"End of epoch {epoch}: val_loss={epoch_eval_loss:.4f}")
            print(
                f"Epoch throughput: tokens={epoch_tokens:,} tok/s={epoch_tokens_per_sec:,.0f} "
                f"duration={epoch_elapsed:.1f}s"
            )
            if USE_WANDB and wandb is not None:
                wandb.log(
                    {
                        "epoch/val_loss": epoch_eval_loss,
                        "epoch/tokens": epoch_tokens,
                        "epoch/tokens_per_sec": epoch_tokens_per_sec,
                        "epoch/duration_sec": epoch_elapsed,
                        "train/epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )

            latest_path = CHECKPOINT_DIR / CHECKPOINT_NAME
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": epoch_eval_loss,
                    "config": config,
                    "vocab_size": vocab_size,
                },
                latest_path,
            )
            print(f"Saved latest checkpoint: {latest_path}")

            # Quick qualitative sanity check after each epoch.
            prompt_ids = tokenizer.encode(SAMPLE_PROMPT)
            prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            model.eval()
            generated_ids = model.generate(
                prompt,
                max_new_tokens=SAMPLE_TOKENS,
                temp=SAMPLE_TEMP,
            )
            generated_text = tokenizer.decode(generated_ids[0].tolist())
            print("Sample generation:")
            print(generated_text)
            if USE_WANDB and wandb is not None:
                wandb.log(
                    {
                        "sample/text": generated_text,
                        "train/epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )
    finally:
        if USE_WANDB and wandb is not None and wandb.run is not None:
            wandb.finish()

    print("Training complete.")


if __name__ == "__main__":
    train()
