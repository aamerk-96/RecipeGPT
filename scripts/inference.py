import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


import torch
import sentencepiece as spm
from scripts.lora import inject_lora
from scripts.model import RecipeGPT

# Just a quick testing script 

CHECKPOINT_PATH = Path(r"D:\ML\ak_GPT\checkpoints\finetuned\best_recipegpt_4HEADS_lora.pt")
LORA_R = 8
LORA_ALPHA = 16

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load(r"D:\ML\ak_GPT\data\recipe_tokenizer.model")

# Load checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# Rebuild model with same config
config = checkpoint["config"]
model = RecipeGPT(
    block_size=config["block_size"],
    vocab_size=checkpoint["vocab_size"],
    d_model=config["d_model"],
    num_heads=config["num_heads"],
    n_layers=config["n_layers"],
    dropout=0.0,  # No dropout during inference
)

# Load the trained weights
state_key = "model_state_dict" if "model_state_dict" in checkpoint else "lora_state_dict"
if state_key == "lora_state_dict":
    model = inject_lora(model, r=LORA_R, alpha=LORA_ALPHA)
model.load_state_dict(checkpoint[state_key])
model.to(device)
model.eval()

# Generate
prompt = "<|ingredients|> 2 cups flour | 1 cup sugar | 3 eggs | 1 tsp vanilla | 1/2 cup butter <|title|>"
prompt_ids = sp.encode(prompt)
x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

output = model.generate(x, max_new_tokens=200, temp=0.8)
print(sp.decode(output[0].tolist()))