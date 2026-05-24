import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


import torch
import sentencepiece as spm
from scripts.model import RecipeGPT

# Just a quick testing script 

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load(r"D:\ML\ak_GPT\data\recipe_tokenizer.model")

# Load checkpoint
checkpoint = torch.load(r"D:\ML\ak_GPT\checkpoints\best_recipegpt_2.pt", map_location="cuda")

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
model.load_state_dict(checkpoint["model_state_dict"])
model.to("cuda")
model.eval()

# Generate
prompt = "<|startofrecipe|> <|title|> Butter Chicken <|ingredients|>"
prompt_ids = sp.encode(prompt)
x = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")

output = model.generate(x, max_new_tokens=200, temp=0.8)
print(sp.decode(output[0].tolist()))