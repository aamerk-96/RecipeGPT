import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch 
from torch import nn

from scripts.model import RecipeGPT

class LoRALinear(nn.Module):
    def __init__(self, original_linear, r, alpha):
        super().__init__()
        self.original_linear = original_linear
        self.A = nn.Parameter(torch.randn(original_linear.in_features, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, original_linear.out_features))
        self.alpha = alpha
        self.r = r
        self.scaling = alpha / r
        # Freeze the original linear layer
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        base = self.original_linear(x)
        update = (x @ self.A @ self.B) * self.scaling
        return base + update

def inject_lora(model, r, alpha):
    for block in model.transformer_blocks:
        block.attention.wq = LoRALinear(block.attention.wq, r, alpha)
        block.attention.wv = LoRALinear(block.attention.wv, r, alpha)
    
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only LoRA parameters
    for block in model.transformer_blocks:
        block.attention.wq.A.requires_grad = True
        block.attention.wq.B.requires_grad = True
        block.attention.wv.A.requires_grad = True
        block.attention.wv.B.requires_grad = True
    
    return model

if __name__ == "__main__":
    checkpoint = torch.load(r"D:\ML\ak_GPT\checkpoints\recipegpt_2.pt", map_location="cpu")
    model = RecipeGPT(
        block_size=256,
        vocab_size=32000,
        d_model=768,
        num_heads=8,
        n_layers=6,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = inject_lora(model, r=8, alpha=16)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total:,}")
    print(f"Trainable: {trainable:,}")
    print(f"Trainable %: {trainable/total*100:.2f}%")
