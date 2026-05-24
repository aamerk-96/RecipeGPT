# RecipeGPT

A 91.9M parameter transformer language model trained to generate recipes. Built entirely from scratch in PyTorch - no HuggingFace transformers, no pretrained weights, no shortcuts. This was my attempt at learning how language models actually work from the ground up.

## What I Built From Scratch (with occasional AI assistance for debugging and concepts)

- Scaled dot-product attention with causal masking
- Multi-head attention with learned Q, K, V projections
- Transformer blocks (pre-norm, residual connections, GELU FFN)
- Full GPT-style decoder-only model
- BPE tokenizer (trained with SentencePiece on 2M+ recipes)
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Training loop with mixed precision, gradient accumulation, cosine LR schedule

## Architecture

| | |
|---|---|
| Parameters | 91,909,376 |
| Layers | 6 |
| Attention heads | 4 |
| Embedding dim | 768 |
| FFN dim | 3072 |
| Context length | 256 tokens |
| Vocab size | 32,000 (BPE) |

The current fine-tuned checkpoint in this repo is based on `best_recipegpt_4HEADS.pt`. There is also an older 8-head checkpoint in `checkpoints/pretrained/best_recipegpt_2.pt`.

## Training

**Pretraining:** Trained on ~2M recipes from RecipeNLG (~398M tokens). The model learns next-token prediction on the full recipe text - titles, ingredients, directions, everything. The completed 5-epoch W&B run in this repo reached `loss/train = 1.62` and `loss/val = 1.81` on an NVIDIA GeForce RTX 5070 Ti.

**LoRA Fine-tuning:** Froze the pretrained weights and added LoRA adapters (rank 8) to the Q and V attention projections. That adds 147,456 trainable parameters, which is 0.16% of the full 92,056,832-parameter LoRA-wrapped model. Fine-tuned on an ingredient-to-recipe task - given a list of ingredients, generate a recipe title and cooking directions.

Experiment tracking with [Weights & Biases](https://wandb.ai/aamerk4716-n-a/RecipeGPT).

## Sample Output

**Prompt:**

```text
<|ingredients|> 2 cups flour | 1 cup sugar | 3 eggs | 1 tsp vanilla | 1/2 cup butter
```

**Generated:**

```text
<|title|> Sweet Pie
<|directions|> Preheat oven to a large skillet, sugar and sugar with a large pan.
| In a small pieces in a medium saucepan or butter or until butter, in flour and
sugar and sugar, egg, mix well. | If baking pan in a large bowl. | Bake in warm.
| Drain. | Pour the cake mix well. | Boil well. | Preheat oven. | Stir in a
large bowl. | Add sugar with aluminum mixture. | Pour over medium-skinned water.
| Mix and baking pan. | Pour into a separate bowl. | Remove from the Cool.
| Pour over medium heat and pan. | Cook overnight. | Transfer to 3 to a cookie
sheet. <|endofrecipe|>
```

Is it great? Probably not but a good starting point! 


## Status - Ongoing and what I'm working on

- **More data:** Only used ~398M tokens. The Chinchilla-optimal amount for a model of this size is much higher, so more data would likely help more than a bigger model.
- **Higher LoRA rank:** Rank 8 was too restrictive for the ingredient-to-recipe task. Rank 32-64 with LoRA on all four attention projections would likely produce much better fine-tuned outputs.
- **RoPE positional encoding:** Used learned positional embeddings instead. RoPE generalizes better to unseen sequence lengths.
- **KV cache:** Generation is slow without it - the model recomputes attention over all previous tokens at every step.

## Data

Download RecipeNLG from [recipenlg.cs.put.poznan.pl](https://recipenlg.cs.put.poznan.pl/) 

## Built With

PyTorch, SentencePiece, Weights & Biases
