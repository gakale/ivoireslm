#!/usr/bin/env python3
"""Charge les poids expérimentaux IvoireSLM 17M et génère une continuation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from modeling_microivoire import MicroIvoireTransformer17M, TrainingConfig


def load_model(weights_path: Path, device: torch.device):
    config = TrainingConfig()
    model = MicroIvoireTransformer17M(config)
    model.load_state_dict(load_file(str(weights_path), device="cpu"))
    model.to(device).eval()
    return model, config


@torch.inference_mode()
def generate(
    model: MicroIvoireTransformer17M,
    token_ids: list[int],
    maximum_new_tokens: int,
    temperature: float,
    top_k: int,
    eos_token_id: int,
    device: torch.device,
) -> list[int]:
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    for _ in range(maximum_new_tokens):
        context = tokens[:, -model.config.block_size :]
        logits, _ = model(context)
        next_logits = logits[:, -1, :] / temperature
        if top_k == 1:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            values, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
            cutoff = values[:, -1].unsqueeze(-1)
            filtered = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
            probabilities = torch.softmax(filtered, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)
        if int(next_token.item()) == eos_token_id:
            break
    return tokens[0].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--weights", type=Path, default=Path("model.safetensors"))
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError("temperature doit être strictement positive")
    if args.top_k < 1:
        raise ValueError("top-k doit être au moins égal à 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model, config = load_model(args.weights, device)
    if sum(parameter.numel() for parameter in model.parameters()) != 17_129_280:
        raise AssertionError("nombre de paramètres inattendu")
    encoded = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if not encoded:
        raise ValueError("le prompt ne peut pas être vide")
    eos_token_id = tokenizer.token_to_id("<EOS>")
    if eos_token_id is None:
        raise ValueError("token <EOS> absent du tokenizer")
    result = generate(
        model,
        encoded,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        eos_token_id,
        device,
    )
    print(tokenizer.decode(result, skip_special_tokens=True))
    print(f"\n[modèle={config.model_id}; appareil={device}; sortie non vérifiée]")


if __name__ == "__main__":
    main()

