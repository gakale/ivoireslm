#!/usr/bin/env python3
"""Interface unique : moteur mathématique exact ou Transformer IvoireSLM."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from inference.hybrid_router import route_request


def load_module(path: Path):
    specification = importlib.util.spec_from_file_location("ivoireslm_transformer", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"impossible d'importer le modèle : {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class TransformerGenerator:
    """Charge paresseusement le Transformer uniquement lorsqu'il est nécessaire."""

    def __init__(self, checkpoint: Path | None, data_dir: Path | None, model_script: Path | None, args):
        self.checkpoint = checkpoint
        self.data_dir = data_dir
        self.model_script = model_script
        self.args = args
        self._loaded = None

    def _load(self):
        if self._loaded is not None:
            return self._loaded
        if not all((self.checkpoint, self.data_dir, self.model_script)):
            raise RuntimeError(
                "une question générale nécessite --checkpoint, --data-dir et --model-script"
            )
        import torch

        module = load_module(self.model_script)
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        config = module.TrainingConfig(**checkpoint["config"])
        model = module.MicroIvoireTransformer(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        tokenizer = json.loads((self.data_dir / "tokenizer.json").read_text(encoding="utf-8"))
        id_to_token = tokenizer["id_to_token"]
        if len(id_to_token) != config.vocab_size:
            raise ValueError("tokenizer incompatible avec le checkpoint")
        token_to_id = {token: index for index, token in enumerate(id_to_token)}
        self._loaded = (torch, model, device, id_to_token, token_to_id)
        return self._loaded

    def __call__(self, prompt: str) -> str:
        torch, model, device, id_to_token, token_to_id = self._load()
        unknown = token_to_id["<UNK>"]
        ids = [token_to_id.get(character, unknown) for character in prompt]
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        generator = torch.Generator(device=device).manual_seed(self.args.seed)
        with torch.inference_mode():
            for _ in range(self.args.max_new_tokens):
                logits, _ = model(tokens[:, -model.config.block_size :])
                logits = logits[:, -1, :] / self.args.temperature
                values, _ = torch.topk(logits, min(self.args.top_k, logits.shape[-1]))
                logits[logits < values[:, [-1]]] = -float("inf")
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1, generator=generator)
                tokens = torch.cat((tokens, next_token), dim=1)
        special = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
        generated = "".join(
            id_to_token[token_id]
            for token_id in tokens[0, len(ids) :].tolist()
            if id_to_token[token_id] not in special
        )
        return generated.strip()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-script", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.temperature <= 0 or args.top_k <= 0 or args.max_new_tokens <= 0:
        raise ValueError("les paramètres de génération doivent être strictement positifs")
    generator = TransformerGenerator(args.checkpoint, args.data_dir, args.model_script, args)
    result = route_request(args.request, generator)
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2) if args.json else result.response)


if __name__ == "__main__":
    main()
