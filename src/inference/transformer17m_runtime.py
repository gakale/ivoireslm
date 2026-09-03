"""Runtime BPE partagé pour l'évaluation et l'interface du Transformer 17M."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Generation:
    text: str
    generated_tokens: int
    stopped_on_eos: bool


def load_model_module(path: Path):
    """Charge aussi les dépendances voisines, notamment v0.4 -> v0.3."""
    path = path.resolve()
    sys.path.insert(0, str(path.parent))
    try:
        specification = importlib.util.spec_from_file_location(
            "ivoireslm_transformer17m_runtime", path
        )
        if specification is None or specification.loader is None:
            raise RuntimeError(f"impossible d'importer le modèle : {path}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class Transformer17MRuntime:
    """Charge un checkpoint IvoireSLM 17M et génère avec son tokenizer BPE."""

    def __init__(self, checkpoint: Path, data_dir: Path, model_script: Path):
        import torch
        from tokenizers import Tokenizer

        self.torch = torch
        self.module = load_model_module(model_script)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw_config = payload.get("config") or payload.get("parent_config")
        if raw_config is None:
            raise KeyError("configuration Transformer absente du checkpoint")
        config = self.module.TrainingConfig(**raw_config)
        model_class = getattr(
            self.module,
            "MicroIvoireTransformer17M",
            getattr(self.module, "MicroIvoireTransformer", None),
        )
        if model_class is None:
            raise AttributeError("classe du Transformer 17M introuvable")
        self.model = model_class(config)
        self.model.load_state_dict(payload["model_state_dict"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
        self.eos_id = self.tokenizer.token_to_id("<EOS>")
        if self.eos_id is None:
            raise RuntimeError("token <EOS> absent")
        self.model_id = (
            payload.get("sft_config", {}).get("model_id")
            or raw_config.get("model_id")
            or "microivoire_transformer_17m"
        )
        self.checkpoint_step = int(payload.get("step", 0))

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_k: int = 30,
        seed: int = 20260903,
    ) -> Generation:
        if not prompt.strip():
            raise ValueError("le prompt est vide")
        if max_new_tokens <= 0 or temperature < 0 or top_k <= 0:
            raise ValueError("paramètres de génération invalides")
        formatted = prompt if "Assistant :" in prompt else f"Utilisateur : {prompt.strip()}\nAssistant :"
        prompt_ids = self.tokenizer.encode(formatted, add_special_tokens=False).ids
        generated: list[int] = []
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        stopped = False
        with self.torch.inference_mode():
            for _ in range(max_new_tokens):
                context = (prompt_ids + generated)[-self.model.config.block_size :]
                inputs = self.torch.tensor([context], dtype=self.torch.long, device=self.device)
                logits, _ = self.model(inputs)
                logits = logits[0, -1]
                if temperature == 0:
                    next_id = int(self.torch.argmax(logits).item())
                else:
                    logits = logits / temperature
                    values, _ = self.torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits[logits < values[-1]] = -float("inf")
                    probabilities = self.torch.softmax(logits, dim=-1)
                    next_id = int(
                        self.torch.multinomial(probabilities, 1, generator=generator).item()
                    )
                if next_id == self.eos_id:
                    stopped = True
                    break
                generated.append(next_id)
        return Generation(
            text=self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            generated_tokens=len(generated),
            stopped_on_eos=stopped,
        )

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt).text
