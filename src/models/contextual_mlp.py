from __future__ import annotations

import torch
from torch import nn


class ContextualCharacterMLP(nn.Module):
    """Prévoit le prochain caractère à partir d'une fenêtre de contexte fixe."""

    def __init__(
        self,
        vocab_size: int,
        *,
        context_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        special_tokens: int = 4,
    ) -> None:
        super().__init__()
        if vocab_size <= special_tokens:
            raise ValueError("le vocabulaire doit contenir des caractères")
        if context_size <= 0 or embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("les dimensions doivent être strictement positives")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout doit être compris entre 0 inclus et 1 exclu")

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.special_tokens = special_tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(context_size * embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != self.context_size:
            raise ValueError(
                f"le contexte doit avoir la forme (lot, {self.context_size})"
            )
        logits = self.network(self.embedding(context))
        # PAD, UNK, BOS et EOS ne sont pas des caractères à générer ici.
        logits[:, : self.special_tokens] = torch.finfo(logits.dtype).min
        return logits

    @torch.inference_mode()
    def generate(
        self,
        prefix: list[int],
        length: int,
        *,
        temperature: float,
        generator: torch.Generator,
    ) -> list[int]:
        if len(prefix) < self.context_size:
            raise ValueError("le préfixe est plus court que la fenêtre de contexte")
        if temperature <= 0:
            raise ValueError("temperature doit être strictement positive")

        self.eval()
        generated = list(prefix)
        device = next(self.parameters()).device
        for _ in range(length):
            context = torch.tensor(
                [generated[-self.context_size :]], dtype=torch.long, device=device
            )
            probabilities = torch.softmax(self(context)[0] / temperature, dim=-1)
            following = int(
                torch.multinomial(probabilities, 1, generator=generator).item()
            )
            generated.append(following)
        return generated

