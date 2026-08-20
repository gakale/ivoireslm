from __future__ import annotations

import math

import numpy as np


class BigramCountModel:
    def __init__(self, vocab_size: int, *, special_tokens: int = 4, alpha: float = 0.1):
        if vocab_size <= special_tokens:
            raise ValueError("le vocabulaire doit contenir des caractères")
        if alpha <= 0:
            raise ValueError("alpha doit être strictement positif")
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.alpha = alpha
        self.counts = np.zeros((vocab_size, vocab_size), dtype=np.uint64)
        self.log_probabilities: np.ndarray | None = None

    def add_transitions(self, current, following) -> None:
        current = np.asarray(current, dtype=np.int64)
        following = np.asarray(following, dtype=np.int64)
        if current.shape != following.shape:
            raise ValueError("les tableaux de transitions doivent avoir la même forme")
        flat = current * self.vocab_size + following
        additions = np.bincount(flat, minlength=self.vocab_size * self.vocab_size)
        self.counts += additions.reshape(self.vocab_size, self.vocab_size).astype(np.uint64)

    def fit(self) -> None:
        probabilities = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)
        real = slice(self.special_tokens, None)
        smoothed = self.counts[:, real].astype(np.float64) + self.alpha
        probabilities[:, real] = smoothed / smoothed.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore"):
            self.log_probabilities = np.log(probabilities).astype(np.float32)

    def negative_log_likelihood(self, current, following) -> float:
        if self.log_probabilities is None:
            raise RuntimeError("le modèle doit être ajusté avant l’évaluation")
        values = -self.log_probabilities[
            np.asarray(current, dtype=np.int64), np.asarray(following, dtype=np.int64)
        ]
        return float(values.mean())

    def training_negative_log_likelihood(self) -> float:
        if self.log_probabilities is None:
            raise RuntimeError("le modèle doit être ajusté avant l’évaluation")
        total = int(self.counts.sum())
        if total == 0:
            raise RuntimeError("aucune transition d’entraînement")
        finite = np.isfinite(self.log_probabilities)
        weighted = self.counts[finite].astype(np.float64) * -self.log_probabilities[finite]
        return float(weighted.sum() / total)

    @staticmethod
    def perplexity(loss: float) -> float:
        return math.exp(loss)

    def generate(self, initial_id: int, length: int, *, seed: int) -> list[int]:
        if self.log_probabilities is None:
            raise RuntimeError("le modèle doit être ajusté avant la génération")
        rng = np.random.default_rng(seed)
        generated = [initial_id]
        current = initial_id
        for _ in range(length):
            probabilities = np.exp(self.log_probabilities[current].astype(np.float64))
            following = int(rng.choice(self.vocab_size, p=probabilities / probabilities.sum()))
            generated.append(following)
            current = following
        return generated
