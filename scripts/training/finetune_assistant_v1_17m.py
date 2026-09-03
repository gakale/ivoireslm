#!/usr/bin/env python3
"""Configure le SFT v0.3 pour le curriculum micro-assistant 17M v1."""

from __future__ import annotations

from dataclasses import dataclass

import finetune_instruction_sft_v03_17m as trainer


trainer.TASK_WEIGHTS = {
    "assistant_core": 0.20,
    "uncertainty_refusal": 0.20,
    "ivoire_grounded": 0.20,
    "reading_comprehension": 0.20,
    "instruction_following": 0.20,
}
trainer.EXACT_FAMILIES = {
    "ivoire_grounded",
    "reading_comprehension",
    "instruction_following",
}


@dataclass
class AssistantSFTConfig(trainer.SFTConfig):
    model_id: str = "microivoire_transformer_v1.1_17m_assistant_pilot"
    max_steps: int = 1_000
    learning_rate: float = 1e-5
    minimum_learning_rate: float = 1e-6
    warmup_steps: int = 50
    raw_language_probability: float = 0.30
    evaluation_interval: int = 250
    checkpoint_interval: int = 250
    generation_examples_per_family: int = 20
    seed: int = 20260903


trainer.SFTConfig = AssistantSFTConfig


if __name__ == "__main__":
    trainer.main()
