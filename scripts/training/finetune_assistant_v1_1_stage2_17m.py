#!/usr/bin/env python3
"""Configure un stage 2 prudent pour corriger les confusions du pilote assistant."""

from __future__ import annotations

from dataclasses import dataclass

import finetune_instruction_sft_v03_17m as trainer


trainer.TASK_WEIGHTS = {
    "assistant_core": 0.20,
    "uncertainty_refusal": 0.20,
    "ivoire_grounded": 0.15,
    "reading_comprehension": 0.15,
    "instruction_following": 0.30,
}
trainer.EXACT_FAMILIES = {
    "ivoire_grounded",
    "reading_comprehension",
    "instruction_following",
}


@dataclass
class AssistantStage2Config(trainer.SFTConfig):
    model_id: str = "microivoire_transformer_v1.1_17m_assistant_stage2"
    max_steps: int = 1_000
    learning_rate: float = 7e-6
    minimum_learning_rate: float = 7e-7
    warmup_steps: int = 50
    raw_language_probability: float = 0.35
    evaluation_interval: int = 250
    checkpoint_interval: int = 250
    generation_examples_per_family: int = 20
    seed: int = 20260904


trainer.SFTConfig = AssistantStage2Config


if __name__ == "__main__":
    trainer.main()
