#!/usr/bin/env python3
"""Stage 3 contrastif du micro-assistant 17M."""

from dataclasses import dataclass

import finetune_instruction_sft_v03_17m as trainer


trainer.TASK_WEIGHTS = {
    "assistant_core": 0.20,
    "uncertainty_refusal": 0.20,
    "ivoire_grounded": 0.10,
    "reading_comprehension": 0.15,
    "instruction_following": 0.35,
}
trainer.EXACT_FAMILIES = {"ivoire_grounded", "reading_comprehension", "instruction_following"}


@dataclass
class AssistantStage3Config(trainer.SFTConfig):
    model_id: str = "microivoire_transformer_v1.1_17m_assistant_stage3"
    max_steps: int = 750
    learning_rate: float = 5e-6
    minimum_learning_rate: float = 5e-7
    warmup_steps: int = 40
    raw_language_probability: float = 0.40
    evaluation_interval: int = 250
    checkpoint_interval: int = 250
    generation_examples_per_family: int = 20
    seed: int = 20260905


trainer.SFTConfig = AssistantStage3Config


if __name__ == "__main__":
    trainer.main()
