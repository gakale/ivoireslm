#!/usr/bin/env python3
"""Stage 6B : rattrapage ciblé après le pilote Stage 6 sous-supervisé."""

from __future__ import annotations

from dataclasses import dataclass

import finetune_assistant_v1_5_stage6_17m as stage6


trainer = stage6.trainer

# Le Stage 6 consacrait seulement 40 % des lots à neuf compétences. Ce palier
# court augmente le signal supervisé sans abandonner la répétition du corpus
# général. Les deux compétences déjà fortes restent présentes mais minoritaires.
trainer.TASK_WEIGHTS = {
    "calibrated_uncertainty": 0.08,
    "conversation": 0.17,
    "dioula_basic": 0.08,
    "general_knowledge": 0.15,
    "identity": 0.10,
    "instruction_following": 0.06,
    "ivoire_grounded": 0.16,
    "math_exact": 0.16,
    "reading_comprehension": 0.04,
}


@dataclass
class AssistantStage6BConfig(trainer.SFTConfig):
    model_id: str = "microivoire_transformer_v1.6_17m_assistant_stage6b"
    max_steps: int = 250
    learning_rate: float = 4e-6
    minimum_learning_rate: float = 4e-7
    warmup_steps: int = 20
    raw_language_probability: float = 0.25
    evaluation_interval: int = 125
    checkpoint_interval: int = 125
    log_interval: int = 25
    generation_examples_per_family: int = 24
    seed: int = 20260910


trainer.SFTConfig = AssistantStage6BConfig
trainer.evaluate_generation = stage6.stage6_evaluation


if __name__ == "__main__":
    trainer.main()
