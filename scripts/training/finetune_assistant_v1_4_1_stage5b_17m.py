#!/usr/bin/env python3
"""Stage 5B : réduit les confusions entre conversation, refus et calcul."""

from __future__ import annotations

from dataclasses import dataclass

import finetune_assistant_v1_4_stage5_17m as stage5


trainer = stage5.trainer
trainer.TASK_WEIGHTS = {
    "conversation": 0.20,
    "general_knowledge": 0.20,
    "ivoire_grounded": 0.25,
    "math_exact": 0.15,
    "instruction_following": 0.08,
    "reading_comprehension": 0.07,
    "calibrated_uncertainty": 0.05,
}
trainer.EXACT_FAMILIES = {
    "general_knowledge",
    "ivoire_grounded",
    "math_exact",
    "instruction_following",
    "reading_comprehension",
}


@dataclass
class AssistantStage5BConfig(trainer.SFTConfig):
    model_id: str = "microivoire_transformer_v1.3_17m_assistant_stage5b"
    max_steps: int = 250
    learning_rate: float = 5e-6
    minimum_learning_rate: float = 5e-7
    warmup_steps: int = 25
    raw_language_probability: float = 0.55
    evaluation_interval: int = 125
    checkpoint_interval: int = 125
    log_interval: int = 25
    generation_examples_per_family: int = 32
    seed: int = 20260908


trainer.SFTConfig = AssistantStage5BConfig
trainer.evaluate_generation = stage5.direct_answer_evaluation


if __name__ == "__main__":
    trainer.main()
