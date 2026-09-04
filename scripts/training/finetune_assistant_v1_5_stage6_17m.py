#!/usr/bin/env python3
"""Stage 6 : bootstrap assistant équilibré depuis le CPT v1.1.1 étape 500."""

from __future__ import annotations

from dataclasses import dataclass

import finetune_assistant_v1_4_stage5_17m as stage5


trainer = stage5.trainer
trainer.TASK_WEIGHTS = {
    "conversation": 0.14,
    "identity": 0.06,
    "general_knowledge": 0.14,
    "ivoire_grounded": 0.17,
    "dioula_basic": 0.05,
    "math_exact": 0.18,
    "instruction_following": 0.10,
    "reading_comprehension": 0.08,
    "calibrated_uncertainty": 0.08,
}
trainer.EXACT_FAMILIES = {
    "identity",
    "general_knowledge",
    "ivoire_grounded",
    "dioula_basic",
    "math_exact",
    "instruction_following",
    "reading_comprehension",
}


def stage6_evaluation(*args, **kwargs):
    """Rend identité et dioula stricts via leurs éléments obligatoires."""
    result = stage5.direct_answer_evaluation(*args, **kwargs)
    for family in ("identity", "dioula_basic"):
        values = result["families"][family]
        values["selection_quality"] = values["semantic_success_rate"]
    base_quality = sum(
        trainer.TASK_WEIGHTS[family] * values["selection_quality"]
        for family, values in result["families"].items()
    )
    result["base_task_quality"] = base_quality
    result["quality_score"] = base_quality - result["direct_answer_penalty"]
    return result


@dataclass
class AssistantStage6Config(trainer.SFTConfig):
    model_id: str = "microivoire_transformer_v1.5_17m_assistant_stage6"
    max_steps: int = 250
    learning_rate: float = 6e-6
    minimum_learning_rate: float = 6e-7
    warmup_steps: int = 25
    raw_language_probability: float = 0.60
    evaluation_interval: int = 125
    checkpoint_interval: int = 125
    log_interval: int = 25
    generation_examples_per_family: int = 24
    seed: int = 20260909


trainer.SFTConfig = AssistantStage6Config
trainer.evaluate_generation = stage6_evaluation


if __name__ == "__main__":
    trainer.main()
