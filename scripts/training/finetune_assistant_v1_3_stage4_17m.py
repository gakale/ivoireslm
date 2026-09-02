#!/usr/bin/env python3
"""Stage 4 du micro-assistant 17M avec sélection anti-collapse."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

import finetune_instruction_sft_v03_17m as trainer


trainer.TASK_WEIGHTS = {
    "conversation": 0.20,
    "calibrated_uncertainty": 0.20,
    "ivoire_grounded": 0.15,
    "math_exact": 0.15,
    "instruction_following": 0.15,
    "reading_comprehension": 0.15,
}
trainer.EXACT_FAMILIES = {
    "ivoire_grounded", "math_exact", "instruction_following", "reading_comprehension"
}


@dataclass
class AssistantStage4Config(trainer.SFTConfig):
    model_id: str = "microivoire_transformer_v1.2_17m_assistant_stage4"
    max_steps: int = 500
    learning_rate: float = 3e-6
    minimum_learning_rate: float = 3e-7
    warmup_steps: int = 50
    raw_language_probability: float = 0.65
    evaluation_interval: int = 125
    checkpoint_interval: int = 125
    generation_examples_per_family: int = 32
    seed: int = 20260906


def normalized(text: str) -> str:
    text = text.casefold().replace("’", "'")
    return re.sub(r"[^\wàâçéèêëîïôùûüÿæœ']+", " ", text).strip()


def anti_collapse_evaluation(model, rows, tokenizer, eos_id, device, autocast_context, examples_per_family):
    selected = trainer.fixed_generation_rows(rows, examples_per_family)
    totals = defaultdict(lambda: {"examples": 0, "exact": 0, "similarity_sum": 0.0})
    samples, predictions, expectations = [], [], []
    for row in selected:
        expected_tokens = len(tokenizer.encode(row["target"], add_special_tokens=False).ids)
        prediction = trainer.greedy_generate(
            model, tokenizer, row["prompt"], eos_id, device, autocast_context,
            min(128, expected_tokens + 24),
        )
        expected_norm, prediction_norm = normalized(row["target"]), normalized(prediction)
        exact = prediction_norm == expected_norm
        similarity = SequenceMatcher(None, prediction_norm, expected_norm).ratio()
        family = row["task_family"]
        totals[family]["examples"] += 1
        totals[family]["exact"] += int(exact)
        totals[family]["similarity_sum"] += similarity
        predictions.append(prediction_norm)
        expectations.append(expected_norm)
        if len([sample for sample in samples if sample["task_family"] == family]) < 2:
            samples.append({"task_family": family, "example_id": row["example_id"], "expected": row["target"].strip(), "prediction": prediction.strip(), "exact": exact, "similarity": similarity})

    families, base_quality = {}, 0.0
    for family, values in sorted(totals.items()):
        exact_rate = values["exact"] / values["examples"]
        similarity = values["similarity_sum"] / values["examples"]
        family_quality = exact_rate if family in trainer.EXACT_FAMILIES else similarity
        families[family] = {"examples": values["examples"], "exact_rate": exact_rate, "mean_similarity": similarity, "selection_quality": family_quality}
        base_quality += trainer.TASK_WEIGHTS[family] * family_quality

    prediction_counts, expected_counts = Counter(predictions), Counter(expectations)
    predicted_unique = len(prediction_counts) / len(predictions)
    expected_unique = len(expected_counts) / len(expectations)
    predicted_dominant = prediction_counts.most_common(1)[0][1] / len(predictions)
    expected_dominant = expected_counts.most_common(1)[0][1] / len(expectations)
    diversity_gap = max(0.0, expected_unique * 0.70 - predicted_unique)
    dominance_excess = max(0.0, predicted_dominant - expected_dominant - 0.05)
    penalty = 0.75 * diversity_gap + 0.75 * dominance_excess
    return {
        "quality_score": base_quality - penalty,
        "base_task_quality": base_quality,
        "anti_collapse_penalty": penalty,
        "collapse_guard_passed": diversity_gap == 0.0 and dominance_excess == 0.0,
        "diversity": {
            "predicted_unique_rate": predicted_unique,
            "expected_unique_rate": expected_unique,
            "predicted_dominant_rate": predicted_dominant,
            "expected_dominant_rate": expected_dominant,
            "diversity_gap": diversity_gap,
            "dominance_excess": dominance_excess,
        },
        "families": families,
        "samples": samples,
    }


trainer.SFTConfig = AssistantStage4Config
trainer.evaluate_generation = anti_collapse_evaluation


if __name__ == "__main__":
    trainer.main()
