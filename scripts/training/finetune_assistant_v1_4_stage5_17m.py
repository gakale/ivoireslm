#!/usr/bin/env python3
"""Stage 5 : réponses directes, complètes et pénalisées en cas d'écho."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import sys

import finetune_instruction_sft_v03_17m as trainer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.assistant_direct_answer import (  # noqa: E402
    is_echo_fragment,
    normalized,
    question_from_prompt,
    required_answer_success,
)


trainer.TASK_WEIGHTS = {
    "conversation": 0.16,
    "general_knowledge": 0.14,
    "ivoire_grounded": 0.20,
    "math_exact": 0.18,
    "instruction_following": 0.12,
    "reading_comprehension": 0.10,
    "calibrated_uncertainty": 0.10,
}
trainer.EXACT_FAMILIES = {
    "general_knowledge",
    "ivoire_grounded",
    "math_exact",
    "instruction_following",
    "reading_comprehension",
}


@dataclass
class AssistantStage5Config(trainer.SFTConfig):
    model_id: str = "microivoire_transformer_v1.3_17m_assistant_stage5"
    max_steps: int = 375
    learning_rate: float = 8e-6
    minimum_learning_rate: float = 8e-7
    warmup_steps: int = 40
    raw_language_probability: float = 0.50
    evaluation_interval: int = 125
    checkpoint_interval: int = 125
    log_interval: int = 25
    generation_examples_per_family: int = 32
    seed: int = 20260907


def direct_answer_evaluation(
    model,
    rows,
    tokenizer,
    eos_id,
    device,
    autocast_context,
    examples_per_family,
):
    selected = trainer.fixed_generation_rows(rows, examples_per_family)
    totals = defaultdict(
        lambda: {
            "examples": 0,
            "success": 0,
            "exact": 0,
            "similarity_sum": 0.0,
            "echo": 0,
            "empty_or_one_word": 0,
        }
    )
    predictions, expectations, samples = [], [], []
    for row in selected:
        expected_tokens = len(tokenizer.encode(row["target"], add_special_tokens=False).ids)
        prediction = trainer.greedy_generate(
            model,
            tokenizer,
            row["prompt"],
            eos_id,
            device,
            autocast_context,
            min(128, max(32, expected_tokens + 32)),
        ).strip()
        expected = row["target"].strip()
        prediction_norm, expected_norm = normalized(prediction), normalized(expected)
        exact = prediction_norm == expected_norm
        semantic_success = required_answer_success(row, prediction)
        similarity = SequenceMatcher(None, prediction_norm, expected_norm).ratio()
        echo = is_echo_fragment(question_from_prompt(row["prompt"]), prediction, expected)
        too_short = len(prediction_norm.split()) <= 1
        family = row["task_family"]
        value = totals[family]
        value["examples"] += 1
        value["success"] += int(semantic_success)
        value["exact"] += int(exact)
        value["similarity_sum"] += similarity
        value["echo"] += int(echo)
        value["empty_or_one_word"] += int(too_short)
        predictions.append(prediction_norm)
        expectations.append(expected_norm)
        if len([item for item in samples if item["task_family"] == family]) < 3:
            samples.append(
                {
                    "task_family": family,
                    "example_id": row["example_id"],
                    "expected": expected,
                    "prediction": prediction,
                    "semantic_success": semantic_success,
                    "exact": exact,
                    "echo_fragment": echo,
                }
            )

    families, base_quality = {}, 0.0
    semantic_families = {"general_knowledge", "ivoire_grounded", "math_exact"}
    exact_families = {"instruction_following", "reading_comprehension"}
    for family, value in sorted(totals.items()):
        examples = value["examples"]
        success_rate = value["success"] / examples
        exact_rate = value["exact"] / examples
        similarity = value["similarity_sum"] / examples
        if family in semantic_families:
            selection_quality = success_rate
        elif family in exact_families:
            selection_quality = exact_rate
        else:
            selection_quality = similarity
        families[family] = {
            "examples": examples,
            "semantic_success_rate": success_rate,
            "exact_rate": exact_rate,
            "mean_similarity": similarity,
            "echo_fragment_rate": value["echo"] / examples,
            "empty_or_one_word_rate": value["empty_or_one_word"] / examples,
            "selection_quality": selection_quality,
        }
        base_quality += trainer.TASK_WEIGHTS[family] * selection_quality

    prediction_counts = Counter(predictions)
    expected_counts = Counter(expectations)
    predicted_unique = len(prediction_counts) / len(predictions)
    expected_unique = len(expected_counts) / len(expectations)
    predicted_dominant = prediction_counts.most_common(1)[0][1] / len(predictions)
    expected_dominant = expected_counts.most_common(1)[0][1] / len(expectations)
    echo_rate = sum(value["echo"] for value in totals.values()) / len(predictions)
    too_short_rate = sum(value["empty_or_one_word"] for value in totals.values()) / len(predictions)
    diversity_gap = max(0.0, expected_unique * 0.70 - predicted_unique)
    dominance_excess = max(0.0, predicted_dominant - expected_dominant - 0.05)
    penalty = (
        0.75 * diversity_gap
        + 0.75 * dominance_excess
        + 0.50 * echo_rate
        + 0.25 * too_short_rate
    )
    return {
        "quality_score": base_quality - penalty,
        "base_task_quality": base_quality,
        "direct_answer_penalty": penalty,
        "direct_answer_guard_passed": echo_rate <= 0.05 and too_short_rate <= 0.05,
        "collapse_guard_passed": diversity_gap == 0.0 and dominance_excess == 0.0,
        "echo_fragment_rate": echo_rate,
        "empty_or_one_word_rate": too_short_rate,
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


trainer.SFTConfig = AssistantStage5Config
trainer.evaluate_generation = direct_answer_evaluation


if __name__ == "__main__":
    trainer.main()
