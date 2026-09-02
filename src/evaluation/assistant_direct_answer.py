"""Mesures pures de réponse directe utilisées par l'assistant Stage 5."""

from __future__ import annotations

import re


def normalized(text: str) -> str:
    text = text.casefold().replace("’", "'")
    return re.sub(r"[^\wàâçéèêëîïôùûüÿæœ']+", " ", text).strip()


def question_from_prompt(prompt: str) -> str:
    return normalized(prompt.removeprefix("Utilisateur :").removesuffix("Assistant :"))


def required_answer_success(row: dict, prediction: str) -> bool:
    prediction_norm = normalized(prediction)
    required = [normalized(value) for value in row.get("required_answers", [])]
    if row["task_family"] == "math_exact":
        numbers = re.findall(r"(?<!\w)-?\d+(?!\w)", prediction_norm)
        return bool(numbers) and bool(required) and numbers[-1] == required[-1]
    if required:
        return all(value in prediction_norm for value in required)
    return prediction_norm == normalized(row["target"])


def is_echo_fragment(question: str, prediction: str, expected: str) -> bool:
    question_norm = normalized(question)
    prediction_norm = normalized(prediction)
    expected_norm = normalized(expected)
    if not prediction_norm:
        return True
    if prediction_norm == question_norm:
        return True
    prediction_words = prediction_norm.split()
    return (
        len(prediction_words) <= 5
        and prediction_norm in question_norm
        and prediction_norm != expected_norm
    )

