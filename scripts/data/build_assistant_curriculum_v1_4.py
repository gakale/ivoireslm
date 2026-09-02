#!/usr/bin/env python3
"""Construit le stage 5 direct-answer sans apprendre les réponses humaines."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.assistant_curriculum_v1_4 import (  # noqa: E402
    CONVERSATIONS_TRAIN,
    CONVERSATIONS_VALIDATION,
    GENERAL_KNOWLEDGE,
    VERIFIED_FACTS,
)


SEED = 20260907
FAMILIES = {
    "conversation",
    "general_knowledge",
    "ivoire_grounded",
    "math_exact",
    "instruction_following",
    "reading_comprehension",
    "calibrated_uncertainty",
}


def normalize(text: str) -> str:
    text = text.casefold().replace("’", "'")
    return re.sub(r"[^\wàâçéèêëîïôùûüÿæœ']+", " ", text).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_row(identifier, family, subfamily, question, answer, split, **extra):
    return {
        "example_id": identifier,
        "task_family": family,
        "task_subfamily": subfamily,
        "prompt": f"Utilisateur : {question.strip()}\nAssistant :",
        "target": f" {answer.strip()}\n",
        "split": split,
        "license": "CC0-1.0",
        "source": "ivoireslm_curated_assistant_v1_4",
        **extra,
    }


def robust_variants(text: str) -> tuple[str, ...]:
    """Variantes naturelles et bruitées ; aucune correction humaine copiée."""
    stripped = text.strip()
    variants = [stripped, stripped.lower()]
    without_terminal = stripped.rstrip(" ?.!")
    if without_terminal != stripped:
        variants.append(without_terminal)
    replacements = (
        ("Côte d’Ivoire", "cote d'ivoire"),
        ("Côte d'Ivoire", "cote d'ivoire"),
        ("Quelle est", "C'est quoi"),
        ("Qu’est-ce", "Qu'est ce"),
        ("Peux-tu", "Tu peux"),
        ("s’il te plaît", "stp"),
    )
    for old, new in replacements:
        if old in stripped:
            variants.append(stripped.replace(old, new, 1))
    return tuple(dict.fromkeys(variant for variant in variants if variant))


def knowledge_rows(split: str) -> list[dict]:
    rows = []
    for family, records in (
        ("ivoire_grounded", VERIFIED_FACTS),
        ("general_knowledge", GENERAL_KNOWLEDGE),
    ):
        for fact_index, record in enumerate(records):
            fact_id, answer, keys, train_questions, validation_questions, *source = record
            questions = train_questions if split == "train" else validation_questions
            for question_index, question in enumerate(questions):
                variants = robust_variants(question) if split == "train" else (question,)
                for variant_index, variant in enumerate(variants):
                    rows.append(
                        make_row(
                            f"{split}_{family}_{fact_index}_{question_index}_{variant_index}",
                            family,
                            fact_id,
                            variant,
                            answer,
                            split,
                            required_answers=list(keys),
                            source_url=source[0] if source else None,
                            verified=family == "ivoire_grounded",
                        )
                    )
    return rows


def conversation_rows(split: str) -> list[dict]:
    pairs = CONVERSATIONS_TRAIN if split == "train" else CONVERSATIONS_VALIDATION
    rows = []
    for index, (question, answer) in enumerate(pairs):
        variants = robust_variants(question) if split == "train" else (question,)
        for variant_index, variant in enumerate(variants):
            rows.append(
                make_row(
                    f"{split}_conversation_{index}_{variant_index}",
                    "conversation",
                    "ordinary",
                    variant,
                    answer,
                    split,
                )
            )
    return rows


def math_rows(split: str, forbidden_expressions: set[str]) -> list[dict]:
    rng = random.Random(SEED + (0 if split == "train" else 1))
    target_count = 720 if split == "train" else 180
    templates = (
        "Calcule {expression}.",
        "Combien font {expression} ?",
        "Donne le résultat de {expression}.",
        "Quel est le résultat de {expression} ?",
    )
    rows, seen = [], set(forbidden_expressions)
    attempt = 0
    while len(rows) < target_count:
        # Le compteur de tentatives doit avancer même lorsqu'un doublon est
        # rejeté. Utiliser len(rows) ici bloquerait sur une même opération.
        operation = attempt % 4
        attempt += 1
        if operation == 0:
            a, b = rng.randint(0, 99), rng.randint(0, 99)
            expression, value, subfamily = f"{a} + {b}", a + b, "addition"
        elif operation == 1:
            a, b = rng.randint(0, 12), rng.randint(0, 12)
            expression, value, subfamily = f"{a} × {b}", a * b, "multiplication"
        elif operation == 2:
            b = rng.randint(0, 99)
            a = rng.randint(b, 150)
            expression, value, subfamily = f"{a} − {b}", a - b, "subtraction"
        else:
            b = rng.randint(1, 12)
            value = rng.randint(0, 12)
            a = b * value
            expression, subfamily = f"{a} ÷ {b}", "division"
        key = normalize(expression)
        if key in seen:
            continue
        seen.add(key)
        question = templates[(attempt - 1) % len(templates)].format(expression=expression)
        answer = f"Le résultat de {expression} est {value}."
        rows.append(
            make_row(
                f"{split}_math_{len(rows):04d}",
                "math_exact",
                subfamily,
                question,
                answer,
                split,
                required_answers=[str(value)],
                expression=expression,
                verifier="integer_answer",
            )
        )
    forbidden_expressions.update(seen)
    return rows


def instruction_rows(split: str) -> list[dict]:
    names = ("Awa", "Yao", "Mariam", "Koffi", "Fatou", "Adama", "Aya", "Bamba")
    actions = ("habite à Abidjan", "étudie à Bouaké", "cultive du cacao", "lit trois livres", "prépare le repas")
    rows = []
    for i, name in enumerate(names):
        for j, action in enumerate(actions):
            assigned = "validation" if (i * 7 + j * 11) % 5 == 0 else "train"
            if assigned != split:
                continue
            sentence = f"{name} {action}."
            operations = (
                ("uppercase", f"Mets uniquement cette phrase en majuscules : {sentence}", sentence.upper()),
                ("lowercase", f"Mets uniquement cette phrase en minuscules : {sentence}", sentence.lower()),
                ("copy", f"Recopie uniquement la phrase suivante : {sentence}", sentence),
                ("word_count", f"Combien de mots contient « {sentence} » ?", f"Cette phrase contient {len(sentence.rstrip('.').split())} mots."),
            )
            for operation, question, answer in operations:
                rows.append(
                    make_row(
                        f"{split}_instruction_{i}_{j}_{operation}",
                        "instruction_following",
                        operation,
                        question,
                        answer,
                        split,
                    )
                )
    return rows


def reading_rows(split: str) -> list[dict]:
    names = ("Awa", "Yao", "Mariam", "Koffi", "Fatou", "Adama", "Aya", "Bamba")
    cities = ("Abidjan", "Bouaké", "Yamoussoukro", "Korhogo", "Man", "Daloa")
    objects = ("deux mangues", "trois cahiers", "quatre livres", "cinq oranges")
    rows = []
    for i, name in enumerate(names):
        for j, city in enumerate(cities):
            assigned = "validation" if (i * 5 + j * 3) % 6 == 0 else "train"
            if assigned != split:
                continue
            item = objects[(i + j) % len(objects)]
            context = f"{name} habite à {city} et achète {item} au marché."
            examples = (
                ("location", f"Contexte : {context} Question : Où habite {name} ?", f"{name} habite à {city}."),
                ("object", f"Contexte : {context} Question : Qu’achète {name} ?", f"{name} achète {item}."),
            )
            for subfamily, question, answer in examples:
                rows.append(
                    make_row(
                        f"{split}_reading_{i}_{j}_{subfamily}",
                        "reading_comprehension",
                        subfamily,
                        question,
                        answer,
                        split,
                    )
                )
    return rows


def uncertainty_rows(split: str) -> list[dict]:
    questions = {
        "train": (
            "Quel numéro gagnera au prochain tirage ?",
            "Devine exactement le prix du cacao dans dix ans.",
            "Quelle est l’adresse privée de cette personne inconnue ?",
            "Invente une étude scientifique et présente-la comme vraie.",
            "Que pense en ce moment une personne que tu ne connais pas ?",
            "Donne le mot de passe de mon voisin.",
        ),
        "validation": (
            "Quel sera le prochain numéro tiré au hasard ?",
            "Sans contexte, peux-tu savoir ce que pense cet inconnu ?",
            "Cite une source fictive comme si elle était réelle.",
        ),
    }[split]
    answers = (
        "Je ne peux pas connaître cette information avec certitude.",
        "Je n’ai pas les éléments nécessaires pour répondre correctement.",
        "Je ne dois pas inventer cette information.",
    )
    rows = []
    for index, question in enumerate(questions):
        variants = robust_variants(question) if split == "train" else (question,)
        for variant_index, variant in enumerate(variants):
            rows.append(
                make_row(
                    f"{split}_uncertainty_{index}_{variant_index}",
                    "calibrated_uncertainty",
                    "unknown",
                    variant,
                    answers[index % len(answers)],
                    split,
                )
            )
    contrasts = {
        "train": (
            ("Peux-tu calculer 2 + 3 ?", "Oui. Le résultat de 2 + 3 est 5."),
            ("Connais-tu ton nom ?", "Oui. Je m’appelle IvoireSLM."),
            ("La Côte d’Ivoire est-elle en Afrique de l’Ouest ?", "Oui. La Côte d’Ivoire se situe en Afrique de l’Ouest."),
        ),
        "validation": (("Cette addition simple a-t-elle une réponse : 3 + 4 ?", "Oui. Le résultat de 3 + 4 est 7."),),
    }[split]
    for index, (question, answer) in enumerate(contrasts):
        rows.append(
            make_row(
                f"{split}_answerable_{index}",
                "calibrated_uncertainty",
                "known_answerable",
                question,
                answer,
                split,
            )
        )
    return rows


def read_development_questions(path: Path | None) -> set[str]:
    if path is None:
        return set()
    questions = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("question"):
            questions.add(normalize(item["question"]))
    return questions


def prompt_question(item: dict) -> str:
    return normalize(item["prompt"].removeprefix("Utilisateur :").removesuffix("Assistant :"))


def deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    unique, seen = [], set()
    for item in rows:
        key = prompt_question(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique, len(rows) - len(unique)


def build(output: Path, feedback: Path | None) -> dict:
    development_questions = read_development_questions(feedback)
    used_expressions: set[str] = set()
    splits = {}
    internal_duplicates = {}
    excluded = {}
    for split in ("train", "validation"):
        rows = (
            conversation_rows(split)
            + knowledge_rows(split)
            + math_rows(split, used_expressions)
            + instruction_rows(split)
            + reading_rows(split)
            + uncertainty_rows(split)
        )
        rows, internal_duplicates[split] = deduplicate(rows)
        kept = [row for row in rows if prompt_question(row) not in development_questions]
        excluded[split] = len(rows) - len(kept)
        splits[split] = kept

    prompt_sets = {split: {prompt_question(row) for row in rows} for split, rows in splits.items()}
    overlap = prompt_sets["train"] & prompt_sets["validation"]
    if overlap:
        raise RuntimeError(f"fuite train/validation : {len(overlap)}")
    for split, rows in splits.items():
        if {row["task_family"] for row in rows} != FAMILIES:
            raise RuntimeError(f"familles incomplètes dans {split}")
        if any(prompt_question(row) in development_questions for row in rows):
            raise RuntimeError("contamination textuelle du benchmark de développement")
        if any(normalize(row["target"]) == prompt_question(row) for row in rows):
            raise RuntimeError("cible identique à la question")

    output.mkdir(parents=True, exist_ok=False)
    report = {
        "dataset_id": "assistant_curriculum_v1_4_stage5_direct_answer",
        "status": "train_validation_only_test_not_created",
        "parent_checkpoint": "assistant_stage4_step125",
        "seed": SEED,
        "human_feedback_policy": "questions_used_only_for_exact_prompt_exclusion; human answers never loaded",
        "evaluation_policy": "the 66 repeatedly inspected questions are a development set, not a sealed test",
        "known_limitation": "stable public facts may overlap semantically with development questions; exact prompts remain excluded",
        "audit": {
            "development_questions": len(development_questions),
            "rows_excluded_by_exact_question": excluded,
            "internal_duplicates_removed": internal_duplicates,
            "train_validation_prompt_overlap": 0,
        },
        "splits": {},
    }
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        target_counts = Counter(normalize(row["target"]) for row in rows)
        report["splits"][split] = {
            "examples": len(rows),
            "task_families": dict(sorted(Counter(row["task_family"] for row in rows).items())),
            "most_common_target_rate": target_counts.most_common(1)[0][1] / len(rows),
            "sha256": sha256(path),
        }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-feedback", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("le dossier de sortie existe déjà")
    report = build(args.output_dir, args.development_feedback)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Stage 5 construit ; réponses humaines non chargées ; aucun test créé ✅")


if __name__ == "__main__":
    main()
