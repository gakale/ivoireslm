#!/usr/bin/env python3
"""Construit le stage 4 anti-collapse, sans réutiliser le benchmark humain."""

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

from data.assistant_curriculum_v1_3 import (
    CONVERSATIONS, FACTS, UNKNOWN_TRAIN, UNKNOWN_VALIDATION,
    VALIDATION_CONVERSATIONS,
)

SEED = 20260906


def normalize(text: str) -> str:
    text = text.casefold().replace("’", "'")
    return re.sub(r"[^\wàâçéèêëîïôùûüÿæœ']+", " ", text).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row(identifier, family, subfamily, prompt, target, split, **extra):
    return {
        "example_id": identifier,
        "task_family": family,
        "task_subfamily": subfamily,
        "prompt": f"Utilisateur : {prompt.strip()}\nAssistant :",
        "target": f" {target.strip()}\n",
        "split": split,
        "license": "CC0-1.0",
        "source": "ivoireslm_curated_assistant_v1_3",
        **extra,
    }


def typo_variants(text: str) -> tuple[str, ...]:
    variants = [text]
    replacements = (("Quelle", "Quel"), ("Est-ce que", "Es ce que"), ("Côte d’Ivoire", "cote d'ivoire"), ("s’il te plaît", "stp"))
    for old, new in replacements:
        if old in text:
            variants.append(text.replace(old, new, 1))
    words = text.split()
    if len(words) >= 5:
        position = min(2, len(words) - 1)
        word = words[position]
        if len(word) >= 5:
            words[position] = word[:2] + word[3] + word[2] + word[4:]
            variants.append(" ".join(words))
    return tuple(dict.fromkeys(variants))


def conversation_rows(split: str) -> list[dict]:
    pairs = CONVERSATIONS if split == "train" else VALIDATION_CONVERSATIONS
    prefixes = ("", "S'il te plaît, ", "Hé, ") if split == "train" else ("",)
    rows = []
    for index, (prompt, answer) in enumerate(pairs):
        variants = typo_variants(prompt) if split == "train" else (prompt,)
        for variant_index, variant in enumerate(variants):
            for prefix_index, prefix in enumerate(prefixes):
                rows.append(row(f"{split}_chat_{index}_{variant_index}_{prefix_index}", "conversation", "ordinary", prefix + variant, answer, split))
    return rows


def fact_rows(split: str) -> list[dict]:
    rows = []
    for index, (fact_id, answer, train_question, validation_question, source_url) in enumerate(FACTS):
        question = train_question if split == "train" else validation_question
        variants = typo_variants(question) if split == "train" else (question,)
        for variant_index, variant in enumerate(variants):
            target = answer if variant_index % 2 == 0 or split == "validation" else f"Réponse : {answer}."
            rows.append(row(f"{split}_fact_{index}_{variant_index}", "ivoire_grounded", fact_id, variant, target, split, source_url=source_url, verified=True))
    return rows


def math_rows(split: str) -> list[dict]:
    rng = random.Random(SEED + (0 if split == "train" else 1))
    rows = []
    count = 900 if split == "train" else 180
    seen = set()
    while len(rows) < count:
        operation = len(rows) % 3
        if operation == 0:
            low, high = (0, 199) if split == "train" else (200, 399)
            a, b = rng.randint(low, high), rng.randint(low, high)
            expression, answer, name = f"{a} + {b}", a + b, "addition"
        elif operation == 1:
            low, high = (2, 30) if split == "train" else (31, 50)
            a, b = rng.randint(low, high), rng.randint(2, 20)
            expression, answer, name = f"{a} × {b}", a * b, "multiplication"
        else:
            low, high = (20, 300) if split == "train" else (301, 500)
            a, b = rng.randint(low, high), rng.randint(0, 19)
            expression, answer, name = f"{a} − {b}", a - b, "subtraction"
        key = (expression, split)
        if key in seen:
            continue
        seen.add(key)
        template = ("Calcule {x}.", "Combien font {x} ?", "Donne le résultat de {x}.")[len(rows) % 3]
        rows.append(row(f"{split}_math_{len(rows):04d}", "math_exact", name, template.format(x=expression), str(answer), split, verifier="integer_exact"))
    return rows


def instruction_rows(split: str) -> list[dict]:
    subjects = ("Awa", "Yao", "Mariam", "Koffi", "une étudiante", "le cultivateur", "la coopérative", "le professeur")
    predicates = ("travaille à Abidjan.", "étudie à Bouaké.", "vérifie cette information.", "cultive du cacao.", "explique la leçon.", "prépare trois cahiers.")
    rows = []
    for i, subject in enumerate(subjects):
        for j, predicate in enumerate(predicates):
            sentence = f"{subject} {predicate}"
            assigned = "validation" if (i * 7 + j * 11) % 5 == 0 else "train"
            if assigned != split:
                continue
            operations = (
                ("uppercase", f"Mets en majuscules : {sentence}", sentence.upper()),
                ("lowercase", f"Mets en minuscules : {sentence}", sentence.lower()),
                ("copy", f"Recopie exactement : {sentence}", sentence),
                ("word_count", f"Combien de mots contient cette phrase : {sentence}", str(len(sentence.rstrip('.').split()))),
            )
            for operation, prompt, target in operations:
                rows.append(row(f"{split}_instruction_{i}_{j}_{operation}", "instruction_following", operation, prompt, target, split))
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
            obj = objects[(i + j) % len(objects)]
            context = f"{name} habite à {city} et achète {obj} au marché."
            rows.append(row(f"{split}_reading_city_{i}_{j}", "reading_comprehension", "location", f"Contexte : {context} Question : Où habite {name} ?", city, split))
            rows.append(row(f"{split}_reading_object_{i}_{j}", "reading_comprehension", "object", f"Contexte : {context} Question : Qu’achète {name} ?", obj, split))
    return rows


def calibration_rows(split: str) -> list[dict]:
    unknown = UNKNOWN_TRAIN if split == "train" else UNKNOWN_VALIDATION
    answers = (
        "Je ne peux pas le savoir avec certitude.",
        "Je ne dois pas inventer cette information.",
        "Je n’ai pas assez d’éléments pour répondre correctement.",
    )
    rows = [row(f"{split}_unknown_{i}", "calibrated_uncertainty", "unknown", prompt, answers[i % len(answers)], split) for i, prompt in enumerate(unknown)]
    # Contrastes : une question simple et vérifiable ne doit pas déclencher un refus.
    contrasts = {
        "train": (
            ("Peux-tu répondre directement à 2 + 2 ?", "Oui : 2 + 2 = 4."),
            ("La Côte d’Ivoire est-elle située en Afrique de l’Ouest ?", "Oui, la Côte d’Ivoire se trouve en Afrique de l’Ouest."),
            ("Sais-tu dire ton propre nom ?", "Oui. Je m’appelle IvoireSLM."),
        ),
        "validation": (
            ("Cette question simple a-t-elle une réponse : 2 + 3 ?", "Oui : 2 + 3 = 5."),
        ),
    }
    for i, (prompt, answer) in enumerate(contrasts[split]):
        rows.append(row(f"{split}_known_{i}", "calibrated_uncertainty", "known_answerable", prompt, answer, split))
    return rows


def read_holdout(path: Path | None) -> set[str]:
    if path is None:
        return set()
    values = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        question = item.get("question")
        if question:
            values.add(normalize(question))
    return values


def prompt_text(item: dict) -> str:
    return normalize(item["prompt"].removeprefix("Utilisateur :").removesuffix("Assistant :"))


def exclude_holdout(splits: dict[str, list[dict]], holdout: set[str]) -> tuple[dict[str, list[dict]], dict[str, int]]:
    dropped = {}
    filtered = {}
    for split, rows in splits.items():
        filtered[split] = [item for item in rows if prompt_text(item) not in holdout]
        dropped[split] = len(rows) - len(filtered[split])
    return filtered, dropped


def validate(splits: dict[str, list[dict]], holdout: set[str], dropped: dict[str, int]) -> dict:
    expected = {"conversation", "ivoire_grounded", "math_exact", "instruction_following", "reading_comprehension", "calibrated_uncertainty"}
    prompt_sets = {}
    identifiers = set()
    for split, rows in splits.items():
        if {row["task_family"] for row in rows} != expected:
            raise RuntimeError(f"familles incomplètes : {split}")
        prompt_sets[split] = set()
        for item in rows:
            if item["example_id"] in identifiers:
                raise RuntimeError(f"identifiant dupliqué : {item['example_id']}")
            identifiers.add(item["example_id"])
            prompt = prompt_text(item)
            if prompt in prompt_sets[split]:
                raise RuntimeError(f"prompt interne dupliqué : {item['example_id']}")
            if prompt in holdout:
                raise RuntimeError(f"contamination du benchmark humain : {item['example_id']}")
            prompt_sets[split].add(prompt)
    overlap = prompt_sets["train"] & prompt_sets["validation"]
    if overlap:
        raise RuntimeError(f"fuite train/validation : {len(overlap)}")
    targets = Counter(normalize(row["target"]) for row in splits["train"])
    return {
        "human_holdout_questions": len(holdout),
        "rows_excluded_by_human_holdout": dropped,
        "train_validation_prompt_overlap": 0,
        "most_common_train_target_count": targets.most_common(1)[0][1],
        "most_common_train_target_rate": targets.most_common(1)[0][1] / len(splits["train"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-out-feedback", type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("le dossier de sortie doit être vide")
    output.mkdir(parents=True, exist_ok=True)
    splits = {split: conversation_rows(split) + fact_rows(split) + math_rows(split) + instruction_rows(split) + reading_rows(split) + calibration_rows(split) for split in ("train", "validation")}
    holdout = read_holdout(args.held_out_feedback)
    splits, dropped = exclude_holdout(splits, holdout)
    audit = validate(splits, holdout, dropped)
    report = {
        "dataset_id": "assistant_curriculum_v1_3_stage4_anti_collapse",
        "status": "train_validation_only_test_not_created",
        "parent_checkpoint": "microivoire_transformer_v1.0_17m_cpt_pilot_step1000",
        "seed": SEED,
        "human_feedback_policy": "held_out_benchmark_only_never_training",
        "audit": audit,
        "splits": {},
    }
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rows), encoding="utf-8")
        report["splits"][split] = {"examples": len(rows), "task_families": dict(sorted(Counter(row["task_family"] for row in rows).items())), "sha256": sha256(path)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Stage 4 anti-collapse construit ; benchmark humain exclu ; aucun test créé ✅")


if __name__ == "__main__":
    main()
