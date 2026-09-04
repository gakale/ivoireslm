#!/usr/bin/env python3
"""Construit le bootstrap assistant v1.5 depuis le CPT v1.1.1.

Le curriculum réunit les compétences stables des stages précédents, un petit
socle dioula traçable et les corrections humaines explicitement consenties.
Les corrections sont filtrées puis séparées par question : une question est
soit entraînée, soit conservée dans le contrôle humain, jamais les deux.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/data"))

import build_assistant_curriculum_v1_4_1 as stage5b  # noqa: E402


SEED = 20260909
FAMILIES = set(stage5b.FAMILIES) | {"identity", "dioula_basic"}
GENERIC_CORRECTIONS = {
    "bonne reponse", "bonne réponse", "correct", "correcte", "faux",
    "oui", "non", "je ne sais pas", "aucune",
}
CATEGORY_TO_FAMILY = {
    "conversation_ordinaire": "conversation",
    "orthographe_formulation": "instruction_following",
    "connaissance_ivoirienne": "ivoire_grounded",
    "identite_limites": "identity",
    "prudence_incertain": "calibrated_uncertainty",
    "traduction": "dioula_basic",
    "instruction": "instruction_following",
    "autre": "general_knowledge",
}


def normalize(text: str) -> str:
    text = text.casefold().replace("’", "'")
    return re.sub(r"[^\wàâçéèêëîïôùûüÿæœɔɛ'+−×÷]+", " ", text).strip()


def prompt_question(row: dict) -> str:
    return normalize(row["prompt"].removeprefix("Utilisateur :").removesuffix("Assistant :"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_holdout(question: str) -> bool:
    digest = hashlib.sha256(f"{SEED}:human:{normalize(question)}".encode()).digest()
    return int.from_bytes(digest[:2], "big") % 5 == 0


def useful_correction(record: dict) -> tuple[bool, str]:
    correction = " ".join(str(record.get("human_correction") or "").split())
    if record.get("consent_for_research_and_training") is not True:
        return False, "consent_missing"
    if record.get("submitter_attested_no_personal_data") is not True:
        return False, "privacy_attestation_missing"
    if not str(record.get("question") or "").strip():
        return False, "question_missing"
    if not (8 <= len(correction) <= 1200):
        return False, "correction_length"
    if normalize(correction) in GENERIC_CORRECTIONS:
        return False, "generic_correction"
    return True, correction


def read_human_feedback(path: Path) -> tuple[list[dict], list[dict], dict]:
    accepted_by_question: dict[str, dict] = {}
    rejected = Counter()
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        record = json.loads(line)
        accepted, value = useful_correction(record)
        if not accepted:
            rejected[value] += 1
            continue
        question = " ".join(str(record["question"]).split())
        key = normalize(question)
        accepted_by_question[key] = {
            "question": question,
            "answer": value,
            "category": record.get("category", "autre"),
            "feedback_id": record.get("feedback_id"),
        }

    train, holdout = [], []
    for key in sorted(accepted_by_question):
        item = accepted_by_question[key]
        (holdout if stable_holdout(item["question"]) else train).append(item)
    return train, holdout, {
        "raw_records": total,
        "accepted_unique": len(accepted_by_question),
        "training": len(train),
        "holdout": len(holdout),
        "rejected": dict(sorted(rejected.items())),
    }


def identity_rows(split: str) -> list[dict]:
    records = {
        "train": (
            ("Comment t'appelles-tu ?", "Je m’appelle IvoireSLM.", ("ivoireslm",)),
            ("Qui es-tu ?", "Je suis IvoireSLM, un petit modèle de langage expérimental.", ("ivoireslm", "modèle")),
            ("Es-tu une personne ?", "Non. Je suis un modèle de langage, pas une personne.", ("modèle", "pas une personne")),
            ("Est-ce que tu sais tout ?", "Non. Mes connaissances sont limitées et je peux me tromper.", ("limitées", "tromper")),
            ("Quel est ton rôle ?", "Mon rôle est d’aider à répondre aux questions aussi clairement que possible.", ("aider", "questions")),
        ),
        "validation": (
            ("Quel est ton nom ?", "Mon nom est IvoireSLM.", ("ivoireslm",)),
            ("Présente-toi simplement.", "Je suis IvoireSLM, un modèle de langage expérimental.", ("ivoireslm", "modèle")),
            ("Es-tu un être humain ?", "Non. Je suis un modèle de langage.", ("non", "modèle")),
        ),
    }[split]
    rows = []
    for index, (question, answer, required) in enumerate(records):
        variants = stage5b.stage5.robust_variants(question) if split == "train" else (question,)
        for variant_index, variant in enumerate(variants):
            rows.append(stage5b.stage5.make_row(
                f"{split}_v15_identity_{index}_{variant_index}", "identity", "self",
                variant, answer, split, required_answers=list(required), verified=True,
            ))
    return rows


def dioula_rows(split: str) -> list[dict]:
    # Petit socle seulement. Les variantes régionales sont déclarées au lieu
    # de prétendre qu'une unique graphie couvre tous les usages.
    records = {
        "train": (
            ("Qu'est-ce que le dioula ?", "Le dioula, aussi appelé jula, est une langue mandingue parlée notamment en Côte d’Ivoire et au Burkina Faso.", ("langue", "mandingue"), "language_definition"),
            ("Comment dit-on bonjour le matin en dioula ?", "Une salutation courante du matin est « i ni sɔgɔma », aussi écrite « ani sogoma » selon les usages.", ("sogoma",), "morning_greeting"),
            ("Que signifie “i ni ce” en dioula ?", "« I ni ce » sert à remercier quelqu’un ; en français, cela correspond à « merci ».", ("merci",), "thanks"),
            ("Traduis merci en dioula.", "On peut dire « i ni ce ».", ("i ni ce",), "thanks"),
        ),
        "validation": (
            ("Le dioula appartient à quelle famille de langues ?", "Le dioula est une langue mandingue.", ("mandingue",), "language_definition"),
            ("Donne une salutation du matin en dioula.", "On peut dire « i ni sɔgɔma » ou « ani sogoma ».", ("sogoma",), "morning_greeting"),
            ("Comment peut-on dire merci en dioula ?", "On peut dire « i ni ce ».", ("i ni ce",), "thanks"),
        ),
    }[split]
    rows = []
    for index, (question, answer, required, subfamily) in enumerate(records):
        variants = stage5b.stage5.robust_variants(question) if split == "train" else (question,)
        for variant_index, variant in enumerate(variants):
            rows.append(stage5b.stage5.make_row(
                f"{split}_v15_dioula_{index}_{variant_index}", "dioula_basic", subfamily,
                variant, answer, split, required_answers=list(required), verified=True,
                source="Koumankan4Dyula plus project human validation",
                license="CC-BY-SA-4.0-and-private-consented-feedback",
            ))
    return rows


def human_training_rows(items: list[dict]) -> list[dict]:
    rows = []
    for index, item in enumerate(items):
        family = CATEGORY_TO_FAMILY.get(item["category"], "general_knowledge")
        variants = stage5b.stage5.robust_variants(item["question"])
        for variant_index, variant in enumerate(variants):
            rows.append(stage5b.stage5.make_row(
                f"train_v15_human_{index}_{variant_index}", family, "human_corrected",
                variant, item["answer"], "train", source="consented_project_human_feedback",
                license="project-private-consented", feedback_id=item["feedback_id"],
            ))
    return rows


def deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    kept, seen = [], set()
    for row in rows:
        key = prompt_question(row)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept, len(rows) - len(kept)


def synthetic_rows(split: str) -> list[dict]:
    return (
        stage5b.stage5.conversation_rows(split)
        + stage5b.extra_conversation_rows(split)
        + identity_rows(split)
        + stage5b.stage5.knowledge_rows(split)
        + stage5b.extra_general_rows(split)
        + dioula_rows(split)
        + stage5b.compact_math_rows(split)
        + stage5b.stage5.instruction_rows(split)
        + stage5b.stage5.reading_rows(split)
        + stage5b.calibrated_rows(split)
    )


def build(output: Path, feedback: Path) -> dict:
    human_train, human_holdout, feedback_audit = read_human_feedback(feedback)
    holdout_questions = {normalize(item["question"]) for item in human_holdout}
    training_questions = {normalize(item["question"]) for item in human_train}

    train_rows, train_duplicates = deduplicate(
        human_training_rows(human_train) + synthetic_rows("train")
    )
    validation_rows, validation_duplicates = deduplicate(synthetic_rows("validation"))

    # Le contrôle humain reste complètement hors train/validation.
    train_rows = [row for row in train_rows if prompt_question(row) not in holdout_questions]
    validation_rows = [
        row for row in validation_rows
        if prompt_question(row) not in holdout_questions | training_questions
    ]
    train_prompts = {prompt_question(row) for row in train_rows}
    validation_prompts = {prompt_question(row) for row in validation_rows}
    overlap = train_prompts & validation_prompts
    if overlap:
        raise RuntimeError(f"fuite train/validation : {len(overlap)}")
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        found = {row["task_family"] for row in rows}
        if found != FAMILIES:
            raise RuntimeError(f"familles {split} inattendues : {sorted(found)}")
        if any(normalize(row["target"]) == prompt_question(row) for row in rows):
            raise RuntimeError("cible identique à la question")

    output.mkdir(parents=True, exist_ok=False)
    splits = {"train": train_rows, "validation": validation_rows}
    report = {
        "dataset_id": "assistant_curriculum_v1.5_cpt500_bootstrap",
        "status": "train_validation_and_human_holdout_only_test_not_created",
        "parent_checkpoint": "microivoire_transformer_v1.4_17m_cpt_v111_step500",
        "seed": SEED,
        "human_feedback_policy": (
            "explicitly consented useful corrections only; deterministic question-level "
            "train/holdout split; holdout excluded from train and validation"
        ),
        "dioula_policy": "small attributed seed; expansion requires competent-speaker validation",
        "audit": {
            "feedback": feedback_audit,
            "internal_duplicates_removed": {
                "train": train_duplicates,
                "validation": validation_duplicates,
            },
            "train_validation_prompt_overlap": 0,
            "human_holdout_overlap": 0,
        },
        "splits": {},
        "test_created": False,
    }
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        report["splits"][split] = {
            "examples": len(rows),
            "task_families": dict(sorted(Counter(row["task_family"] for row in rows).items())),
            "sha256": sha256(path),
        }

    holdout_path = output / "human_holdout.jsonl"
    holdout_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in human_holdout),
        encoding="utf-8",
    )
    report["human_holdout"] = {
        "examples": len(human_holdout),
        "sha256": sha256(holdout_path),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--human-feedback", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    report = build(args.output_dir, args.human_feedback)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Curriculum assistant v1.5 construit ; test final absent ✅")


if __name__ == "__main__":
    main()
