#!/usr/bin/env python3
"""Construit le correctif Stage 5B ciblé sur les confusions observées."""

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
sys.path.insert(0, str(ROOT / "scripts/data"))

import build_assistant_curriculum_v1_4 as stage5  # noqa: E402


SEED = 20260908
FAMILIES = stage5.FAMILIES


def normalize(text: str) -> str:
    """Normalise le texte sans effacer l'opération mathématique."""
    text = text.casefold().replace("’", "'")
    return re.sub(r"[^\wàâçéèêëîïôùûüÿæœ'+−×÷]+", " ", text).strip()


def prompt_question(item: dict) -> str:
    return normalize(item["prompt"].removeprefix("Utilisateur :").removesuffix("Assistant :"))


def read_development_questions(path: Path) -> set[str]:
    questions = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("question"):
            questions.add(normalize(item["question"]))
    return questions


def deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    unique, seen = [], set()
    for row in rows:
        key = prompt_question(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique, len(rows) - len(unique)


EXTRA_CONVERSATIONS = {
    "train": (
        ("Ça va ?", "Oui, merci. Et toi, comment vas-tu ?"),
        ("Moi ça va bien.", "Content de l’apprendre !"),
        ("Je suis fatigué.", "Je comprends. Essaie de prendre un peu de repos si tu le peux."),
        ("Je suis content aujourd’hui.", "C’est une bonne nouvelle !"),
        ("On commence ?", "Oui, je suis prêt. Quelle est ta question ?"),
        ("Aide-moi s’il te plaît.", "Bien sûr. Explique-moi ce que tu veux faire."),
        ("Tu peux répéter ?", "Oui. Indique-moi ce que tu veux que je répète."),
        ("Fais plus court.", "D’accord, je vais être plus bref."),
        ("Explique davantage.", "D’accord, je vais ajouter des détails utiles."),
        ("Je ne suis pas d’accord.", "D’accord. Dis-moi quel point tu contestes."),
        ("C’est intéressant.", "Oui. Veux-tu approfondir ce sujet ?"),
        ("Quel est ton rôle ?", "Mon rôle est d’aider à répondre aux questions, avec des limites."),
        ("Est-ce que tu sais tout ?", "Non. Mes connaissances sont limitées et je peux me tromper."),
        ("Réponds à ma question.", "D’accord. Pose ta question précisément."),
        ("Bonjour, tu vas bien ?", "Bonjour ! Oui, merci. Et toi ?"),
        ("Slt", "Salut ! Comment puis-je t’aider ?"),
        ("Bjr", "Bonjour ! Comment puis-je t’aider ?"),
        ("merci hein", "Avec plaisir !"),
        ("On se parle demain.", "D’accord, à demain !"),
        ("Bonne nuit.", "Bonne nuit et repose-toi bien !"),
    ),
    "validation": (
        ("Comment ça va aujourd’hui ?", "Je vais bien, merci. Et toi ?"),
        ("Je suis heureux.", "C’est une bonne nouvelle !"),
        ("Tu peux m’aider un peu ?", "Oui. Explique-moi ce dont tu as besoin."),
        ("Réponds simplement.", "D’accord, je vais répondre simplement."),
        ("Quel travail fais-tu ?", "Je suis un modèle de langage conçu pour aider à répondre aux questions."),
        ("À demain.", "D’accord, à demain !"),
    ),
}


EXTRA_GENERAL = (
    ("computer", "Un ordinateur est une machine électronique qui traite et stocke des informations.", ("ordinateur", "machine"), ("C’est quoi un ordinateur ?", "Définis simplement un ordinateur."), ("Comment expliquer ce qu’est un ordinateur ?",)),
    ("artificial_intelligence", "L’intelligence artificielle regroupe des techniques qui permettent à des machines d’effectuer certaines tâches associées à l’intelligence humaine.", ("machines", "tâches"), ("C’est quoi l’intelligence artificielle ?", "Explique simplement l’intelligence artificielle."), ("Comment définir l’IA ?",)),
    ("search_engine", "Un moteur de recherche aide à trouver des pages et des informations sur Internet.", ("trouver", "internet"), ("À quoi sert un moteur de recherche ?", "Définis un moteur de recherche."), ("Que permet de faire un moteur de recherche ?",)),
    ("science", "La science étudie le monde par l’observation, le raisonnement et la vérification des résultats.", ("observation", "vérification"), ("C’est quoi la science ?", "Explique la science en une phrase."), ("Comment peut-on définir la science ?",)),
    ("history", "L’histoire étudie les sociétés humaines et les événements du passé à partir de sources.", ("passé", "sources"), ("C’est quoi l’histoire ?", "Que cherche à étudier l’histoire ?"), ("Comment définir l’histoire ?",)),
    ("geography", "La géographie étudie les territoires, les populations et leurs relations avec l’environnement.", ("territoires", "populations"), ("C’est quoi la géographie ?", "Que fait la géographie ?"), ("Quel est l’objet de la géographie ?",)),
    ("democracy", "La démocratie est un système politique dans lequel le pouvoir vient des citoyens, notamment par le vote.", ("citoyens", "vote"), ("C’est quoi la démocratie ?", "Explique la démocratie simplement."), ("Comment définir une démocratie ?",)),
    ("dictionary", "Un dictionnaire rassemble des mots et donne des informations sur leur sens, leur orthographe ou leur usage.", ("mots", "sens"), ("À quoi sert un dictionnaire ?", "C’est quoi un dictionnaire ?"), ("Que trouve-t-on dans un dictionnaire ?",)),
    ("book", "Un livre est un ouvrage composé de pages, imprimé ou numérique, destiné à être lu.", ("pages", "lu"), ("C’est quoi un livre ?", "Définis le mot livre."), ("Comment expliquer ce qu’est un livre ?",)),
    ("water", "L’eau est une substance indispensable à la vie, constituée de molécules de formule H₂O.", ("vie", "h₂o"), ("C’est quoi l’eau ?", "Pourquoi l’eau est-elle importante ?"), ("De quoi l’eau est-elle constituée ?",)),
)


def extra_conversation_rows(split: str) -> list[dict]:
    rows = []
    for index, (question, answer) in enumerate(EXTRA_CONVERSATIONS[split]):
        variants = stage5.robust_variants(question) if split == "train" else (question,)
        for variant_index, variant in enumerate(variants):
            rows.append(stage5.make_row(
                f"{split}_recovery_conversation_{index}_{variant_index}",
                "conversation", "ordinary_recovery", variant, answer, split,
            ))
    return rows


def extra_general_rows(split: str) -> list[dict]:
    rows = []
    for index, (fact_id, answer, keys, train_questions, validation_questions) in enumerate(EXTRA_GENERAL):
        questions = train_questions if split == "train" else validation_questions
        for question_index, question in enumerate(questions):
            variants = stage5.robust_variants(question) if split == "train" else (question,)
            for variant_index, variant in enumerate(variants):
                rows.append(stage5.make_row(
                    f"{split}_recovery_general_{index}_{question_index}_{variant_index}",
                    "general_knowledge", fact_id, variant, answer, split,
                    required_answers=list(keys), verified=False,
                ))
    return rows


def compact_math_rows(split: str) -> list[dict]:
    """Tables apprises comme faits, avec formulations train/validation disjointes."""
    facts = []
    for a in range(21):
        for b in range(21):
            facts.append((f"{a} + {b}", a + b, "addition_basic"))
    for a in range(13):
        for b in range(13):
            facts.append((f"{a} × {b}", a * b, "multiplication_table"))
    for a in range(21):
        for b in range(a + 1):
            facts.append((f"{a} − {b}", a - b, "subtraction_basic"))
    rng = random.Random(SEED)
    rng.shuffle(facts)
    if split == "validation":
        facts = facts[:210]
    templates = (
        "Calcule {expression}.",
        "Combien font {expression} ?",
        "Donne seulement le résultat de {expression}.",
    ) if split == "train" else (
        "Sans utiliser de calculatrice, combien vaut {expression} ?",
    )
    answer_templates = (
        "{expression} = {value}.",
        "Cela fait {value}.",
        "La réponse est {value}.",
    )
    rows = []
    for index, (expression, value, subfamily) in enumerate(facts):
        for template_index, template in enumerate(templates):
            rows.append(stage5.make_row(
                f"{split}_recovery_math_{index}_{template_index}",
                "math_exact", subfamily,
                template.format(expression=expression),
                answer_templates[(index + template_index) % len(answer_templates)].format(
                    expression=expression, value=value
                ),
                split,
                required_answers=[str(value)], expression=expression,
                verifier="integer_answer", skill_overlap_declared=True,
            ))
    return rows


def calibrated_rows(split: str) -> list[dict]:
    unknown = {
        "train": (
            "Quel numéro sortira exactement au prochain tirage ?",
            "Que pense actuellement une personne inconnue ?",
            "Donne-moi une source inventée en disant qu’elle existe.",
            "Quel sera exactement le prix du cacao dans dix ans ?",
        ),
        "validation": (
            "Peux-tu prévoir avec certitude le prochain tirage ?",
            "Sans information, peux-tu connaître la pensée d’un inconnu ?",
            "Peux-tu garantir le prix exact du cacao en 2036 ?",
        ),
    }[split]
    answers = (
        "Je ne peux pas le savoir avec certitude.",
        "Je n’ai pas assez d’informations pour répondre correctement.",
        "Je ne dois pas inventer une source.",
    )
    known = {
        "train": (
            ("Combien font 2 + 2 ?", "2 + 2 = 4."),
            ("Quelle est la capitale politique ivoirienne ?", "La capitale politique de la Côte d’Ivoire est Yamoussoukro."),
            ("C’est quoi Internet ?", "Internet est un réseau mondial qui relie des appareils."),
            ("Comment t’appelles-tu ?", "Je m’appelle IvoireSLM."),
            ("La Côte d’Ivoire est-elle en Afrique de l’Ouest ?", "Oui, la Côte d’Ivoire se situe en Afrique de l’Ouest."),
            ("Quelle monnaie utilise la Côte d’Ivoire ?", "La Côte d’Ivoire utilise le franc CFA, code XOF."),
            ("Peux-tu définir un ordinateur ?", "Un ordinateur est une machine électronique qui traite des informations."),
            ("Combien font 10 + 10 ?", "10 + 10 = 20."),
        ),
        "validation": (
            ("La question 3 + 3 possède-t-elle une réponse certaine ?", "Oui. 3 + 3 = 6."),
            ("Sais-tu dans quelle région se trouve la Côte d’Ivoire ?", "Oui. Elle se trouve en Afrique de l’Ouest."),
            ("Peux-tu dire ton nom ?", "Oui. Je m’appelle IvoireSLM."),
        ),
    }[split]
    rows = []
    for index, question in enumerate(unknown):
        rows.append(stage5.make_row(
            f"{split}_recovery_unknown_{index}", "calibrated_uncertainty",
            "unknown", question, answers[index % len(answers)], split,
        ))
    for index, (question, answer) in enumerate(known):
        rows.append(stage5.make_row(
            f"{split}_recovery_known_{index}", "calibrated_uncertainty",
            "known_answerable", question, answer, split,
        ))
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output: Path, feedback: Path) -> dict:
    development_questions = read_development_questions(feedback)
    splits = {}
    removed = {}
    for split in ("train", "validation"):
        rows = (
            stage5.conversation_rows(split)
            + extra_conversation_rows(split)
            + stage5.knowledge_rows(split)
            + extra_general_rows(split)
            + compact_math_rows(split)
            + stage5.instruction_rows(split)
            + stage5.reading_rows(split)
            + calibrated_rows(split)
        )
        rows, duplicate_count = deduplicate(rows)
        kept = [row for row in rows if prompt_question(row) not in development_questions]
        removed[split] = {
            "internal_duplicates": duplicate_count,
            "development_exact_prompts": len(rows) - len(kept),
        }
        splits[split] = kept
    prompts = {
        split: {prompt_question(row) for row in rows}
        for split, rows in splits.items()
    }
    overlap = prompts["train"] & prompts["validation"]
    if overlap:
        raise RuntimeError(f"fuite textuelle train/validation : {len(overlap)}")
    for split, rows in splits.items():
        if {row["task_family"] for row in rows} != FAMILIES:
            raise RuntimeError(f"familles incomplètes : {split}")
        if any(prompt_question(row) in development_questions for row in rows):
            raise RuntimeError("question de développement non exclue")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "dataset_id": "assistant_curriculum_v1_4_1_stage5b_targeted_recovery",
        "status": "train_validation_only_test_not_created",
        "parent_checkpoint": "assistant_stage5_step250",
        "seed": SEED,
        "human_feedback_policy": "questions used only for exact exclusion; answers never loaded",
        "validation_policy": "paraphrase validation; stable facts and arithmetic facts may overlap semantically",
        "audit": {
            "development_questions": len(development_questions),
            "removed": removed,
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
        report["splits"][split] = {
            "examples": len(rows),
            "task_families": dict(sorted(Counter(row["task_family"] for row in rows).items())),
            "sha256": sha256(path),
        }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-feedback", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    print(json.dumps(build(args.output_dir, args.development_feedback), ensure_ascii=False, indent=2))
    print("Stage 5B construit ; aucun test créé ; aucune réponse humaine chargée ✅")


if __name__ == "__main__":
    main()
