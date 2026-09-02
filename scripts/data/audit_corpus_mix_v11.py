#!/usr/bin/env python3
"""Audite un supplément avant toute continuation de préentraînement v1.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/corpus_mix_v11_candidate.json"


def compute_audit(report: dict, config: dict) -> dict:
    domains = report.get("domain_characters", {})
    total = int(report.get("characters", 0))
    if total <= 0:
        raise ValueError("le rapport ne contient aucun caractère")

    shares = {name: int(value) / total for name, value in sorted(domains.items())}
    natural_french = int(domains.get("natural_french_open", 0))
    natural_ivoirian = int(domains.get("natural_ivoirian_conversation_verified", 0))
    natural_language = sum(
        int(value) for name, value in domains.items() if name.startswith("natural_")
    )
    gates = config["supplement_admission_gates"]
    checks = {
        "minimum_natural_french_share": natural_french / total
        >= gates["minimum_natural_french_share"],
        "minimum_natural_language_share": natural_language / total
        >= gates["minimum_natural_language_share"],
        "maximum_mathematics_share": shares.get("mathematics_reasoning", 0.0)
        <= gates["maximum_mathematics_share"],
        "maximum_code_agents_share": shares.get("code_agents", 0.0)
        <= gates["maximum_code_agents_share"],
        "maximum_cybersecurity_share": shares.get("cybersecurity_defensive", 0.0)
        <= gates["maximum_cybersecurity_share"],
        "minimum_natural_french_characters": natural_french
        >= gates["minimum_natural_french_characters"],
        "minimum_ivoirian_conversation_characters": natural_ivoirian
        >= gates["minimum_ivoirian_conversation_characters"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "dataset_id": report.get("dataset_id"),
        "mixture_id": config["mixture_id"],
        "characters": total,
        "domain_characters": domains,
        "domain_shares": shares,
        "natural_language_share": natural_language / total,
        "checks": checks,
        "failed_checks": failed,
        "training_authorized": not failed,
        "test_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit = compute_audit(report, config)
    rendered = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refus d'écraser {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not audit["training_authorized"]:
        print("ENTRAÎNEMENT BLOQUÉ : le supplément est déséquilibré.")


if __name__ == "__main__":
    main()
