#!/usr/bin/env python3
"""Petite interface de routage d'un énoncé vers le moteur mathématique."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from inference.hybrid_router import route_math_request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, help="énoncé ou prompt IvoireSLM en français")
    parser.add_argument("--json", action="store_true", help="produire un objet JSON")
    args = parser.parse_args()
    result = route_math_request(args.problem)
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2) if args.json else result.completion.strip())


if __name__ == "__main__":
    main()
