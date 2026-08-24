"""Routeur minimal entre une requête IvoireSLM et le moteur mathématique."""
from __future__ import annotations

import re
from dataclasses import dataclass

from tools.math_solver import MathSolution, solve_math_problem


@dataclass(frozen=True)
class MathRouteResult:
    route: str
    problem: str
    solution: MathSolution

    @property
    def completion(self) -> str:
        return self.solution.completion

    def as_dict(self) -> dict:
        return {
            "route": self.route,
            "problem": self.problem,
            **self.solution.as_dict(),
        }


def extract_problem(request: str) -> str:
    """Extrait l'énoncé d'un prompt IvoireSLM ou conserve l'énoncé direct."""
    if not isinstance(request, str) or not request.strip():
        raise ValueError("la requête doit être une chaîne non vide")
    match = re.fullmatch(
        r"\s*\[MATHÉMATIQUES\s+—\s+[^\]\n]+\]\s*\n"
        r"Problème\s*:\s*(.+?)\s*\nMéthode\s*:\s*",
        request,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else request.strip()


def route_math_request(request: str) -> MathRouteResult:
    """Route une requête mathématique vers le solveur, sans réponse attendue."""
    problem = extract_problem(request)
    return MathRouteResult(
        route="deterministic_math_tool_v0.1",
        problem=problem,
        solution=solve_math_problem(problem),
    )
