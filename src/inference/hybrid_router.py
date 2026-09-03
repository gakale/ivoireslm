"""Routeur minimal entre une requête IvoireSLM et le moteur mathématique."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from tools.math_solver import MathSolution, UnsupportedMathProblem, solve_math_problem


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


@dataclass(frozen=True)
class HybridResponse:
    route: str
    response: str
    verified: bool
    detail: str | None = None

    def as_dict(self) -> dict:
        return {
            "route": self.route,
            "response": self.response,
            "verified": self.verified,
            "detail": self.detail,
        }


MATH_INTENT = re.compile(
    r"(?:\[MATHÉMATIQUES|\bcalculer\b|\bcombien\s+(?:font|fait)\b|"
    r"\bça\s+fait\s+combien\b|\brésoudre\b|\béquation\b|\bfraction\b|"
    r"\bpourcentage\b|\brectangle\b|\bsuite\s+arithmétique\b|\bmonnaie\b|"
    r"\bcoopérative\b|\d\s*[+*/×÷-]\s*\d)",
    flags=re.IGNORECASE,
)


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


def route_request(request: str, language_generator: Callable[[str], str]) -> HybridResponse:
    """Route vers le calcul exact ou vers le modèle de langue.

    Une demande clairement mathématique mais non prise en charge ne tombe jamais
    silencieusement vers le Transformer : elle reçoit un refus explicite afin
    d'éviter un résultat numérique inventé.
    """
    try:
        result = route_math_request(request)
        return HybridResponse(
            route=result.route,
            response=result.completion.strip(),
            verified=True,
        )
    except UnsupportedMathProblem as exc:
        if MATH_INTENT.search(request):
            return HybridResponse(
                route="unsupported_math_guard",
                response=(
                    "Je reconnais une demande mathématique, mais ce type d’exercice "
                    "n’est pas encore pris en charge par mon moteur de calcul exact."
                ),
                verified=True,
                detail=str(exc),
            )
    return HybridResponse(
        route=getattr(language_generator, "model_id", "microivoire_transformer_v0.2_5m"),
        response=language_generator(request),
        verified=False,
    )
