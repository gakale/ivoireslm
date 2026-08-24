"""Solveur déterministe pour les dix familles mathématiques IvoireSLM.

Le solveur ne reçoit que l'énoncé. Il n'utilise jamais la réponse de référence du
benchmark : celle-ci reste réservée à l'évaluation indépendante.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from fractions import Fraction


class UnsupportedMathProblem(ValueError):
    """L'énoncé ne correspond pas exactement à une famille prise en charge."""


@dataclass(frozen=True)
class MathSolution:
    family: str
    method: str
    solution: str
    answer: str

    @property
    def completion(self) -> str:
        return f" {self.method}\nSolution : {self.solution}\nRéponse : {self.answer}\n"

    def as_dict(self) -> dict[str, str]:
        return {
            "family": self.family,
            "method": self.method,
            "solution": self.solution,
            "answer": self.answer,
            "completion": self.completion,
        }


INTEGER = r"([-+]?\d+)"
MAX_ABSOLUTE_VALUE = 10**18


def _fullmatch(pattern: str, problem: str) -> re.Match[str] | None:
    return re.fullmatch(pattern, problem.strip(), flags=re.IGNORECASE)


def _safe_arithmetic(node: ast.AST) -> Fraction:
    """Évalue un arbre arithmétique fermé, sans appel de code Python."""
    if isinstance(node, ast.Expression):
        return _safe_arithmetic(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        if abs(node.value) > MAX_ABSOLUTE_VALUE:
            raise UnsupportedMathProblem("nombre trop grand")
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_arithmetic(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _safe_arithmetic(node.left), _safe_arithmetic(node.right)
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise UnsupportedMathProblem("division par zéro")
            value = left / right
        elif isinstance(node.op, ast.Pow):
            if right.denominator != 1 or not 0 <= right.numerator <= 10:
                raise UnsupportedMathProblem("exposant non pris en charge")
            value = left ** right.numerator
        else:
            raise UnsupportedMathProblem("opérateur non pris en charge")
        if abs(value.numerator) > MAX_ABSOLUTE_VALUE or value.denominator > MAX_ABSOLUTE_VALUE:
            raise UnsupportedMathProblem("résultat trop grand")
        return value
    raise UnsupportedMathProblem("expression arithmétique non autorisée")


def _natural_arithmetic_expression(problem: str) -> tuple[str, Fraction] | None:
    """Reconnaît une expression arithmétique libre mais strictement numérique."""
    text = problem.strip().rstrip("?.!").strip()
    prefixes = (
        r"combien\s+(?:font|fait)",
        r"quel(?:le)?\s+est\s+(?:le\s+résultat|la\s+valeur)\s+de",
        r"(?:peux-tu\s+)?(?:calculer|calcule)",
        r"ça\s+fait\s+combien",
    )
    prefix = re.match(rf"(?:{'|'.join(prefixes)})\s*:?[\s]*", text, flags=re.IGNORECASE)
    expression = text[prefix.end() :].strip() if prefix else text
    if not prefix and not re.fullmatch(r"[\d\s()+\-*/×÷^]+", expression):
        return None
    normalized = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
    if not re.fullmatch(r"[\d\s()+\-*/]+", normalized) or len(normalized) > 100:
        return None
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise UnsupportedMathProblem("expression arithmétique mal formée") from exc
    return expression, _safe_arithmetic(tree)


def solve_math_problem(problem: str) -> MathSolution:
    """Analyse et résout un énoncé pris en charge, sans métadonnée externe."""
    if not isinstance(problem, str) or not problem.strip():
        raise UnsupportedMathProblem("l'énoncé doit être une chaîne non vide")
    # Uniformiser les signes typographiques fréquemment utilisés en français.
    problem = problem.replace("−", "-")

    match = _fullmatch(rf"Calculer\s+{INTEGER}\s*\+\s*{INTEGER}\s*\.", problem)
    if match:
        left, right = map(int, match.groups())
        result = left + right
        return MathSolution(
            "addition",
            "Additionner les deux nombres.",
            f"{left} + {right} = {result}.",
            str(result),
        )

    match = _fullmatch(rf"Calculer\s+{INTEGER}\s*[×x*]\s*{INTEGER}\s*\.", problem)
    if match:
        left, right = map(int, match.groups())
        result = left * right
        return MathSolution(
            "multiplication",
            "Multiplier les deux facteurs.",
            f"{left} × {right} = {result}.",
            str(result),
        )

    natural_arithmetic = _natural_arithmetic_expression(problem)
    if natural_arithmetic:
        expression, result = natural_arithmetic
        answer = (
            str(result.numerator)
            if result.denominator == 1
            else f"{result.numerator}/{result.denominator}"
        )
        return MathSolution(
            "arithmetic_expression",
            "Respecter les priorités opératoires, puis effectuer le calcul.",
            f"{expression} = {answer}.",
            answer,
        )

    match = _fullmatch(
        rf"Résoudre\s+(?:l[’']équation\s+)?{INTEGER}x\s*\+\s*\(\s*{INTEGER}\s*\)\s*=\s*{INTEGER}\s*\.",
        problem,
    )
    if match:
        coefficient, constant, right = map(int, match.groups())
        if coefficient == 0 or (right - constant) % coefficient:
            raise UnsupportedMathProblem("l'équation n'a pas de solution entière unique")
        root = (right - constant) // coefficient
        return MathSolution(
            "linear_equation",
            "Soustraire le terme constant, puis diviser par le coefficient de x.",
            f"{coefficient}x = {right - constant}, donc x = {root}.",
            f"x = {root}",
        )

    match = _fullmatch(
        rf"Résoudre\s+(?:l[’']équation\s+)?x²\s*\+\s*\(\s*{INTEGER}\s*\)x\s*\+\s*\(\s*{INTEGER}\s*\)\s*=\s*0\s*\.",
        problem,
    )
    natural_quadratic = None
    if not match:
        natural_quadratic = _fullmatch(
            r"(?:Soit\s+x\s+un\s+nombre\s+réel\.\s*)?Résoudre\s+(?:l[’']équation\s+)?"
            r"x²\s*([+-])\s*(\d+)x\s*([+-])\s*(\d+)\s*=\s*0\s*\.",
            problem,
        )
        if natural_quadratic:
            linear_sign, linear_value, constant_sign, constant_value = natural_quadratic.groups()
            linear = int(linear_value) * (-1 if linear_sign == "-" else 1)
            constant = int(constant_value) * (-1 if constant_sign == "-" else 1)
    if match:
        linear, constant = map(int, match.groups())
    if match or natural_quadratic:
        discriminant = linear * linear - 4 * constant
        if discriminant < 0:
            raise UnsupportedMathProblem("le trinôme n'a pas de racines réelles")
        square_root = math.isqrt(discriminant)
        if square_root * square_root != discriminant:
            raise UnsupportedMathProblem("les racines ne sont pas entières")
        numerators = (-linear - square_root, -linear + square_root)
        if any(value % 2 for value in numerators):
            raise UnsupportedMathProblem("les racines ne sont pas entières")
        roots = sorted({value // 2 for value in numerators})
        answer = " ou ".join(f"x = {root}" for root in roots)
        return MathSolution(
            "quadratic_factorization",
            "Calculer le discriminant, puis les racines du trinôme.",
            f"Δ = {discriminant} et √Δ = {square_root}; les racines sont {', '.join(map(str, roots))}.",
            answer,
        )

    match = _fullmatch(
        rf"Réduire(?:\s+la\s+fraction)?\s+{INTEGER}\s*/\s*{INTEGER}(?:\s+sous\s+forme\s+irréductible)?\s*\.",
        problem,
    )
    if match:
        numerator, denominator = map(int, match.groups())
        if denominator == 0:
            raise UnsupportedMathProblem("le dénominateur ne peut pas être nul")
        reduced = Fraction(numerator, denominator)
        divisor = math.gcd(numerator, denominator)
        return MathSolution(
            "fraction_reduction",
            "Diviser le numérateur et le dénominateur par leur PGCD.",
            f"PGCD({numerator}, {denominator}) = {divisor}, donc {numerator}/{denominator} = {reduced.numerator}/{reduced.denominator}.",
            f"{reduced.numerator}/{reduced.denominator}",
        )

    match = _fullmatch(rf"Calculer\s+{INTEGER}\s*%\s+de\s+{INTEGER}\s*\.", problem)
    if match:
        percent, base = map(int, match.groups())
        result = Fraction(percent * base, 100)
        if result.denominator != 1:
            raise UnsupportedMathProblem("le résultat attendu n'est pas entier")
        value = result.numerator
        return MathSolution(
            "percentage",
            "Multiplier la valeur par le pourcentage, puis diviser par 100.",
            f"({percent} × {base}) ÷ 100 = {value}.",
            str(value),
        )

    match = _fullmatch(
        rf"Un\s+rectangle\s+mesure\s+{INTEGER}\s*cm\s+de\s+longueur\s+et\s+{INTEGER}\s*cm\s+de\s+largeur\.\s*Calculer\s+son\s+aire\s+et\s+son\s+périmètre\s*\.",
        problem,
    ) or _fullmatch(
        rf"Rectangle\s+de\s+{INTEGER}\s*cm\s+sur\s+{INTEGER}\s*cm\s*:\s*aire\s+et\s+périmètre\s*\?",
        problem,
    )
    if match:
        length, width = map(int, match.groups())
        area, perimeter = length * width, 2 * (length + width)
        return MathSolution(
            "rectangle",
            "Multiplier les côtés pour l'aire et doubler leur somme pour le périmètre.",
            f"Aire = {length} × {width} = {area} cm². Périmètre = 2 × ({length} + {width}) = {perimeter} cm.",
            f"aire = {area} cm² ; périmètre = {perimeter} cm",
        )

    match = _fullmatch(
        rf"Une\s+suite\s+arithmétique\s+vérifie\s+u₁\s*=\s*{INTEGER}\s+et\s+a\s+pour\s+raison\s+{INTEGER}\s*\.\s*Calculer\s+u_(\d+)\s*\.",
        problem,
    ) or _fullmatch(
        rf"Suite\s+arithmétique\s*:\s*u₁\s*=\s*{INTEGER}\s*,\s*raison\s+{INTEGER}\s*\.\s*Calculer\s+u_(\d+)\s*\.",
        problem,
    )
    if match:
        first, difference, rank = map(int, match.groups())
        if rank < 1:
            raise UnsupportedMathProblem("le rang doit être positif")
        result = first + (rank - 1) * difference
        return MathSolution(
            "arithmetic_sequence",
            "Appliquer la formule uₙ = u₁ + (n − 1)r.",
            f"u_{rank} = {first} + ({rank} − 1) × ({difference}) = {result}.",
            f"u_{rank} = {result}",
        )

    match = re.fullmatch(
        rf".+?,\s*Awa\s+achète\s+{INTEGER}\s+paniers\s+à\s+{INTEGER}\s+FCFA(?:\s+l[’']unité)?\s+et\s+paie\s+{INTEGER}\s+FCFA\.\s*(?:Quelle\s+monnaie\s+doit-on\s+lui\s+rendre\s*\?|Monnaie\s*\?)",
        problem.strip(),
        flags=re.IGNORECASE,
    )
    if match:
        quantity, unit_price, payment = map(int, match.groups())
        total = quantity * unit_price
        change = payment - total
        if change < 0:
            raise UnsupportedMathProblem("le paiement est inférieur au prix total")
        return MathSolution(
            "ivorian_market_change",
            "Calculer le prix total, puis le soustraire au montant payé.",
            f"Prix total = {quantity} × {unit_price} = {total} FCFA. Monnaie = {payment} − {total} = {change} FCFA.",
            f"{change} FCFA",
        )

    match = _fullmatch(
        rf"Une\s+coopérative(?:\s+de\s+cacao)?\s+(?:répartit\s+équitablement|partage)\s+{INTEGER}\s+sacs\s+entre\s+{INTEGER}\s+membres\.\s*(?:Combien\s+chaque\s+membre\s+reçoit-il\s+et\s+combien\s+de\s+sacs\s+restent-ils\s*\?|Part\s+et\s+reste\s*\?)",
        problem,
    )
    if match:
        bags, members = map(int, match.groups())
        if members <= 0:
            raise UnsupportedMathProblem("le nombre de membres doit être positif")
        quotient, remainder = divmod(bags, members)
        return MathSolution(
            "ivorian_cooperative_sharing",
            "Effectuer la division euclidienne du nombre de sacs par le nombre de membres.",
            f"{bags} = {members} × {quotient} + {remainder}.",
            f"{quotient} sacs par membre ; reste = {remainder} sacs",
        )

    raise UnsupportedMathProblem(f"énoncé mathématique non reconnu : {problem!r}")
