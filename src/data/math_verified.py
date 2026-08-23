from __future__ import annotations

import hashlib
import math
import random
from fractions import Fraction


FAMILIES = (
    "addition",
    "multiplication",
    "linear_equation",
    "quadratic_factorization",
    "fraction_reduction",
    "percentage",
    "rectangle",
    "arithmetic_sequence",
)


def _rng(seed: int, family: str, index: int) -> random.Random:
    material = f"{seed}:{family}:{index}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(value)


def _record(exercise_id: str, family: str, problem: str, method: str, solution: str, answer: str, verification: dict):
    text = (
        f"[MATHÉMATIQUES — {family}]\n"
        f"Problème : {problem}\n"
        f"Méthode : {method}\n"
        f"Solution : {solution}\n"
        f"Réponse : {answer}\n"
    )
    return {
        "exercise_id": exercise_id,
        "family": family,
        "problem": problem,
        "method": method,
        "solution": solution,
        "answer": answer,
        "verification": verification,
        "text": text,
    }


def generate_exercise(family: str, index: int, *, seed: int = 20260822) -> dict:
    if family not in FAMILIES:
        raise ValueError(f"famille mathématique inconnue : {family}")
    rng = _rng(seed, family, index)
    exercise_id = f"math_{family}_{index:06d}"

    if family == "addition":
        left, right = rng.randint(10, 9999), rng.randint(10, 9999)
        result = left + right
        return _record(
            exercise_id, family,
            f"Calculer {left} + {right}.",
            "Additionner les unités, puis les dizaines, les centaines et les milliers en tenant compte des retenues.",
            f"{left} + {right} = {result}.",
            str(result),
            {"left": left, "right": right, "result": result},
        )

    if family == "multiplication":
        left, right = rng.randint(2, 499), rng.randint(2, 99)
        result = left * right
        return _record(
            exercise_id, family,
            f"Calculer {left} × {right}.",
            "Décomposer un facteur, appliquer la distributivité, puis additionner les produits partiels.",
            f"{left} × {right} = {result}.",
            str(result),
            {"left": left, "right": right, "result": result},
        )

    if family == "linear_equation":
        coefficient = rng.randint(2, 15)
        root = rng.randint(-30, 30)
        constant = rng.randint(-40, 40)
        right = coefficient * root + constant
        return _record(
            exercise_id, family,
            f"Résoudre l’équation {coefficient}x + ({constant}) = {right}.",
            "Isoler le terme en x, puis diviser les deux membres par son coefficient.",
            f"{coefficient}x = {right - constant}, donc x = ({right - constant}) ÷ {coefficient} = {root}.",
            f"x = {root}",
            {"coefficient": coefficient, "constant": constant, "right": right, "root": root},
        )

    if family == "quadratic_factorization":
        first = rng.randint(-12, 12)
        second = rng.randint(-12, 12)
        while second == first:
            second = rng.randint(-12, 12)
        linear = -(first + second)
        constant = first * second
        roots = sorted((first, second))
        return _record(
            exercise_id, family,
            f"Résoudre l’équation x² + ({linear})x + ({constant}) = 0.",
            "Chercher deux racines entières dont la somme vaut l’opposé du coefficient de x et dont le produit vaut le terme constant.",
            f"x² + ({linear})x + ({constant}) = (x - ({first}))(x - ({second})). Un produit est nul si l’un de ses facteurs est nul.",
            f"x = {roots[0]} ou x = {roots[1]}",
            {"linear": linear, "constant": constant, "roots": roots},
        )

    if family == "fraction_reduction":
        numerator, denominator = rng.randint(2, 500), rng.randint(2, 500)
        factor = rng.randint(2, 20)
        numerator *= factor
        denominator *= factor
        reduced = Fraction(numerator, denominator)
        divisor = math.gcd(numerator, denominator)
        return _record(
            exercise_id, family,
            f"Réduire la fraction {numerator}/{denominator} sous forme irréductible.",
            "Calculer le plus grand commun diviseur du numérateur et du dénominateur, puis diviser les deux par ce nombre.",
            f"PGCD({numerator}, {denominator}) = {divisor}. Ainsi {numerator}/{denominator} = {reduced.numerator}/{reduced.denominator}.",
            f"{reduced.numerator}/{reduced.denominator}",
            {"numerator": numerator, "denominator": denominator, "reduced_numerator": reduced.numerator, "reduced_denominator": reduced.denominator},
        )

    if family == "percentage":
        base = 100 * rng.randint(2, 500)
        percent = rng.randint(1, 99)
        result = base * percent // 100
        return _record(
            exercise_id, family,
            f"Calculer {percent} % de {base}.",
            "Multiplier la valeur par le pourcentage, puis diviser par 100.",
            f"({percent} × {base}) ÷ 100 = {result}.",
            str(result),
            {"base": base, "percent": percent, "result": result},
        )

    if family == "rectangle":
        length, width = rng.randint(2, 100), rng.randint(2, 100)
        area, perimeter = length * width, 2 * (length + width)
        return _record(
            exercise_id, family,
            f"Un rectangle mesure {length} cm de longueur et {width} cm de largeur. Calculer son aire et son périmètre.",
            "Multiplier la longueur par la largeur pour l’aire, puis doubler leur somme pour le périmètre.",
            f"Aire = {length} × {width} = {area} cm². Périmètre = 2 × ({length} + {width}) = {perimeter} cm.",
            f"aire = {area} cm² ; périmètre = {perimeter} cm",
            {"length": length, "width": width, "area": area, "perimeter": perimeter},
        )

    first = rng.randint(-50, 50)
    difference = rng.randint(-12, 12)
    while difference == 0:
        difference = rng.randint(-12, 12)
    rank = rng.randint(5, 50)
    result = first + (rank - 1) * difference
    return _record(
        exercise_id, family,
        f"Une suite arithmétique vérifie u₁ = {first} et a pour raison {difference}. Calculer u_{rank}.",
        "Utiliser la formule uₙ = u₁ + (n − 1)r.",
        f"u_{rank} = {first} + ({rank} − 1) × ({difference}) = {result}.",
        f"u_{rank} = {result}",
        {"first": first, "difference": difference, "rank": rank, "result": result},
    )


def verify_exercise(record: dict) -> bool:
    values = record["verification"]
    family = record["family"]
    if family == "addition":
        return values["left"] + values["right"] == values["result"]
    if family == "multiplication":
        return values["left"] * values["right"] == values["result"]
    if family == "linear_equation":
        return values["coefficient"] * values["root"] + values["constant"] == values["right"]
    if family == "quadratic_factorization":
        roots = values["roots"]
        return sum(roots) == -values["linear"] and roots[0] * roots[1] == values["constant"]
    if family == "fraction_reduction":
        return Fraction(values["numerator"], values["denominator"]) == Fraction(values["reduced_numerator"], values["reduced_denominator"])
    if family == "percentage":
        return values["base"] * values["percent"] // 100 == values["result"]
    if family == "rectangle":
        return values["length"] * values["width"] == values["area"] and 2 * (values["length"] + values["width"]) == values["perimeter"]
    if family == "arithmetic_sequence":
        return values["first"] + (values["rank"] - 1) * values["difference"] == values["result"]
    return False


def split_for_exercise(exercise_id: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(exercise_id.encode("utf-8")).digest()[:4], "big") % 100
    if bucket < 5:
        return "test"
    if bucket < 10:
        return "validation"
    return "train"

