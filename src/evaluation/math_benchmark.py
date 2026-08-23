from __future__ import annotations

import hashlib
import math
import random
import re
from fractions import Fraction


BENCHMARK_FAMILIES = (
    "addition_large",
    "multiplication_large",
    "linear_equation_extended",
    "quadratic_extended",
    "fraction_extended",
    "percentage_extended",
    "rectangle_extended",
    "arithmetic_sequence_extended",
    "ivorian_market_change",
    "ivorian_cooperative_sharing",
)

IVORIAN_PLACES = (
    "au marché d’Adjamé",
    "au grand marché de Bouaké",
    "au marché de Korhogo",
    "au marché de Daloa",
    "au marché de San-Pédro",
    "au marché de Yamoussoukro",
)


def _rng(seed: int, family: str, index: int) -> random.Random:
    material = f"benchmark:{seed}:{family}:{index}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(value)


def _record(
    benchmark_id: str,
    family: str,
    problem: str,
    method: str,
    solution: str,
    answer: str,
    expected: dict,
    verification: dict,
) -> dict:
    prompt = f"[MATHÉMATIQUES — {family}]\nProblème : {problem}\nMéthode :"
    reference = f"{prompt} {method}\nSolution : {solution}\nRéponse : {answer}\n"
    return {
        "benchmark_id": benchmark_id,
        "family": family,
        "problem": problem,
        "prompt": prompt,
        "reference_method": method,
        "reference_solution": solution,
        "reference_answer": answer,
        "expected": expected,
        "verification": verification,
        "reference_text": reference,
    }


def generate_benchmark_exercise(
    family: str, index: int, *, seed: int = 20260824
) -> dict:
    if family not in BENCHMARK_FAMILIES:
        raise ValueError(f"famille de benchmark inconnue : {family}")
    rng = _rng(seed, family, index)
    benchmark_id = f"benchmark_{family}_{index:06d}"

    if family == "addition_large":
        left, right = rng.randint(10_000, 99_999), rng.randint(10_000, 99_999)
        result = left + right
        return _record(
            benchmark_id,
            family,
            f"Calculer {left} + {right}.",
            "Aligner les chiffres de même rang, puis effectuer l’addition avec les retenues.",
            f"{left} + {right} = {result}.",
            str(result),
            {"kind": "integer", "value": result},
            {"left": left, "right": right, "result": result},
        )

    if family == "multiplication_large":
        left, right = rng.randint(500, 2_000), rng.randint(100, 500)
        result = left * right
        return _record(
            benchmark_id,
            family,
            f"Calculer {left} × {right}.",
            "Décomposer le second facteur, calculer les produits partiels, puis les additionner.",
            f"{left} × {right} = {result}.",
            str(result),
            {"kind": "integer", "value": result},
            {"left": left, "right": right, "result": result},
        )

    if family == "linear_equation_extended":
        coefficient = rng.randint(16, 40)
        root = rng.randint(-100, 100)
        constant = rng.randint(-150, 150)
        right = coefficient * root + constant
        return _record(
            benchmark_id,
            family,
            f"Résoudre l’équation {coefficient}x + ({constant}) = {right}.",
            "Soustraire le terme constant aux deux membres, puis diviser par le coefficient de x.",
            f"{coefficient}x = {right - constant}, donc x = ({right - constant}) ÷ {coefficient} = {root}.",
            f"x = {root}",
            {"kind": "integer", "value": root},
            {"coefficient": coefficient, "constant": constant, "right": right, "root": root},
        )

    if family == "quadratic_extended":
        first = rng.choice(tuple(range(-30, -12)) + tuple(range(13, 31)))
        second = rng.randint(-30, 30)
        while second == first:
            second = rng.randint(-30, 30)
        linear = -(first + second)
        constant = first * second
        roots = sorted((first, second))
        return _record(
            benchmark_id,
            family,
            f"Résoudre l’équation x² + ({linear})x + ({constant}) = 0.",
            "Factoriser le trinôme à partir de deux nombres dont la somme et le produit correspondent aux coefficients.",
            f"x² + ({linear})x + ({constant}) = (x - ({first}))(x - ({second})).",
            f"x = {roots[0]} ou x = {roots[1]}",
            {"kind": "roots", "values": roots},
            {"linear": linear, "constant": constant, "roots": roots},
        )

    if family == "fraction_extended":
        reduced_numerator = rng.randint(501, 2_000)
        reduced_denominator = rng.randint(501, 2_000)
        reduced = Fraction(reduced_numerator, reduced_denominator)
        factor = rng.randint(21, 50)
        numerator = reduced.numerator * factor
        denominator = reduced.denominator * factor
        divisor = math.gcd(numerator, denominator)
        return _record(
            benchmark_id,
            family,
            f"Réduire la fraction {numerator}/{denominator} sous forme irréductible.",
            "Calculer le PGCD du numérateur et du dénominateur, puis diviser les deux termes par ce PGCD.",
            f"PGCD({numerator}, {denominator}) = {divisor}, donc la fraction irréductible est {reduced.numerator}/{reduced.denominator}.",
            f"{reduced.numerator}/{reduced.denominator}",
            {"kind": "fraction", "numerator": reduced.numerator, "denominator": reduced.denominator},
            {"numerator": numerator, "denominator": denominator, "reduced_numerator": reduced.numerator, "reduced_denominator": reduced.denominator},
        )

    if family == "percentage_extended":
        base = 100 * rng.randint(501, 5_000)
        percent = rng.randint(1, 99)
        result = base * percent // 100
        return _record(
            benchmark_id,
            family,
            f"Calculer {percent} % de {base}.",
            "Multiplier la valeur par le pourcentage, puis diviser le produit par 100.",
            f"({percent} × {base}) ÷ 100 = {result}.",
            str(result),
            {"kind": "integer", "value": result},
            {"base": base, "percent": percent, "result": result},
        )

    if family == "rectangle_extended":
        length, width = rng.randint(101, 500), rng.randint(101, 500)
        area, perimeter = length * width, 2 * (length + width)
        return _record(
            benchmark_id,
            family,
            f"Un rectangle mesure {length} cm de longueur et {width} cm de largeur. Calculer son aire et son périmètre.",
            "Multiplier longueur et largeur pour l’aire, puis doubler leur somme pour le périmètre.",
            f"Aire = {length} × {width} = {area} cm². Périmètre = 2 × ({length} + {width}) = {perimeter} cm.",
            f"aire = {area} cm² ; périmètre = {perimeter} cm",
            {"kind": "rectangle", "area": area, "perimeter": perimeter},
            {"length": length, "width": width, "area": area, "perimeter": perimeter},
        )

    if family == "arithmetic_sequence_extended":
        first = rng.randint(-200, 200)
        difference = rng.choice(tuple(range(-30, -12)) + tuple(range(13, 31)))
        rank = rng.randint(51, 200)
        result = first + (rank - 1) * difference
        return _record(
            benchmark_id,
            family,
            f"Une suite arithmétique vérifie u₁ = {first} et a pour raison {difference}. Calculer u_{rank}.",
            "Appliquer la formule uₙ = u₁ + (n − 1)r.",
            f"u_{rank} = {first} + ({rank} − 1) × ({difference}) = {result}.",
            f"u_{rank} = {result}",
            {"kind": "integer", "value": result},
            {"first": first, "difference": difference, "rank": rank, "result": result},
        )

    if family == "ivorian_market_change":
        place = rng.choice(IVORIAN_PLACES)
        quantity = rng.randint(3, 20)
        unit_price = 25 * rng.randint(10, 80)
        total = quantity * unit_price
        payment = 1_000 * math.ceil((total + rng.randint(500, 5_000)) / 1_000)
        change = payment - total
        return _record(
            benchmark_id,
            family,
            f"{place}, Awa achète {quantity} paniers à {unit_price} FCFA l’unité et paie {payment} FCFA. Quelle monnaie doit-on lui rendre ?",
            "Calculer le prix total des paniers, puis le soustraire au montant payé.",
            f"Prix total = {quantity} × {unit_price} = {total} FCFA. Monnaie = {payment} − {total} = {change} FCFA.",
            f"{change} FCFA",
            {"kind": "integer", "value": change},
            {"quantity": quantity, "unit_price": unit_price, "payment": payment, "total": total, "change": change},
        )

    members = rng.randint(11, 80)
    quotient = rng.randint(5, 90)
    remainder = rng.randint(1, members - 1)
    bags = members * quotient + remainder
    return _record(
        benchmark_id,
        family,
        f"Une coopérative de cacao répartit équitablement {bags} sacs entre {members} membres. Combien chaque membre reçoit-il et combien de sacs restent-ils ?",
        "Effectuer la division euclidienne du nombre de sacs par le nombre de membres.",
        f"{bags} = {members} × {quotient} + {remainder}.",
        f"{quotient} sacs par membre ; reste = {remainder} sacs",
        {"kind": "sharing", "quotient": quotient, "remainder": remainder},
        {"bags": bags, "members": members, "quotient": quotient, "remainder": remainder},
    )


def verify_benchmark_exercise(record: dict) -> bool:
    values = record["verification"]
    family = record["family"]
    if family == "addition_large":
        return values["left"] + values["right"] == values["result"]
    if family == "multiplication_large":
        return values["left"] * values["right"] == values["result"]
    if family == "linear_equation_extended":
        return values["coefficient"] * values["root"] + values["constant"] == values["right"]
    if family == "quadratic_extended":
        roots = values["roots"]
        return sum(roots) == -values["linear"] and roots[0] * roots[1] == values["constant"]
    if family == "fraction_extended":
        return Fraction(values["numerator"], values["denominator"]) == Fraction(values["reduced_numerator"], values["reduced_denominator"])
    if family == "percentage_extended":
        return values["base"] * values["percent"] // 100 == values["result"]
    if family == "rectangle_extended":
        return values["length"] * values["width"] == values["area"] and 2 * (values["length"] + values["width"]) == values["perimeter"]
    if family == "arithmetic_sequence_extended":
        return values["first"] + (values["rank"] - 1) * values["difference"] == values["result"]
    if family == "ivorian_market_change":
        return values["quantity"] * values["unit_price"] == values["total"] and values["payment"] - values["total"] == values["change"]
    if family == "ivorian_cooperative_sharing":
        return values["bags"] == values["members"] * values["quotient"] + values["remainder"] and 0 <= values["remainder"] < values["members"]
    return False


def _answer_segment(completion: str) -> str | None:
    match = re.search(r"Réponse\s*:\s*([^\n]+)", completion, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _integers(text: str) -> list[int]:
    normalized = text.replace("−", "-").replace(" ", "").replace(" ", "")
    return [int(value) for value in re.findall(r"[-+]?\d+", normalized)]


def score_completion(record: dict, completion: str) -> dict:
    segment = _answer_segment(completion)
    if segment is None:
        return {"formatted": False, "correct": False, "parsed": None}
    expected = record["expected"]
    kind = expected["kind"]
    parsed = None
    correct = False

    if kind == "fraction":
        match = re.search(r"(-?\d+)\s*/\s*(-?\d+)", segment)
        if match and int(match.group(2)) != 0:
            fraction = Fraction(int(match.group(1)), int(match.group(2)))
            parsed = {"numerator": fraction.numerator, "denominator": fraction.denominator}
            correct = parsed == {"numerator": expected["numerator"], "denominator": expected["denominator"]}
    elif kind == "roots":
        values = _integers(segment)
        if values:
            parsed = sorted(set(values))
            correct = parsed == sorted(expected["values"])
    elif kind == "rectangle":
        area_match = re.search(r"aire\s*=\s*([-+]?\d[\d\s  ]*)", segment, flags=re.IGNORECASE)
        perimeter_match = re.search(r"périmètre\s*=\s*([-+]?\d[\d\s  ]*)", segment, flags=re.IGNORECASE)
        if area_match and perimeter_match:
            area = _integers(area_match.group(1))[0]
            perimeter = _integers(perimeter_match.group(1))[0]
            parsed = {"area": area, "perimeter": perimeter}
            correct = parsed == {"area": expected["area"], "perimeter": expected["perimeter"]}
    elif kind == "sharing":
        values = _integers(segment)
        if len(values) >= 2:
            parsed = {"quotient": values[0], "remainder": values[-1]}
            correct = parsed == {"quotient": expected["quotient"], "remainder": expected["remainder"]}
    else:
        values = _integers(segment)
        if values:
            parsed = values[-1]
            correct = parsed == expected["value"]

    return {"formatted": True, "correct": correct, "parsed": parsed, "answer_segment": segment}
