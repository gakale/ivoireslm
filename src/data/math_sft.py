from __future__ import annotations

import hashlib
import math
import random
from fractions import Fraction


SFT_FAMILIES = (
    "addition",
    "multiplication",
    "linear_equation",
    "quadratic_factorization",
    "fraction_reduction",
    "percentage",
    "rectangle",
    "arithmetic_sequence",
    "ivorian_market_change",
    "ivorian_cooperative_sharing",
)

PLACES = ("Adjamé", "Bouaké", "Korhogo", "Daloa", "San-Pédro", "Yamoussoukro")


def _rng(seed: int, family: str, index: int) -> random.Random:
    digest = hashlib.sha256(f"sft:{seed}:{family}:{index}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _record(identifier: str, family: str, difficulty: int, problem: str, method: str, solution: str, answer: str, verification: dict) -> dict:
    prompt = f"[MATHÉMATIQUES — {family}]\nProblème : {problem}\nMéthode :"
    target = f" {method}\nSolution : {solution}\nRéponse : {answer}\n"
    return {
        "example_id": identifier,
        "family": family,
        "difficulty": difficulty,
        "problem": problem,
        "prompt": prompt,
        "target": target,
        "text": prompt + target,
        "answer": answer,
        "verification": verification,
    }


def generate_sft_exercise(family: str, index: int, *, seed: int = 20260826) -> dict:
    if family not in SFT_FAMILIES:
        raise ValueError(f"famille SFT inconnue : {family}")
    rng = _rng(seed, family, index)
    difficulty = index % 4 + 1
    identifier = f"math_sft_{family}_{index:07d}"

    if family == "addition":
        maximum = (99, 999, 9_999, 999_999)[difficulty - 1]
        minimum = (0, 10, 100, 10_000)[difficulty - 1]
        left, right = rng.randint(minimum, maximum), rng.randint(minimum, maximum)
        result = left + right
        return _record(identifier, family, difficulty, f"Calculer {left} + {right}.", "Additionner les deux nombres.", f"{left} + {right} = {result}.", str(result), {"left": left, "right": right, "result": result})

    if family == "multiplication":
        left_max = (20, 99, 499, 9_999)[difficulty - 1]
        right_max = (10, 20, 99, 999)[difficulty - 1]
        left, right = rng.randint(2, left_max), rng.randint(2, right_max)
        result = left * right
        return _record(identifier, family, difficulty, f"Calculer {left} × {right}.", "Multiplier les deux facteurs.", f"{left} × {right} = {result}.", str(result), {"left": left, "right": right, "result": result})

    if family == "linear_equation":
        coefficient_max = (10, 20, 50, 100)[difficulty - 1]
        root_bound = (20, 50, 150, 500)[difficulty - 1]
        constant_bound = (20, 100, 500, 1_000)[difficulty - 1]
        coefficient = rng.randint(2, coefficient_max)
        root = rng.randint(-root_bound, root_bound)
        constant = rng.randint(-constant_bound, constant_bound)
        right = coefficient * root + constant
        return _record(identifier, family, difficulty, f"Résoudre {coefficient}x + ({constant}) = {right}.", "Isoler x puis diviser.", f"{coefficient}x = {right - constant}, donc x = {root}.", f"x = {root}", {"coefficient": coefficient, "constant": constant, "right": right, "root": root})

    if family == "quadratic_factorization":
        bound = (10, 20, 50, 100)[difficulty - 1]
        first, second = rng.randint(-bound, bound), rng.randint(-bound, bound)
        while second == first:
            second = rng.randint(-bound, bound)
        roots = sorted((first, second))
        linear, constant = -(first + second), first * second
        return _record(identifier, family, difficulty, f"Résoudre x² + ({linear})x + ({constant}) = 0.", "Factoriser puis annuler chaque facteur.", f"(x - ({first}))(x - ({second})) = 0.", f"x = {roots[0]} ou x = {roots[1]}", {"linear": linear, "constant": constant, "roots": roots})

    if family == "fraction_reduction":
        bound = (50, 200, 1_000, 10_000)[difficulty - 1]
        numerator, denominator = rng.randint(2, bound), rng.randint(2, bound)
        factor = rng.randint(2, (10, 20, 50, 100)[difficulty - 1])
        numerator *= factor
        denominator *= factor
        reduced = Fraction(numerator, denominator)
        divisor = math.gcd(numerator, denominator)
        return _record(identifier, family, difficulty, f"Réduire {numerator}/{denominator}.", "Diviser par le PGCD.", f"PGCD = {divisor}, donc {numerator}/{denominator} = {reduced.numerator}/{reduced.denominator}.", f"{reduced.numerator}/{reduced.denominator}", {"numerator": numerator, "denominator": denominator, "reduced_numerator": reduced.numerator, "reduced_denominator": reduced.denominator})

    if family == "percentage":
        base = 100 * rng.randint(1, (50, 500, 2_000, 10_000)[difficulty - 1])
        percent = rng.randint(1, 99)
        result = base * percent // 100
        return _record(identifier, family, difficulty, f"Calculer {percent} % de {base}.", "Multiplier puis diviser par 100.", f"({percent} × {base}) ÷ 100 = {result}.", str(result), {"base": base, "percent": percent, "result": result})

    if family == "rectangle":
        bound = (20, 100, 500, 1_000)[difficulty - 1]
        length, width = rng.randint(2, bound), rng.randint(2, bound)
        area, perimeter = length * width, 2 * (length + width)
        return _record(identifier, family, difficulty, f"Rectangle de {length} cm sur {width} cm : aire et périmètre ?", "Appliquer A = L×l et P = 2(L+l).", f"A = {area} cm² et P = {perimeter} cm.", f"aire = {area} cm² ; périmètre = {perimeter} cm", {"length": length, "width": width, "area": area, "perimeter": perimeter})

    if family == "arithmetic_sequence":
        bound = (20, 50, 200, 1_000)[difficulty - 1]
        difference_bound = (5, 12, 30, 100)[difficulty - 1]
        rank_max = (20, 50, 200, 500)[difficulty - 1]
        first = rng.randint(-bound, bound)
        difference = rng.randint(-difference_bound, difference_bound)
        while difference == 0:
            difference = rng.randint(-difference_bound, difference_bound)
        rank = rng.randint(2, rank_max)
        result = first + (rank - 1) * difference
        return _record(identifier, family, difficulty, f"Suite arithmétique : u₁ = {first}, raison {difference}. Calculer u_{rank}.", "Utiliser uₙ = u₁ + (n−1)r.", f"u_{rank} = {first} + ({rank}−1)×({difference}) = {result}.", f"u_{rank} = {result}", {"first": first, "difference": difference, "rank": rank, "result": result})

    if family == "ivorian_market_change":
        place = rng.choice(PLACES)
        quantity = rng.randint(2, (10, 20, 50, 100)[difficulty - 1])
        unit_price = 25 * rng.randint(2, (20, 80, 200, 400)[difficulty - 1])
        total = quantity * unit_price
        payment = 500 * math.ceil((total + rng.randint(100, 5_000)) / 500)
        change = payment - total
        return _record(identifier, family, difficulty, f"À {place}, Awa achète {quantity} paniers à {unit_price} FCFA et paie {payment} FCFA. Monnaie ?", "Soustraire le prix total du paiement.", f"{payment} − ({quantity} × {unit_price}) = {change} FCFA.", f"{change} FCFA", {"quantity": quantity, "unit_price": unit_price, "payment": payment, "total": total, "change": change})

    members = rng.randint(2, (20, 50, 100, 200)[difficulty - 1])
    quotient = rng.randint(1, (20, 50, 200, 500)[difficulty - 1])
    remainder = rng.randint(0, members - 1)
    bags = members * quotient + remainder
    return _record(identifier, family, difficulty, f"Une coopérative partage {bags} sacs entre {members} membres. Part et reste ?", "Faire la division euclidienne.", f"{bags} = {members} × {quotient} + {remainder}.", f"{quotient} sacs par membre ; reste = {remainder} sacs", {"bags": bags, "members": members, "quotient": quotient, "remainder": remainder})


def verify_sft_exercise(record: dict) -> bool:
    v, family = record["verification"], record["family"]
    if family == "addition":
        return v["left"] + v["right"] == v["result"]
    if family == "multiplication":
        return v["left"] * v["right"] == v["result"]
    if family == "linear_equation":
        return v["coefficient"] * v["root"] + v["constant"] == v["right"]
    if family == "quadratic_factorization":
        return sum(v["roots"]) == -v["linear"] and v["roots"][0] * v["roots"][1] == v["constant"]
    if family == "fraction_reduction":
        return Fraction(v["numerator"], v["denominator"]) == Fraction(v["reduced_numerator"], v["reduced_denominator"])
    if family == "percentage":
        return v["base"] * v["percent"] // 100 == v["result"]
    if family == "rectangle":
        return v["length"] * v["width"] == v["area"] and 2 * (v["length"] + v["width"]) == v["perimeter"]
    if family == "arithmetic_sequence":
        return v["first"] + (v["rank"] - 1) * v["difference"] == v["result"]
    if family == "ivorian_market_change":
        return v["quantity"] * v["unit_price"] == v["total"] and v["payment"] - v["total"] == v["change"]
    if family == "ivorian_cooperative_sharing":
        return v["bags"] == v["members"] * v["quotient"] + v["remainder"] and 0 <= v["remainder"] < v["members"]
    return False
