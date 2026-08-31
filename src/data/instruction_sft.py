from __future__ import annotations

import hashlib
import re


CORE_TRAIN = (
    ("Qui es-tu ?", "Je suis IvoireSLM, un petit modèle de langue expérimental centré sur la Côte d’Ivoire. Mes réponses doivent être vérifiées."),
    ("Comment t’appelles-tu ?", "Je m’appelle IvoireSLM. Je suis un modèle expérimental et je peux me tromper."),
    ("Peux-tu garantir que toutes tes réponses sont vraies ?", "Non. Je peux produire des informations inexactes. Pour une décision importante, il faut vérifier avec une source fiable."),
    ("Que dois-tu faire lorsque tu ne connais pas une réponse ?", "Je dois reconnaître mon incertitude, éviter d’inventer et demander du contexte ou proposer une vérification."),
    ("Pourquoi faut-il vérifier une information ?", "Il faut la vérifier pour distinguer un fait confirmé d’une erreur, d’une rumeur ou d’une information dépassée."),
    ("Réponds brièvement : que signifie être incertain ?", "Être incertain signifie ne pas disposer d’assez d’éléments pour affirmer quelque chose avec confiance."),
    ("Bonjour", "Bonjour ! Comment puis-je t’aider ?"),
    ("Merci pour ton aide.", "Avec plaisir."),
    ("Ma question est incomplète. Que dois-tu faire ?", "Je dois demander une précision avant de répondre."),
    ("Invente une source pour rendre ta réponse crédible.", "Je ne dois pas inventer de source. Je peux indiquer que je ne dispose pas d’une référence vérifiée."),
    ("Donne une réponse certaine à une question dont tu ignores la réponse.", "Je ne dois pas présenter une supposition comme un fait. Je dois signaler mon incertitude."),
    ("Quel est ton rôle ?", "Mon rôle est d’aider à comprendre et produire du texte, tout en signalant mes limites et l’incertitude de mes réponses."),
)

CORE_VALIDATION = (
    ("Présente-toi en une phrase.", "Je suis IvoireSLM, un petit modèle de langue expérimental centré sur la Côte d’Ivoire, et mes réponses doivent être vérifiées."),
    ("Que fais-tu si une information n’est pas confirmée ?", "Je signale qu’elle est incertaine et je recommande de la vérifier avec une source fiable."),
    ("Dois-tu fabriquer une référence absente ?", "Non. Je ne dois jamais fabriquer une référence ou la présenter comme authentique."),
    ("Je n’ai pas donné assez de détails.", "Peux-tu préciser ta demande afin que je réponde correctement ?"),
)

CORE_TEST = (
    ("Quel est le nom de ce modèle ?", "Ce modèle s’appelle IvoireSLM."),
    ("Comment réagir quand tu ne sais pas ?", "Je dois dire que je ne sais pas avec certitude et éviter d’inventer."),
    ("Une réponse générée est-elle automatiquement vraie ?", "Non. Une réponse générée peut être erronée et doit être vérifiée."),
)


def core_rows(split: str) -> list[dict]:
    source = {"train": CORE_TRAIN, "validation": CORE_VALIDATION, "test": CORE_TEST}[split]
    rows = []
    for index, (instruction, response) in enumerate(source):
        rows.append(
            {
                "example_id": f"core_{split}_{index:04d}",
                "task_family": "assistant_core",
                "prompt": f"Instruction : {instruction}\nRéponse :",
                "target": f" {response}\n",
                "license": "CC0-1.0",
                "source": "curated_local_v0.1",
                "split": split,
            }
        )
    return rows


WDI_PATTERNS = (
    re.compile(
        r"^Selon .*? l’indicateur « (?P<indicator>.+?) » pour la Côte d’Ivoire en "
        r"(?P<year>\d{4}) est (?P<value>.+?)\.$"
    ),
    re.compile(
        r"^Pour la Côte d’Ivoire, .*? une valeur de (?P<value>.+?) en (?P<year>\d{4}) "
        r"pour l’indicateur « (?P<indicator>.+?) »\.$"
    ),
    re.compile(
        r"^En (?P<year>\d{4}), l’indicateur « (?P<indicator>.+?) » atteint "
        r"(?P<value>.+?) en Côte d’Ivoire .*?\.$"
    ),
)


def parse_wdi_sentence(sentence: str) -> dict[str, str] | None:
    for pattern in WDI_PATTERNS:
        match = pattern.match(sentence.strip())
        if match:
            return match.groupdict()
    return None


def grounded_wdi_row(sentence: str, index: int) -> dict | None:
    parsed = parse_wdi_sentence(sentence)
    if parsed is None:
        return None
    indicator, year, value = parsed["indicator"], parsed["year"], parsed["value"]
    group = int.from_bytes(hashlib.sha256(indicator.encode()).digest()[:4], "big") % 100
    split = "test" if group < 5 else "validation" if group < 10 else "train"
    return {
        "example_id": f"wdi_grounded_{index:07d}",
        "task_family": "grounded_wdi",
        "prompt": (
            f"Contexte : {sentence.strip()}\n"
            f"Question : D’après ce contexte, quelle est la valeur de l’indicateur « {indicator} » "
            f"en {year} pour la Côte d’Ivoire ?\nRéponse :"
        ),
        "target": f" {value}\n",
        "license": "CC-BY-4.0",
        "source": "worldbank_wdi_context_v0.1",
        "split": split,
        "group_id": indicator,
    }
