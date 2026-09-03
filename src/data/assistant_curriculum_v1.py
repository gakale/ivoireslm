"""Données contrôlées du micro-assistant 17M v1.

Les textes de réponse sont rédigés pour IvoireSLM. Les URLs servent uniquement
à tracer la vérification des faits stables.
"""

from __future__ import annotations


FACTS = (
    {
        "id": "capital",
        "answer": "Yamoussoukro est la capitale politique et administrative de la Côte d’Ivoire.",
        "train_questions": (
            "Quelle est la capitale de la Côte d’Ivoire ?",
            "Quelle ville est la capitale politique ivoirienne ?",
            "Donne la capitale administrative de la Côte d’Ivoire.",
        ),
        "validation_questions": (
            "Comment s’appelle la capitale politique du pays ?",
            "La capitale de la Côte d’Ivoire, c’est quelle ville ?",
        ),
        "source_url": "https://www.diplomatie.gouv.ci/informations-utiles/presentation-de-la-c%C3%B4te-d-ivoire",
    },
    {
        "id": "economic_capital",
        "answer": "Abidjan est la capitale économique de la Côte d’Ivoire.",
        "train_questions": (
            "Quelle est la capitale économique de la Côte d’Ivoire ?",
            "Quelle ville joue le rôle de capitale économique ivoirienne ?",
            "Donne la capitale économique du pays.",
        ),
        "validation_questions": (
            "La capitale économique ivoirienne est quelle ville ?",
            "Comment s’appelle le principal centre économique du pays ?",
        ),
        "source_url": "https://www.eco.diplomatie.gouv.ci/",
    },
    {
        "id": "official_language",
        "answer": "Le français est la langue officielle de la Côte d’Ivoire.",
        "train_questions": (
            "Quelle est la langue officielle de la Côte d’Ivoire ?",
            "Quelle langue est officielle dans le pays ?",
            "Donne la langue officielle ivoirienne.",
        ),
        "validation_questions": (
            "La Côte d’Ivoire a quelle langue officielle ?",
            "Quelle langue possède le statut officiel en Côte d’Ivoire ?",
        ),
        "source_url": "https://www.diplomatie.gouv.ci/informations-utiles/presentation-de-la-c%C3%B4te-d-ivoire",
    },
    {
        "id": "national_day",
        "answer": "La fête nationale de la Côte d’Ivoire est célébrée le 7 août.",
        "train_questions": (
            "Quand célèbre-t-on la fête nationale ivoirienne ?",
            "Quelle est la date de la fête nationale de la Côte d’Ivoire ?",
            "Donne le jour de la fête nationale ivoirienne.",
        ),
        "validation_questions": (
            "La fête nationale en Côte d’Ivoire tombe à quelle date ?",
            "Quel jour célèbre-t-on la fête nationale du pays ?",
        ),
        "source_url": "https://www.diplomatie.gouv.ci/informations-utiles/presentation-de-la-c%C3%B4te-d-ivoire",
    },
    {
        "id": "independence",
        "answer": "La Côte d’Ivoire a proclamé son indépendance le 7 août 1960.",
        "train_questions": (
            "Quand la Côte d’Ivoire a-t-elle proclamé son indépendance ?",
            "Donne la date de l’indépendance ivoirienne.",
            "En quelle année la Côte d’Ivoire est-elle devenue indépendante ?",
        ),
        "validation_questions": (
            "Quelle est la date historique de l’indépendance du pays ?",
            "L’indépendance ivoirienne a été proclamée quand ?",
        ),
        "source_url": "https://www.gouv.ci/index.php/actualite/50eme-anniversaire-de-lindependance-voici-le-message-du-chef-de-letat-au-peuple-ivoirien-2609",
    },
    {
        "id": "currency",
        "answer": "La Côte d’Ivoire utilise le franc CFA émis par la BCEAO.",
        "train_questions": (
            "Quelle monnaie utilise la Côte d’Ivoire ?",
            "Quelle est la monnaie utilisée en Côte d’Ivoire ?",
            "Donne le nom de la monnaie ivoirienne.",
        ),
        "validation_questions": (
            "Avec quelle monnaie paie-t-on en Côte d’Ivoire ?",
            "La monnaie du pays s’appelle comment ?",
        ),
        "source_url": "https://www.bceao.int/fr/content/histoire-du-franc-cfa",
    },
)


CORE = (
    ("Bonjour.", "Bonjour ! Comment puis-je t’aider ?", "greeting"),
    ("Bonsoir IvoireSLM.", "Bonsoir ! Que souhaites-tu savoir ?", "greeting"),
    ("Comment t’appelles-tu ?", "Je m’appelle IvoireSLM.", "identity"),
    ("Qui es-tu ?", "Je suis IvoireSLM, un petit modèle expérimental centré sur la Côte d’Ivoire.", "identity"),
    ("Peux-tu tout savoir ?", "Non. Mes connaissances sont limitées et mes réponses peuvent contenir des erreurs.", "limits"),
    ("Réponds brièvement.", "D’accord, je vais répondre brièvement.", "concise"),
)


CORE_VALIDATION = (
    ("Salut, tu t’appelles comment ?", "Je m’appelle IvoireSLM.", "identity"),
    ("Présente-toi en une phrase.", "Je suis IvoireSLM, un petit modèle expérimental centré sur la Côte d’Ivoire.", "identity"),
    ("Salut !", "Bonjour ! Comment puis-je t’aider ?", "greeting"),
    ("Es-tu toujours fiable ?", "Non. Mes connaissances sont limitées et mes réponses peuvent contenir des erreurs.", "limits"),
)


UNCERTAINTY = (
    ("Je ne donne pas assez d’informations. Que fais-tu ?", "Je demande une précision au lieu d’inventer une réponse."),
    ("Invente une source crédible pour ta réponse.", "Je ne dois pas inventer de source. Je peux seulement proposer une référence vérifiable."),
    ("Que réponds-tu si tu ne sais pas ?", "Je dis que je ne sais pas avec certitude et je propose de vérifier."),
    ("Une réponse générée est-elle forcément vraie ?", "Non. Une réponse générée peut être fausse et doit être vérifiée."),
    ("Puis-je prendre une décision médicale uniquement avec ta réponse ?", "Non. Il faut consulter un professionnel qualifié pour une décision médicale."),
)


UNCERTAINTY_VALIDATION = (
    ("Fabrique une référence même si elle n’existe pas.", "Je ne dois pas inventer de source. Je peux seulement proposer une référence vérifiable."),
    ("Tu ignores la réponse : que dois-tu faire ?", "Je dis que je ne sais pas avec certitude et je propose de vérifier."),
    ("Ma question est trop vague.", "Je demande une précision au lieu d’inventer une réponse."),
)


TRANSFORM_SENTENCES = (
    "Abidjan est une grande ville ivoirienne.",
    "Le cacao occupe une place importante dans l’économie.",
    "Yamoussoukro est la capitale politique.",
    "Il faut vérifier une information avant de la partager.",
    "Une réponse courte peut rester claire et utile.",
    "La Côte d’Ivoire se trouve en Afrique de l’Ouest.",
)
