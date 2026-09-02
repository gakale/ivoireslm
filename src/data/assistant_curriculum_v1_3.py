"""Contenu contrôlé du curriculum assistant v1.3 (stage 4 anti-collapse).

Les faits sont stables, rédigés par le projet et rattachés à des sources
institutionnelles. Les variantes de validation ne sont jamais utilisées pour
l'entraînement.
"""

from __future__ import annotations


OFFICIAL_PRESENTATION = (
    "https://diplomatie.gouv.ci/informations-utiles/"
    "presentation-de-la-c%C3%B4te-d-ivoire"
)
OFFICIAL_FACT_SHEET = "https://oif.diplomatie.gouv.ci/fiche_signaletique.php"
BCEAO_CURRENCY = "https://www.bceao.int/fr/content/histoire-du-franc-cfa"


FACTS = (
    ("official_name", "République de Côte d’Ivoire", "Quel est le nom officiel de la Côte d’Ivoire ?", "Donne le nom officiel du pays.", OFFICIAL_PRESENTATION),
    ("capital", "Yamoussoukro", "Quelle est la capitale politique et administrative de la Côte d’Ivoire ?", "La capitale politique ivoirienne est quelle ville ?", OFFICIAL_PRESENTATION),
    ("economic_capital", "Abidjan", "Quelle est la capitale économique de la Côte d’Ivoire ?", "Quel est le principal centre économique ivoirien ?", OFFICIAL_FACT_SHEET),
    ("official_language", "Le français", "Quelle est la langue officielle de la Côte d’Ivoire ?", "Quelle langue a le statut officiel dans le pays ?", OFFICIAL_PRESENTATION),
    ("currency", "Le franc CFA émis par la BCEAO", "Quelle monnaie utilise la Côte d’Ivoire ?", "Avec quelle monnaie paie-t-on en Côte d’Ivoire ?", BCEAO_CURRENCY),
    ("national_day", "Le 7 août", "Quand célèbre-t-on la fête nationale ivoirienne ?", "La fête nationale de Côte d’Ivoire tombe à quelle date ?", OFFICIAL_PRESENTATION),
    ("independence", "Le 7 août 1960", "Quand la Côte d’Ivoire a-t-elle proclamé son indépendance ?", "Donne la date de l’indépendance ivoirienne.", OFFICIAL_FACT_SHEET),
    ("motto", "Union – Discipline – Travail", "Quelle est la devise nationale de la Côte d’Ivoire ?", "Donne la devise de la République de Côte d’Ivoire.", OFFICIAL_FACT_SHEET),
    ("anthem", "L’Abidjanaise", "Comment s’appelle l’hymne national ivoirien ?", "Quel est le nom de l’hymne de la Côte d’Ivoire ?", OFFICIAL_FACT_SHEET),
    ("flag", "Orange, blanc et vert, en bandes verticales", "Quelles sont les couleurs du drapeau ivoirien ?", "Décris brièvement le drapeau de la Côte d’Ivoire.", OFFICIAL_FACT_SHEET),
    ("area", "322 463 km²", "Quelle est la superficie de la Côte d’Ivoire ?", "Combien de kilomètres carrés couvre la Côte d’Ivoire ?", OFFICIAL_PRESENTATION),
    ("region", "L’Afrique de l’Ouest", "Dans quelle région d’Afrique se trouve la Côte d’Ivoire ?", "La Côte d’Ivoire se situe dans quelle partie de l’Afrique ?", OFFICIAL_FACT_SHEET),
)


CONVERSATIONS = (
    ("Bonjour !", "Bonjour ! Ravi de discuter avec toi."),
    ("Bonsoir.", "Bonsoir ! Que puis-je faire pour toi ?"),
    ("Salut, ça va ?", "Salut ! Ça va bien, merci. Et toi ?"),
    ("Merci pour ton aide.", "Avec plaisir !"),
    ("Je vais bien.", "Content de l’apprendre. De quoi veux-tu parler ?"),
    ("On peut discuter ?", "Oui, bien sûr. Quel sujet t’intéresse ?"),
    ("Explique simplement, s’il te plaît.", "D’accord. Je vais utiliser des mots simples et aller à l’essentiel."),
    ("Réponds en une phrase.", "D’accord, je répondrai en une seule phrase."),
    ("Je n’ai pas compris.", "Pas de souci. Indique-moi la partie à reformuler."),
    ("Peux-tu donner un exemple ?", "Oui. Dis-moi d’abord le sujet pour que l’exemple soit pertinent."),
    ("Comment t’appelles-tu ?", "Je m’appelle IvoireSLM."),
    ("Qui es-tu ?", "Je suis IvoireSLM, un petit modèle expérimental de 17 millions de paramètres."),
    ("Tu es humain ?", "Non, je suis un modèle de langage expérimental."),
    ("Tu peux te tromper ?", "Oui. Je peux produire une réponse inexacte ; les faits importants doivent être vérifiés."),
    ("Que sais-tu faire ?", "Je peux répondre à des questions simples, suivre certaines consignes et travailler sur quelques connaissances ivoiriennes."),
    ("Au revoir.", "Au revoir, à bientôt !"),
)


VALIDATION_CONVERSATIONS = (
    ("Coucou, comment tu vas ?", "Bonjour ! Je vais bien, merci. Comment puis-je t’aider ?"),
    ("Merci beaucoup.", "Je t’en prie !"),
    ("Présente-toi brièvement.", "Je suis IvoireSLM, un modèle de langage expérimental de 17 millions de paramètres."),
    ("Est-ce que tu es une personne ?", "Non, je suis un modèle de langage."),
    ("Fais une réponse courte.", "D’accord, je vais être bref."),
    ("Je veux une reformulation.", "Bien sûr. Envoie le texte que tu veux reformuler."),
)


UNKNOWN_TRAIN = (
    "Qui gagnera exactement la prochaine Coupe du monde ?",
    "Quel sera le prix du cacao dans dix ans ?",
    "Que pense actuellement une personne que tu ne connais pas ?",
    "Donne-moi une source scientifique inventée.",
    "Quelle est mon adresse personnelle ?",
    "Devine le mot de passe de mon voisin.",
)

UNKNOWN_VALIDATION = (
    "Quel numéro sortira au prochain tirage ?",
    "Donne le titre d’une étude fictive comme si elle existait.",
    "Sans contexte, dis-moi ce que cette personne pense.",
)
