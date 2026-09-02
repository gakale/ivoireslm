"""Contenu vérifié du stage 5 : réponses directes et anti-écho.

Les réponses du benchmark humain ne figurent pas dans ce module. Les faits
ivoiriens viennent de pages institutionnelles et les savoir-faire généraux
sont des formulations originales publiées sous CC0 par le projet.
"""

from __future__ import annotations


DIPLOMATIE_CI = (
    "https://diplomatie.gouv.ci/informations-utiles/"
    "presentation-de-la-c%C3%B4te-d-ivoire"
)
OIF_CI = "https://oif.diplomatie.gouv.ci/fiche_signaletique.php"
BCEAO_CURRENCY = "https://www.bceao.int/fr/content/histoire-du-franc-cfa"
BCEAO_HEADQUARTERS = (
    "https://www.bceao.int/sites/default/files/2021-10/"
    "Rapport%20annuel%20de%20la%20BCEAO%202020.pdf"
)
AFDB_ABOUT = "https://www.afdb.org/en/about-us"


# id, réponse complète, éléments obligatoires, questions train, questions val,
# source. Les variantes de validation ne sont jamais entraînées.
VERIFIED_FACTS = (
    (
        "official_name",
        "Le nom officiel du pays est la République de Côte d’Ivoire.",
        ("république de côte d’ivoire",),
        ("Quel est le nom officiel du pays ?", "Donne le nom officiel de la Côte d’Ivoire.", "Comment s’appelle officiellement la Côte d’Ivoire ?"),
        ("Quelle est l’appellation officielle de la Côte d’Ivoire ?", "Sous quel nom officiel le pays est-il désigné ?"),
        DIPLOMATIE_CI,
    ),
    (
        "capital",
        "La capitale politique et administrative de la Côte d’Ivoire est Yamoussoukro.",
        ("yamoussoukro",),
        ("Quelle ville est la capitale politique de la Côte d’Ivoire ?", "Donne la capitale administrative ivoirienne.", "Quelle est la capitale de la République de Côte d’Ivoire ?"),
        ("Où se trouve la capitale politique ivoirienne ?", "Quelle ville exerce le rôle de capitale administrative du pays ?"),
        DIPLOMATIE_CI,
    ),
    (
        "economic_capital",
        "Abidjan est la capitale économique et la principale ville de Côte d’Ivoire.",
        ("abidjan", "capitale économique"),
        ("Quelle est la capitale économique ivoirienne ?", "Quel est le principal centre économique de Côte d’Ivoire ?", "Quelle grande ville joue le rôle de capitale économique ?"),
        ("Où se situe le cœur économique du pays ?", "Quelle ville ivoirienne est considérée comme la capitale économique ?"),
        DIPLOMATIE_CI,
    ),
    (
        "official_language",
        "Le français est la langue officielle de la Côte d’Ivoire.",
        ("français",),
        ("Quelle langue est officielle en Côte d’Ivoire ?", "Donne la langue officielle du pays.", "Quelle langue a le statut officiel en Côte d’Ivoire ?"),
        ("Dans quelle langue officielle fonctionne l’administration ivoirienne ?", "Le pays reconnaît quelle langue comme langue officielle ?"),
        DIPLOMATIE_CI,
    ),
    (
        "currency",
        "La monnaie utilisée en Côte d’Ivoire est le franc CFA de l’Union économique et monétaire ouest-africaine, code XOF.",
        ("franc cfa", "xof"),
        ("Avec quelle monnaie paie-t-on en Côte d’Ivoire ?", "Donne la monnaie ivoirienne et son code.", "Quelle monnaie circule dans le pays ?"),
        ("Quel est le nom de la devise monétaire utilisée par les Ivoiriens ?", "Quelle unité monétaire utilise la Côte d’Ivoire ?"),
        BCEAO_CURRENCY,
    ),
    (
        "national_day",
        "La fête nationale de la Côte d’Ivoire est célébrée le 7 août.",
        ("7 août",),
        ("Quand célèbre-t-on la fête nationale ivoirienne ?", "À quelle date tombe la fête nationale de Côte d’Ivoire ?", "Donne la date de la fête nationale ivoirienne."),
        ("Quel jour du calendrier correspond à la fête nationale ivoirienne ?", "À quelle date les Ivoiriens célèbrent-ils leur fête nationale ?"),
        DIPLOMATIE_CI,
    ),
    (
        "independence",
        "La Côte d’Ivoire a proclamé son indépendance le 7 août 1960.",
        ("7 août 1960",),
        ("Quand la Côte d’Ivoire est-elle devenue indépendante ?", "Donne la date de l’indépendance ivoirienne.", "En quelle année la Côte d’Ivoire a-t-elle obtenu son indépendance ?"),
        ("À quelle date l’indépendance ivoirienne a-t-elle été proclamée ?", "L’indépendance de la Côte d’Ivoire remonte à quelle date ?"),
        OIF_CI,
    ),
    (
        "motto",
        "La devise nationale de la Côte d’Ivoire est « Union – Discipline – Travail ».",
        ("union", "discipline", "travail"),
        ("Donne la devise de la Côte d’Ivoire.", "Quelle est la devise nationale ivoirienne ?", "Quels trois mots forment la devise du pays ?"),
        ("Quelle formule sert de devise à la République de Côte d’Ivoire ?", "Rappelle les trois termes de la devise ivoirienne."),
        OIF_CI,
    ),
    (
        "anthem",
        "L’hymne national de la Côte d’Ivoire s’appelle L’Abidjanaise.",
        ("abidjanaise",),
        ("Comment s’appelle l’hymne national ivoirien ?", "Donne le nom de l’hymne de la Côte d’Ivoire.", "Quel chant est l’hymne national du pays ?"),
        ("Sous quel titre connaît-on l’hymne national ivoirien ?", "Quel est le titre du chant national de Côte d’Ivoire ?"),
        OIF_CI,
    ),
    (
        "flag",
        "Le drapeau ivoirien comporte trois bandes verticales orange, blanche et verte.",
        ("orange", "blanche", "verte"),
        ("Décris les couleurs du drapeau ivoirien.", "Quelles couleurs composent le drapeau de Côte d’Ivoire ?", "Comment sont disposées les couleurs du drapeau ivoirien ?"),
        ("À quoi ressemble le drapeau de la Côte d’Ivoire ?", "Cite dans l’ordre les couleurs du drapeau ivoirien."),
        OIF_CI,
    ),
    (
        "area",
        "La superficie de la Côte d’Ivoire est de 322 463 kilomètres carrés.",
        ("322 463",),
        ("Quelle est la superficie de la Côte d’Ivoire ?", "Combien de kilomètres carrés couvre le pays ?", "Donne l’étendue du territoire ivoirien en kilomètres carrés."),
        ("Quelle surface occupe le territoire de Côte d’Ivoire ?", "À combien de kilomètres carrés s’élève la superficie ivoirienne ?"),
        DIPLOMATIE_CI,
    ),
    (
        "west_africa",
        "La Côte d’Ivoire se situe en Afrique de l’Ouest.",
        ("afrique de l’ouest",),
        ("Dans quelle partie de l’Afrique se trouve la Côte d’Ivoire ?", "Situe la Côte d’Ivoire sur le continent africain.", "Dans quelle région d’Afrique se trouve le pays ?"),
        ("La Côte d’Ivoire appartient à quelle région africaine ?", "Où se place la Côte d’Ivoire en Afrique ?"),
        DIPLOMATIE_CI,
    ),
    (
        "main_cities",
        "Parmi les principales villes ivoiriennes figurent Abidjan, Bouaké, San Pedro, Korhogo, Daloa et Yamoussoukro.",
        ("abidjan", "bouaké", "yamoussoukro"),
        ("Cite plusieurs grandes villes de Côte d’Ivoire.", "Donne trois villes principales du pays.", "Quelles sont quelques villes importantes de Côte d’Ivoire ?"),
        ("Peux-tu nommer plusieurs centres urbains ivoiriens ?", "Cite au moins trois villes importantes de Côte d’Ivoire."),
        DIPLOMATIE_CI,
    ),
    (
        "cocoa",
        "La Côte d’Ivoire est le premier producteur mondial de cacao et représente plus de 40 % du marché mondial.",
        ("cacao",),
        ("Pour quel produit agricole la Côte d’Ivoire est-elle numéro un mondial ?", "Quel produit place le pays au premier rang mondial ?", "La Côte d’Ivoire domine la production mondiale de quel produit ?"),
        ("Quel produit agricole est emblématique du rang mondial de la Côte d’Ivoire ?", "Dans quelle production agricole le pays occupe-t-il la première place mondiale ?"),
        DIPLOMATIE_CI,
    ),
    (
        "uemoa_share",
        "La Côte d’Ivoire représente environ 40 % du produit intérieur brut de l’UEMOA.",
        ("40", "uemoa"),
        ("Quelle part du PIB de l’UEMOA vient de la Côte d’Ivoire ?", "Quel poids économique la Côte d’Ivoire a-t-elle dans l’UEMOA ?", "Donne la part ivoirienne dans le PIB de l’UEMOA."),
        ("Environ quel pourcentage du PIB de l’UEMOA représente le pays ?", "Quel est le poids de l’économie ivoirienne dans l’Union économique et monétaire ouest-africaine ?"),
        DIPLOMATIE_CI,
    ),
    (
        "economic_sectors",
        "En 2023, le secteur tertiaire représentait 56 % du PIB ivoirien, contre 22 % pour le primaire et 22 % pour le secondaire.",
        ("tertiaire", "56"),
        ("Quel secteur pesait le plus dans le PIB ivoirien en 2023 ?", "Quelle était la part du secteur tertiaire en 2023 ?", "Compare les trois grands secteurs du PIB ivoirien en 2023."),
        ("En 2023, quel secteur dominait l’économie ivoirienne ?", "Quelle branche représentait 56 % du PIB ivoirien en 2023 ?"),
        DIPLOMATIE_CI,
    ),
    (
        "afdb_headquarters",
        "Le siège du Groupe de la Banque africaine de développement se trouve à Abidjan.",
        ("banque africaine de développement", "abidjan"),
        ("Quelle grande banque de développement a son siège à Abidjan ?", "Où se trouve le siège de la Banque africaine de développement ?", "Quelle institution financière panafricaine est basée à Abidjan ?"),
        ("Quelle banque multilatérale africaine a établi son siège en Côte d’Ivoire ?", "Dans quelle ville siège le Groupe de la Banque africaine de développement ?"),
        AFDB_ABOUT,
    ),
    (
        "bceao_headquarters",
        "Le siège social de la BCEAO se trouve à Dakar, au Sénégal, et non à Abidjan.",
        ("dakar",),
        ("Dans quelle ville se trouve le siège de la BCEAO ?", "La BCEAO a-t-elle son siège à Abidjan ?", "Où est établi le siège social de la BCEAO ?"),
        ("Quelle capitale accueille le siège de la BCEAO ?", "Le siège central de la BCEAO se situe dans quelle ville ?"),
        BCEAO_HEADQUARTERS,
    ),
)


GENERAL_KNOWLEDGE = (
    ("google", "Google est une entreprise technologique connue notamment pour son moteur de recherche sur Internet.", ("google", "entreprise", "moteur de recherche"), ("C’est quoi Google ?", "Explique simplement ce qu’est Google.", "À quoi correspond le nom Google ?"), ("Présente Google en une phrase.", "Google désigne quoi ?")),
    ("philosophy", "La philosophie est une réflexion rationnelle et critique sur le monde, la connaissance et l’existence humaine.", ("réflexion", "critique"), ("Explique ce qu’est la philosophie.", "Parle brièvement de la philosophie.", "Donne une définition simple de la philosophie."), ("Comment définirais-tu la philosophie ?", "La philosophie étudie quoi ?")),
    ("mathematics", "Les mathématiques étudient les nombres, les formes, les structures et les relations à l’aide d’un raisonnement logique.", ("nombres", "raisonnement"), ("C’est quoi les mathématiques ?", "Explique simplement les maths.", "Que font les mathématiques ?"), ("Comment présenter les mathématiques en une phrase ?", "Quel est l’objet des mathématiques ?")),
    ("internet", "Internet est un réseau mondial qui relie des appareils et permet d’échanger des informations.", ("réseau", "mondial"), ("C’est quoi Internet ?", "Explique Internet simplement.", "À quoi sert Internet ?"), ("Comment définir Internet ?", "Internet désigne quoi ?")),
    ("verification", "Vérifier une information permet de réduire les erreurs, de comparer les sources et d’éviter de transmettre une affirmation fausse.", ("erreurs", "sources"), ("Pourquoi faut-il vérifier une information ?", "À quoi sert la vérification des informations ?", "Pourquoi comparer plusieurs sources ?"), ("Quel est l’intérêt de vérifier un fait ?", "Pourquoi ne faut-il pas partager une information sans contrôle ?")),
)


CONVERSATIONS_TRAIN = (
    ("Bonjour", "Bonjour ! Comment puis-je t’aider ?"),
    ("Bonsoir", "Bonsoir ! Que puis-je faire pour toi ?"),
    ("Salut", "Salut ! De quoi veux-tu parler ?"),
    ("Comment tu vas ?", "Je vais bien, merci. Et toi ?"),
    ("Je vais bien.", "Content de l’apprendre ! De quoi veux-tu discuter ?"),
    ("Merci.", "Avec plaisir !"),
    ("Merci beaucoup pour ton aide.", "Je t’en prie, avec plaisir."),
    ("On peut parler ?", "Oui, bien sûr. Quel sujet t’intéresse ?"),
    ("Je n’ai pas compris.", "Pas de souci. Dis-moi quelle partie je dois expliquer autrement."),
    ("Donne un exemple.", "D’accord. Indique-moi le sujet pour que je choisisse un exemple pertinent."),
    ("Réponds brièvement.", "D’accord, je vais répondre en peu de mots."),
    ("Explique avec des mots simples.", "D’accord, je vais utiliser un vocabulaire simple."),
    ("Comment t’appelles-tu ?", "Je m’appelle IvoireSLM."),
    ("Qui es-tu ?", "Je suis IvoireSLM, un petit modèle de langage expérimental."),
    ("Tu es une personne ?", "Non, je suis un modèle de langage, pas une personne."),
    ("Tu peux faire des erreurs ?", "Oui. Je peux me tromper, donc les informations importantes doivent être vérifiées."),
    ("Tu connais les maths ?", "Oui, je peux effectuer certains calculs simples. Donne-moi le calcul à résoudre."),
    ("Au revoir", "Au revoir et à bientôt !"),
)


CONVERSATIONS_VALIDATION = (
    ("Coucou !", "Bonjour ! Comment puis-je t’aider ?"),
    ("Ça va chez toi ?", "Je vais bien, merci. Et toi ?"),
    ("Je te remercie.", "Je t’en prie !"),
    ("Présente-toi en peu de mots.", "Je suis IvoireSLM, un petit modèle de langage expérimental."),
    ("Peux-tu calculer ?", "Oui, je peux résoudre certains calculs simples."),
    ("À plus tard.", "D’accord, à plus tard !"),
)
