# IvoireSLM 17M — assistant outillé v1

Cette interface sépare explicitement les capacités contenues dans les poids des
capacités fournies à l'inférence. Le mode **Modèle seul** charge le CPT v1.1.1
étape 500 et n'utilise aucun outil. Le mode **Assistant outillé** choisit une
route avant de répondre.

Les routes locales couvrent le calcul déterministe, l'identité d'IvoireSLM, les
salutations courantes, un petit ensemble de faits institutionnels ivoiriens et
trois connaissances dioula validées. Les réponses locales indiquent leur source
ou leur caractère contrôlé. Les autres demandes vont au modèle et sont marquées
comme non vérifiées.

Lorsque l'utilisateur active Internet, une question factuelle inconnue utilise
l'API Action officielle de Wikipédia en français. L'interface restitue un court
extrait et les URL consultées sans demander au 17M de réinventer ou fusionner les
faits. Une source Internet reste signalée comme à vérifier.

Les retours humains sont écrits séparément dans Drive avec consentement et
attestation d'absence de données personnelles. Ils ne déclenchent jamais un
entraînement automatique.

Cette version est une démonstration expérimentale, pas un assistant général ni
une source adaptée aux décisions médicales, juridiques ou financières.
