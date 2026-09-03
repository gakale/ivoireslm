# Assistant 17M Stage 6 — bootstrap depuis le CPT 500

Le Stage 6 repart du meilleur checkpoint CPT v1.1.1 à l’étape 500. Ce
checkpoint améliore les validations française, conversationnelle et
ivoirienne sans dégrader le corpus v0.9, mais il n’est pas encore un assistant :
il répète les questions et ne termine pas avec `<EOS>`.

Le curriculum supervisé v1.5 couvre conversation, identité, connaissances
générales, faits ivoiriens, notions élémentaires de dioula, calcul exact,
suivi d’instructions, compréhension et incertitude calibrée. Les corrections
humaines ne sont acceptées qu’avec consentement, attestation d’absence de
données personnelles et contenu exploitable. Leur séparation train/contrôle
est déterministe au niveau de la question. Le contrôle humain, la validation
et l’entraînement restent disjoints ; aucun test final n’est créé.

Le pilote comporte 125 étapes à faible taux d’apprentissage, avec 60 % de
rejeu du corpus général. La sélection utilise des générations greedy et des
gardes contre l’écho, les réponses vides, la perte de diversité et la
dégradation de la langue générale. Une réussite du pilote autorise seulement
un contrôle humain : elle ne suffit pas à qualifier le modèle pour publication.
