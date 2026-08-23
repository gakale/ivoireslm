# Évaluation mathématique du Transformer v0.2 5M

Date : 23 août 2026  
Benchmark : `math_reasoning_v0.1`  
Checkpoint : étape 9 250, SHA256 `ca1dfc4639f2a9887a7dadaaddeac3ffbd28da85f279039cb5093a80b56374f4`

## Résultats

- Exercices : 1 000
- Réponses portant une étiquette `Réponse :` exploitable : 400, soit 40 %
- Réponses exactement correctes : 0, soit 0 %
- Décodage : greedy déterministe
- Durée : 928,39 secondes

Le modèle reproduit souvent la structure des suites, équations linéaires et équations quadratiques, mais invente les opérations et les nombres. Un audit manuel des prédictions confirme que le score nul ne vient pas du parseur.

## Interprétation

La perplexité test de 3,0791 mesure une bonne prédiction locale des caractères, pas l'exécution correcte d'un algorithme. Le benchmark utilise volontairement des plages numériques plus larges que le préentraînement et révèle une absence de généralisation mathématique.

Ce benchmark devient un jeu de développement après consultation de ses résultats. Il ne servira pas de preuve finale après le fine-tuning.

## Artefacts

`gs://legbairai-ivoireslm-artifacts-20260822/evaluations/microivoire_transformer_v0.2_5m/math_reasoning_v0.1/`

- `predictions.jsonl`
- `predictions.partial.jsonl`
- `report.json`
