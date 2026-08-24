# IvoireSLM — diagnostic du SFT mathématique et moteur déterministe v0.1

## Résultat du modèle 5M après SFT

Le SFT a appris à respecter le format demandé, mais pas à exécuter les calculs.

| Évaluation de développement | Format valide | Exactitude |
|---|---:|---:|
| Benchmark étendu, 1 000 exercices | 100 % | 0,2 % (2/1 000) |
| Diagnostic dans la distribution SFT, 1 000 exercices | 100 % | 0,3 % (3/1 000) |

Les trois réponses exactes du second diagnostic se trouvent uniquement aux
niveaux 1 et 2. Les niveaux 3 et 4 obtiennent 0 %. Un nouvel entraînement avec
la même architecture et les mêmes cibles n'est donc pas justifié.

## Décision d'architecture

IvoireSLM conserve deux responsabilités séparées :

1. le Transformer produit et comprend le français, le contexte ivoirien et la
   structure de la réponse ;
2. `deterministic_math_tool_v0.1` analyse l'énoncé et garantit le calcul exact.

Le moteur ne reçoit que le champ `problem`. Il n'accède ni à `expected`, ni à
`reference_answer`, ni aux données de vérification. Il refuse les énoncés
inconnus ou ambigus au lieu de deviner.

## Validation du moteur

| Benchmark de développement | Exactitude | Erreurs du solveur | Temps serveur |
|---|---:|---:|---:|
| `math_reasoning_v0.1` | 100 % (1 000/1 000) | 0 | 0,044 s |
| `math_sft_validation_diagnostic_v0.1` | 100 % (1 000/1 000) | 0 | 0,043 s |

Les dix familles sont couvertes : addition, multiplication, équation linéaire,
trinôme factorisable, fraction, pourcentage, rectangle, suite arithmétique,
monnaie rendue dans un marché ivoirien et partage d'une coopérative.

Le benchmark final `math_reasoning_final_v0.2` reste gelé et n'a pas été ouvert
pendant le développement. Il ne devra être exécuté qu'une seule fois après la
validation complète du système hybride.
