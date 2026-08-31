# Supervision équilibrée du modèle IvoireSLM 17M

## Objectif

La phase SFT v0.2 spécialise le modèle de base `microivoire_transformer_v0.4_17m`
sans remplacer son préentraînement. Elle lui apprend des formats de réponse utiles tout
en limitant la mémorisation et l'oubli du français général.

Le mélange supervisé est composé de :

- 2 % d'identité, prudence et gestion de l'incertitude ;
- 23 % de questions dont la réponse est explicitement présente dans un contexte WDI ;
- 30 % d'exercices mathématiques vérifiés automatiquement ;
- 22,5 % de traduction dioula vers français ;
- 22,5 % de traduction français vers dioula.

En plus, 15 % des micro-lots sont prélevés dans le corpus général v0.9.0. Ce rappel
réduit le risque d'oubli catastrophique.

## Principe de la loss ciblée

Chaque exemple contient un `prompt` et une `target`. Les tokens du prompt sont masqués
avec la valeur `-100` : ils donnent le contexte, mais ne participent pas directement à
la loss. Le gradient apprend seulement à prédire la réponse attendue.

Le meilleur checkpoint est choisi avec une loss de validation pondérée selon les cinq
familles ci-dessus. Une seconde mesure fixe sur le corpus général surveille la perte de
qualité linguistique.

## Séparation des données

- `train.jsonl` sert à modifier les poids ;
- `validation.jsonl` sert à choisir le meilleur checkpoint ;
- `test.jsonl` reste uniquement sur la VM jusqu'à la décision finale ;
- les benchmarks mathématiques indépendants restent également scellés.

Les phrases WDI sont séparées par identifiant d'indicateur, et les traductions par
composante parallèle. Un même groupe ne peut donc pas apparaître dans deux splits.

## Exécution par paliers

Le premier palier doit s'arrêter à 250 étapes. On compare alors :

1. le score supervisé global et par famille ;
2. la loss de langue générale ;
3. quelques générations fixes ;
4. l'intégrité SHA256 de `latest.pt`.

Si les résultats restent sains, la reprise exacte utilise `latest.pt` et avance aux
paliers 500, 1 000, 2 000 puis 4 000. Le test scellé n'est ouvert qu'après sélection du
checkpoint final.
