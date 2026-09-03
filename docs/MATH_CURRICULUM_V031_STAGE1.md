# IvoireSLM 17M — Curriculum mathématique v0.3.1, stage 1

## Pourquoi cette branche existe

L'expérience SFT v0.3 a réduit la loss supervisée, mais son exactitude
mathématique est restée presque nulle. Le diagnostic par sous-famille a montré
que le modèle apprenait la forme des réponses sans apprendre fiablement les
résultats numériques. Cette branche recommence donc depuis le meilleur
checkpoint pré-SFT v0.3 et isole deux compétences simples.

## Données du stage 1

- Additions avec deux termes compris entre 0 et 20.
- Tables de multiplication avec deux facteurs compris entre 2 et 12.
- Réponse canonique courte : uniquement le nombre attendu.
- Couples commutatifs gardés dans le même split : `3×7` et `7×3` ne peuvent
  pas se retrouver de part et d'autre de train et validation.
- Aucun split test créé.
- Toutes les réponses sont recalculées et vérifiées par programme.

Le dataset déterministe contient 448 exemples train et 114 exemples validation.

## Entraînement pilote

- Modèle parent : `best.pt` de SFT v0.3, étape 0.
- Nouvel optimiseur, aucune reprise de l'optimiseur v0.3.
- 35 % des micro-lots proviennent du corpus de langue générale.
- Parmi les micro-lots supervisés : 70 % addition, 30 % multiplication.
- Learning rate maximal : `1e-5`.
- Premier palier : 250 étapes.
- Sélection de `best.pt` : taux exact pondéré, uniquement si la garde de
  langue générale reste respectée.

## Seuils décidés avant l'expérience

- Addition exacte : au moins 25 %.
- Multiplication exacte : au moins 15 %.
- Hausse maximale de la loss de langue générale : 2 %.

Le passage au stage 2 n'est autorisé que si les trois conditions sont remplies.
Une condition manquante conduit à poursuivre prudemment le stage 1 ou à revoir
la recette. Aucun test final ne doit être ouvert.

## Fichiers

- `scripts/data/build_math_curriculum_v031.py`
- `scripts/training/finetune_math_curriculum_v031_17m.py`
- `tests/test_math_curriculum_v031.py`
