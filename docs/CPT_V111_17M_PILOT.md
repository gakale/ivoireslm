# Pilote CPT 17M v1.1.1

Date : 2026-09-03

Le corpus `ivoireslm_corpus_supplement_v1.1.1` a franchi toutes les gardes de
continuation de préentraînement. Ce pilote repart du checkpoint CPT v0.4
original, conserve exactement le BPE v0.4 et n'ouvre aucun test.

## Mélange du pilote

| Domaine | Probabilité |
|---|---:|
| Corpus v0.9 original | 70 % |
| Français naturel ouvert | 20 % |
| Conversation française ouverte | 5 % |
| Données ivoiriennes officielles ancrées | 5 % |

Le mélange garde une majorité de données historiques pour limiter l'oubli. Le
pilote est limité à 125 étapes sur un maximum expérimental de 500, avec un taux
d'apprentissage maximal de `1e-5`.

## Gardes

- l'audit v1.1.1 doit autoriser explicitement le CPT ;
- tokenizer et corpus v0.4 doivent être présents et intacts ;
- le supplément ne doit contenir aucun split test ;
- la loss de validation du corpus historique ne peut augmenter de plus de
  `0.015` nat ;
- les checkpoints sont écrits dans une nouvelle branche Drive ;
- aucune supervision conversationnelle n'est exécutée.

Après le palier 125, le candidat doit être évalué sur la loss par domaine, la
diversité, la répétition, les faits ivoiriens et un ensemble de questions de
développement. Le test final reste fermé jusqu'à la sélection finale.
