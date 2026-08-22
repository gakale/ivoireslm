# Contextual MLP-CI v0.1

Date : 22 août 2026  
Corpus : `ivoireslm_corpus_v0.5.0`  
Tokenizer : `ivoireslm_character_v0.1`

## Objectif

Mesurer l'apport d'une fenêtre de 16 caractères et d'embeddings appris par rapport à la baseline Bigram-CI. Le modèle est une étape pédagogique entre le Bigram et un modèle causal avec attention.

## Architecture et entraînement

- Modèle : MLP caractère à contexte fixe
- Paramètres : 249 526
- Contexte : 16 caractères
- Embedding : 32 dimensions
- Couche cachée : 128 unités, activation GELU
- Dropout : 0,1
- Optimiseur : AdamW, taux d'apprentissage 0,001
- Lot : 256 exemples
- Étapes : 10 000, soit 2 560 000 exemples vus
- Matériel : CPU
- Graine : 20260822
- Ajustement : split `train` uniquement
- Sélection du checkpoint : meilleure loss sur un échantillon de validation fixe
- Évaluation finale : tous les exemples séquentiels de `validation` et `test`

Le meilleur checkpoint a été sélectionné à l'étape 4 400. L'entraînement complet a duré environ 167 secondes.

## Résultats

| Split | Loss (nats) | Perplexité |
|---|---:|---:|
| Train, échantillon déterministe | 1,1997 | 3,3192 |
| Validation, exhaustive | 2,2587 | 9,5706 |
| Test, exhaustive | 2,7935 | 16,3381 |
| Test Bigram-CI, référence | 2,7437 | 15,5437 |

Le MLP améliore fortement la validation et produit des segments plus longs qui ressemblent au format du corpus. Il ne bat cependant pas la baseline Bigram sur le test gelé : sa perplexité test est supérieure de 5,11 %.

## Interprétation

Le résultat n'est pas déclaré supérieur au Bigram. L'écart train/validation/test montre une généralisation insuffisante et un changement de domaine :

- `validation` contient le commerce poisson/viande et les stations pluviométriques ;
- `test` contient les admissions au baccalauréat et le recensement RGPH 2021 ;
- `train` est dominé par des faits répétitifs, les données WDI, les mathématiques et le Wiktionnaire.

Le MLP concatène les 16 positions et mémorise facilement les patrons propres aux sources. Il ne partage pas efficacement les informations entre positions et ne dispose pas d'attention. Ce résultat motive l'étape suivante sans remettre en cause la qualité ou l'absence de fuite des splits.

## Artefacts sur la VM

- Checkpoint : `/home/gnakalehacker/ivoireslm-storage/models/contextual_mlp_ci_v0.1/contextual_mlp_ci_v0.1.pt`
- SHA-256 : `e574d62a6d46924161ba0a64a8c02e9e1e387d39a2a55b2f8439bce606fa9916`
- Rapport JSON : `/home/gnakalehacker/ivoireslm-storage/models/contextual_mlp_ci_v0.1/report.json`
- Échantillon : `/home/gnakalehacker/ivoireslm-storage/models/contextual_mlp_ci_v0.1/sample.txt`

## Décision

Conserver MLP-CI v0.1 comme baseline contextuelle expérimentale et passer à un petit modèle causal avec attention. Le critère minimal de la prochaine phase sera de battre la perplexité test Bigram de 15,5437 avec un protocole identique et reproductible.

