# Livraison du corpus IvoireSLM v0.6.0

Date : 23 août 2026  
Parent : `ivoireslm_corpus_v0.5.0`

## Résultat

- Documents : 25
- Caractères : 14 316 283
- Mots : 2 217 561
- Lignes : 83 983
- Mathématiques : 1 623 209 caractères, soit 11,3382 %
- Français naturel ivoirien ajouté : 585 064 caractères, soit 4,0867 %
- Quality gate : réussi
- Empreintes SHA-256 : 36/36 vérifiées

## Changements principaux

- Réduction déterministe des six patrons WDI de 40 477 à 3 808 lignes.
- Plafonnement du Wiktionnaire à environ 4 millions de caractères.
- Plafonnement des documentations et sources factuelles très répétitives.
- Ajout de 6 400 exercices mathématiques structurés en problème, méthode, solution et réponse.
- Vérification programmatique de 100 % des réponses mathématiques.
- Ajout de 179 articles francophones liés à la Côte d’Ivoire avec révisions et attributions Wikimedia.
- Déplacement des anciens tests déjà consultés vers `train`.
- Création d’un nouveau test gelé par groupes de sources et identifiants de pages/exercices.

## Splits

| Split | Documents | Caractères | Mots |
|---|---:|---:|---:|
| Train | 17 | 14 060 827 | 2 177 978 |
| Validation | 4 | 183 073 | 27 715 |
| Test gelé | 4 | 72 383 | 11 868 |

Le nouveau test combine deux sources ivoiriennes factuelles inédites, des exercices mathématiques distincts et des pages naturelles distinctes. Aucun `group_id` n'est partagé entre les splits.

## Contrôles

- Fuite de groupe : 0
- Donnée personnelle détectée : 0
- Caractère de contrôle : 0
- Caractère inconnu après tokenisation : 0 % sur train, validation et test
- Candidate rejetée conservée : `ivoireslm_corpus_v0.6.0_rejected_math16p56`

## Artefacts VM

- Corpus : `/home/gnakalehacker/ivoireslm-storage/corpora/ivoireslm_corpus_v0.6.0`
- Mathématiques vérifiées : `/home/gnakalehacker/ivoireslm-storage/derived/math_verified_v0.1`
- Snapshot naturel : `/home/gnakalehacker/ivoireslm-storage/snapshots/wikipedia_ci_fr_v0.1`
- Tokenizer : `/home/gnakalehacker/ivoireslm-storage/tokenizers/character_v0.2`

## Cloud Storage

`gs://legbairai-ivoireslm-artifacts-20260822/datasets/character_v0.2/ivoireslm_character_v0.2.tar.gz`

