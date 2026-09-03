# Décision scientifique — diversification v1.0, étape 1000

Date : 2026-09-02
Modèle : `microivoire_transformer_v1.0_17m_cpt_pilot`
Paramètres : 17 129 280
Parent : meilleur checkpoint dense v0.4 à l'étape 20 750

## Objectif

Vérifier qu'une continuation prudente sur un mélange diversifié améliore la couverture du modèle sans dégrader fortement le français général, avant d'envisager un entraînement dense de 100 millions de paramètres.

## Mélange réellement échantillonné

- corpus général v0.9 : 60 % ;
- Wikipédia français : 15 % ;
- Wikipédia anglais : 5 % ;
- raisonnement mathématique : 10 % ;
- code et agents : 5 % ;
- cybersécurité défensive : 5 %.

Le test est resté scellé pendant toute l'expérience.

## Résultats de validation

| Domaine | Loss avant | Loss étape 1000 | Évolution |
|---|---:|---:|---:|
| Score pondéré | 3,1619 | 2,8577 | -0,3042 |
| Corpus général v0.9 | 2,8881 | 2,8951 | +0,0070 |
| Wikipédia français | 3,0599 | 2,9059 | -0,1540 |
| Wikipédia anglais | 4,0119 | 3,6850 | -0,3268 |
| Raisonnement mathématique | 3,8734 | 2,6384 | -1,2351 |
| Code et agents | 4,1901 | 3,3140 | -0,8761 |
| Cybersécurité défensive | 3,4523 | 1,4177 | -2,0345 |

La garde du corpus général autorisait une hausse maximale de 0,03. La hausse observée est de 0,0070 : la garde est validée.

## Contrôle qualitatif déterministe

Sept amorces identiques ont été générées avec le parent et le pilote, en décodage glouton (`top_k=1`, 96 nouveaux tokens). Le pilote reste fortement répétitif et produit encore des affirmations fausses ou incohérentes. Il ne répond pas correctement à une question simple et ne génère pas une fonction Python correcte. L'amélioration de loss ne se traduit donc pas encore par une amélioration suffisante de l'usage réel.

Diagnostic : `QUALITATIVE_DIAGNOSTIC_STEP1000.json`
SHA256 : `032fe4cc986b20b1f22dff436a1d773842252015db1aaf8770d59c990dcca0a9`

## Décision

1. Figer le pilote à l'étape 1000 et conserver `best.pt`, `latest.pt`, `progress.json` et le diagnostic.
2. Ne pas lancer le modèle 100 M avec le corpus actuel.
3. Construire un corpus v1.1 beaucoup plus naturel, moins répétitif et mieux équilibré, avec provenance et licences documentées.
4. Ajouter des évaluations de génération et de suivi d'instructions aux gardes automatiques ; une baisse de perplexité ne suffit plus.
5. N'envisager le 100 M dense qu'après validation de ces gardes et une augmentation importante du volume de tokens uniques de qualité.
6. Reporter l'architecture MoE : elle sera étudiée après l'obtention d'un dense 100 M solide, car le routage d'experts ne corrige pas un corpus insuffisant.

## Publication

Le pipeline, les manifestes, les empreintes et les résultats négatifs peuvent être publiés comme parcours de recherche reproductible. Le checkpoint ne doit pas être présenté comme un assistant conversationnel fiable.
