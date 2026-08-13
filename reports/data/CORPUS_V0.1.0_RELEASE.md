# Livraison — IvoireSLM Corpus v0.1.0

Date de construction : 13 août 2026  
Statut : validé pour les expériences pédagogiques et les premiers petits modèles  
Commit du pipeline : `9b19c76238a7cdc5c7048557a64af9629c77bcb7`

## Emplacement sur la VM

- Corpus : `/home/gnakalehacker/ivoireslm-storage/corpora/ivoireslm_corpus_v0.1.0/`
- Archive : `/home/gnakalehacker/ivoireslm-storage/corpora/ivoireslm_corpus_v0.1.0.tar.gz`
- SHA-256 de l'archive : `2ebf8b63428199aea6d0c7e266f1770bdcba659e2e2741eda3f0e011566050e2`
- Rapport qualité : `reports/quality_report.json` dans le corpus.
- Dataset card : `DATASET_CARD.md` dans le corpus.

## Contenu admis

- 13 documents et 13 groupes de sources.
- 34 482 phrases factuelles.
- 46 791 faits atomiques déclarés.
- 1 054 825 mots approximatifs.
- 6 262 547 caractères.
- Domaines : agriculture, démographie, économie, éducation, environnement et climat.
- 13 documents classés `A_REDISTRIBUTABLE` ou couverts par une licence ouverte dans leur manifeste source.

## Splits

| Split | Documents | Phrases | Faits atomiques |
|---|---:|---:|---:|
| Train | 9 | 32 946 | 43 221 |
| Validation | 2 | 791 | 791 |
| Test | 2 | 745 | 2 779 |

Les splits sont séparés par groupe de source. Aucun groupe ne traverse deux splits. Le test contient le RGPH 2021 et les résultats du BAC ; il est protégé et ne doit pas servir à entraîner le tokenizer ou à choisir les hyperparamètres.

## Quality gate

- Hashes sources : conformes.
- Doublons exacts de documents : 0.
- Doublons exacts de phrases après normalisation : 0.
- Paires de documents proches au seuil de 0,80 : 0.
- Similarité maximale observée : 0,030807, entre les deux sources de prix.
- Fuites de groupes entre les splits : 0.
- Courriels détectés : 0.
- Numéros de téléphone ivoiriens détectés : 0.
- Caractères de contrôle résiduels : 0.
- Marqueurs techniques `nan`, `None` ou caractère de remplacement : 0.
- Deux constructions successives ont produit le même hash agrégé : `cbe0b2084445b0b1fedabe1ff14c97220853f9d18d7508437c0164384e99bec1`.

## Données conservées hors corpus

Les documents sous copyright, les droits inconnus, les copies tierces, les transcriptions sans preuve de consentement, les anciens jeux d'instructions et le vieux pool contaminé restent dans :

`/home/gnakalehacker/ivoireslm-storage/imports/legacy_quarantine_2026-08-13/`

Les fichiers reçus depuis le Mac restent dans :

`/home/gnakalehacker/ivoireslm-storage/incoming/mac_2026-08-12/`

Aucun de ces éléments n'entre dans les splits officiels.

## Reproduction

```bash
cd ~/ivoireslm
source ~/venv/bin/activate
python3 scripts/data/build_corpus_v01.py
python3 scripts/data/audit_corpus_v01.py
```

## Limite principale

Le corpus est propre mais encore peu diversifié et dominé par les prix de marché. Il est adapté à la validation du pipeline, au tokenizer pédagogique et à de petits modèles expérimentaux. Il n'est pas encore suffisant pour un modèle ivoirien généraliste. La prochaine campagne de données doit cibler des textes naturels explicitement autorisés, notamment l'administration, l'éducation, la santé, les médias, la littérature autorisée, le nouchi et les langues ivoiriennes.
