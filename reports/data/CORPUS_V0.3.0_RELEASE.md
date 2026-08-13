# IvoireSLM corpus v0.3.0

Date de construction : 13 août 2026.

## Résultat

- 17 192 011 caractères, soit 17 192 011 tokens avec le tokenizer caractère prévu par la roadmap.
- 2 781 979 mots approximatifs.
- 91 870 phrases et 104 179 faits atomiques déclarés.
- 15 documents : 11 train, 2 validation, 2 test.
- 6 domaines : agriculture, démographie, économie, éducation, environnement/climat et multidomaine.
- 15 documents sur 15 classés `A_REDISTRIBUTABLE`.

Le seuil de 17 millions de caractères-tokens est atteint à 101,13 %. Cette mesure ne doit pas être confondue avec le nombre de tokens d'un futur tokenizer BPE ou SentencePiece.

## Nouveaux apports depuis v0.1.0

### World Development Indicators

- 40 477 observations numériques ivoiriennes.
- 1 423 indicateurs observés sur 1960–2025.
- 7 112 880 caractères générés de manière déterministe.
- Licence : CC BY 4.0.

### FAOSTAT — Production végétale et animale

- 17 541 lignes source, dont 16 911 observations numériques retenues.
- 121 produits, 8 éléments statistiques et 64 années sur 1961–2024.
- Aucune clé produit/élément/année/unité dupliquée.
- 3 816 584 caractères générés de manière déterministe.
- Licence : CC BY 4.0, complétée par les conditions d'utilisation des bases statistiques FAO.

Les libellés anglais officiels, codes FAOSTAT/CPC, unités et drapeaux de qualité sont conservés. Aucune traduction de mesure ou d'unité n'est inventée.

## Contrôle qualité

Le quality gate et l'audit indépendant passent :

- aucun hash source invalide ;
- aucun document ou ligne normalisée dupliqué ;
- aucune paire de documents quasi dupliquée au seuil 0,80 ;
- aucune fuite de groupe entre train, validation et test ;
- aucune adresse électronique ou numéro de téléphone détecté ;
- aucun marqueur technique suspect ni caractère de contrôle.

## Artefacts sur la VM

Répertoire :

`/home/gnakalehacker/ivoireslm-storage/corpora/ivoireslm_corpus_v0.3.0`

Archive :

`/home/gnakalehacker/ivoireslm-storage/corpora/ivoireslm_corpus_v0.3.0.tar.gz`

SHA-256 de l'archive :

`ee0c471573738b1f47358d50bb636371aac2c2f08013be5b43d93d5f6374d5c8`

## Reproduction

```bash
cd ~/ivoireslm
source ~/venv/bin/activate
python3 scripts/data/build_faostat_production_factual_v01.py
python3 scripts/data/build_corpus_v03.py
python3 scripts/data/audit_corpus_v03.py
```

Le snapshot source immuable est stocké sous :

`/home/gnakalehacker/ivoireslm-storage/snapshots/ivoiredata_2026-08-13_growth_v0.2`

## Limite principale

La cible quantitative est atteinte, mais le corpus demeure principalement factuel et structuré. La prochaine priorité n'est plus d'ajouter du volume artificiel : elle est d'augmenter la diversité avec du texte naturel ivoirien explicitement autorisé, notamment administration, santé, éducation, médias, littérature ouverte et langues locales.
