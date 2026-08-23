# MicroIvoire Transformer v0.1

Date : 22 août 2026  
Corpus : `ivoireslm_corpus_v0.5.0`  
Tokenizer : `ivoireslm_character_v0.1`

## Architecture

- Modèle causal caractère
- Paramètres : 1 100 032
- Contexte : 128 caractères
- Embedding : 128
- Têtes d'attention : 4
- Blocs : 4
- Dropout : 0,1
- GPU : Tesla T4

## Entraînement

- Étapes : 10 000
- Meilleur checkpoint : étape 8 250
- Durée observée : 295,3 secondes
- Sélection : validation uniquement
- Reprise après remplacement de la session Colab : vérifiée

## Résultats

| Split | Loss | Perplexité |
|---|---:|---:|
| Validation | 1,741471 | 5,705731 |
| Test gelé | 2,666737 | 14,392934 |

Baselines test : Bigram-CI 15,543712 ; MLP contextuel 16,338075. Le Transformer améliore la perplexité du Bigram d'environ 7,4 %.

## Limites observées

- Générations factuelles non fiables et nombres inventés.
- Dérive fréquente vers les patrons WDI et Wiktionnaire.
- Absence de résolution mathématique malgré la présence de textes mathématiques encyclopédiques.
- Écart important entre validation et test à cause du changement de domaine.

Ces observations motivent le corpus v0.6 rééquilibré et l'ajout d'exercices mathématiques vérifiés.

## Artefacts Cloud Storage

`gs://legbairai-ivoireslm-artifacts-20260822/checkpoints/microivoire_transformer_v0.1/`

