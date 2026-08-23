# MicroIvoire Transformer v0.2 5M

Date : 23 août 2026  
Corpus : `ivoireslm_corpus_v0.6.0`  
Tokenizer : `ivoireslm_character_v0.2`

## Architecture

- Modèle causal caractère de type decoder-only
- Paramètres : 4 758 144
- Contexte : 256 caractères
- Embedding : 192
- Têtes d'attention : 6
- Blocs : 10
- Dropout : 0,15
- GPU : Tesla T4

## Entraînement

- Étapes : 10 000
- Meilleur checkpoint : étape 9 250
- Durée : 2 500,79 secondes, soit environ 41 min 41 s
- Sélection du checkpoint : validation uniquement
- Test gelé ouvert une seule fois après la sélection
- Checkpoints et résultats sauvegardés dans Cloud Storage avant la déconnexion Colab

## Résultats

| Split | Tokens évalués | Loss | Perplexité |
|---|---:|---:|---:|
| Validation | 183 040 | 1,218356 | 3,381623 |
| Test gelé | 72 192 | 1,124651 | 3,079141 |

Ces résultats ne doivent pas être comparés directement aux perplexités du modèle v0.1, car le corpus, le tokenizer et surtout les splits ont changé. Une comparaison valide exige de réentraîner ou au minimum de réévaluer les baselines sur les splits v0.2.

## Analyse qualitative

- Le texte factuel est nettement plus structuré et lisible.
- Le modèle reproduit correctement la forme des exercices mathématiques.
- Les solutions numériques restent souvent fausses : il imite une méthode sans exécuter un calcul fiable.
- Le français conversationnel reste instable et contient des mots inventés.
- Les générations factuelles peuvent inventer des nombres et ne doivent pas être présentées comme des faits vérifiés.

## Prochaine validation requise

1. Créer un benchmark mathématique déterministe et mesurer l'exactitude de la réponse finale.
2. Réentraîner les baselines Bigram et MLP sur v0.2 pour une comparaison équitable.
3. Constituer un JSONL de corrections humaines avec prompt, réponse du modèle, réponse corrigée et statut.
4. Évaluer séparément mathématiques, français ivoirien naturel et faits ivoiriens.

## Artefacts Cloud Storage

`gs://legbairai-ivoireslm-artifacts-20260822/checkpoints/microivoire_transformer_v0.2_5m/`

- `best.pt` : 19 061 395 octets
- `latest.pt` : 57 197 437 octets
- `report.json` : 1 199 octets
- `sample.txt` : 2 673 octets
- SHA256 du meilleur checkpoint : `ca1dfc4639f2a9887a7dadaaddeac3ffbd28da85f279039cb5093a80b56374f4`
