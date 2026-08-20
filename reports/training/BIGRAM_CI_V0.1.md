# Bigram-CI v0.1

Date d'exécution : 20 août 2026.

## Résultat

Bigramme caractère entraîné à partir de zéro par comptage exact des transitions du split `train`, avec lissage additif `alpha=0,1`.

- Vocabulaire : 1 142 tokens.
- Paramètres : 1 304 164 transitions.
- Transitions d'entraînement : 35 895 977.
- Loss uniforme de référence : 7,0370 nats.
- Loss train : 2,4472 nats ; perplexité 11,56.
- Loss validation : 2,4735 nats ; perplexité 11,86.
- Loss test : 2,7437 nats ; perplexité 15,54.
- Ajustement effectué uniquement sur train.
- Génération et checkpoint reproductibles avec la graine `20260820`.

Le modèle apprend correctement les transitions locales de caractères, mais il ne possède ni contexte long ni compréhension. La sortie encore incohérente est normale et constitue la référence minimale pour mesurer les futurs modèles.

Artefacts :

`/home/gnakalehacker/ivoireslm-storage/models/bigram_ci_v0.1`

SHA-256 du checkpoint :

`a296df9576c0dde6bf25a0d3f7de3b2f4e9a6e0726669a5480dfcecb45d5b303`

## Reproduction

```bash
cd ~/ivoireslm
source ~/venv/bin/activate
python3 scripts/training/train_bigram_ci_v01.py
```

## Étape suivante

Construire un petit réseau contextuel puis MicroIvoire (1 à 5 millions de paramètres), en comparant systématiquement sa loss au Bigram-CI sur les mêmes splits gelés.
