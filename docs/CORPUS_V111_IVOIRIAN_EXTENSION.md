# Extension ivoirienne du corpus v1.1.1

La collecte française v1.1.0 a produit plus de 71 millions de caractères, mais
l'audit initial confondait deux objectifs : apprendre des connaissances
ivoiriennes pendant la continuation de préentraînement et apprendre un style de
conversation ivoirien pendant la supervision.

La version v1.1.1 sépare donc les décisions :

- le pilote CPT exige au moins 5 millions de caractères ivoiriens factuels,
  attribués et réutilisables ;
- une nouvelle SFT exige en plus 250 000 caractères de conversations ivoiriennes
  explicitement consenties et nettoyées ;
- aucun résultat de test et aucune conversation privée ne sont admis.

## Source publique ivoirienne

`snapshot_datagouvci_open_v011.py` utilise l'API officielle de data.gouv.ci. Il
conserve uniquement les jeux portant une Licence Ouverte, l'attribution, la date
de mise à jour et l'URL. Les schémas contenant des champs typiques de données
personnelles sont rejetés. La séparation train/validation se fait au niveau du
jeu de données et aucun split test n'est créé.

La collecte est reprenable grâce à un cache par jeu de données.

## Langues ivoiriennes

Koumankan4Dyula reste une source séparée sous CC BY-SA 4.0. Son accès est soumis
à l'acceptation des conditions sur Hugging Face. Les données sous partage à
l'identique restent identifiables afin de permettre une publication conforme.

## Exécution Colab

Après extraction du paquet de code, exécuter :

```bash
python -u scripts/colab/run_corpus_v111_ivoirian_collection.py
```

Le lanceur collecte, reconstruit et audite. Il ne lance jamais l'entraînement et
ne crée pas de test.
