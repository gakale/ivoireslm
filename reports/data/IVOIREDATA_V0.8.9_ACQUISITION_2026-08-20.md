# Acquisition IvoireData v0.8.9 — 20 août 2026

## État du corpus de référence

Le corpus IvoireSLM v0.5.0 contient 17 192 011 caractères associés à la Côte d'Ivoire sur 36 151 493, soit 47,5555 %, répartis dans 15 documents. Cette quantité permet de commencer les prototypes, mais la partie ivoirienne reste concentrée dans `multidomain` (41,37 %), `economy` (34,69 %) et `agriculture` (23,16 %).

## Collecte effectuée

IvoireData v0.8.9 a été installé dans `/home/gnakalehacker/IvoireData`. Son stockage est séparé du dépôt :

`/home/gnakalehacker/ivoireslm-storage/ivoiredata`

Neuf sources ouvertes ont terminé leur synchronisation : circonscriptions administratives, commerce extérieur, sites touristiques, production forestière, produits pétroliers et gaziers, pétroles bruts SIR 2017, télévisions autorisées, geoBoundaries et projets Banque mondiale.

Après exclusion des tables techniques répétées du catalogue, huit sources livrent 1 260 lignes métier candidates. La source des télévisions est signalée comme faux positif de livraison : son chargement ne contient aucune table métier et ne doit pas entrer dans le corpus.

Le passage général du catalogue a été interrompu proprement après sauvegarde de 59 artefacts, car le portail était trop lent. Son état incrémental permet une reprise ultérieure.

## Décision

Ces données sont conservées comme file d'intégration v0.6.0. Elles ne sont pas ajoutées automatiquement au corpus : chaque table métier doit recevoir un rendu factuel, un hash source, un contrôle de droits, une déduplication contre v0.5 et un audit PII.

Le volume ivoirien actuel est suffisant pour passer au tokenizer et au modèle pédagogique. L'acquisition continue désormais en parallèle, avec priorité à l'éducation, la santé, l'environnement, la culture, le nouchi et les langues ivoiriennes sous droits explicites.
