# Collecte du corpus naturel v1.1

Cette phase corrige le déséquilibre mesuré dans le supplément v1.0 sans lancer
d'entraînement.

## Sources admises dans ce premier pilote

- Wikipédia français via l'API MediaWiki, texte en clair, révisions et URLs
  d'attribution conservées, licence CC BY-SA 4.0.
- OpenAssistant OASST1 à la révision
  `fdf72ae0827c1cda404aff25b6603abec9e3399b`, messages français humains ayant
  passé la revue, licence Apache-2.0. Les messages supprimés et synthétiques sont
  rejetés.
- Conversations ivoiriennes facultatives, uniquement avec consentement explicite,
  absence déclarée de données personnelles et licence CC0-1.0 ou CC-BY-4.0.

Facebook, X/Twitter et les conversations privées ne sont pas aspirés. Un contenu
visible publiquement n'est pas automatiquement réutilisable pour l'entraînement.

## Garanties

- téléchargement reprenable et empreintes SHA256 pour OASST1 ;
- groupes de conversation séparés entre entraînement et validation ;
- aucun split test créé ;
- déduplication exacte normalisée ;
- masquage des courriels et formes courantes de secrets ;
- refus d'entraîner tant que les seuils v1.1 ne sont pas atteints.

Le lanceur Colab est `scripts/colab/run_corpus_v11_collection.py`. La première
collecte Wikipédia peut durer longtemps, mais son cache permet de reprendre après
une déconnexion. À la fin, l'audit indique précisément ce qui manque encore ; il
ne lance jamais le modèle.
