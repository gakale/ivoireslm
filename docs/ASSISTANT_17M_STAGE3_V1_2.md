# Assistant 17M — stage 3 contrastif v1.2

La qualification du stage 2 à l'étape 500 montre que les 30 échecs de
transformation concernent exclusivement les majuscules. La copie et les minuscules
réussissent 60 cas sur 60. Les erreurs restantes d'identité et d'incertitude sont
également des confusions entre réponses valides apprises.

Le stage 3 est donc une nouvelle expérience issue du meilleur checkpoint stage 2,
sans reprise de son optimiseur et sans modification des branches précédentes.

- 866 exemples d'entraînement et 122 de validation ;
- 288 exemples d'entraînement spécifiquement consacrés aux majuscules ;
- séparation par combinaisons sujet-prédicat, plutôt que par sujets entiers ;
- nouvelles formulations contrastives d'identité, salutation, limites et prudence ;
- learning rate prudent de `5e-6` ;
- 40 % de lots de langue générale pour limiter l'oubli ;
- aucun split test créé ou ouvert.

Le premier palier est limité à 250 étapes locales et doit être qualifié avant toute
continuation.
