# Continuation de préentraînement v1.0 — pilote 17M

Ce pilote vérifie le nouveau mélange avant toute architecture de 100 millions de paramètres.

Le modèle repart des poids du meilleur checkpoint v0.4 à l'étape 20 750. L'optimiseur est volontairement réinitialisé et le taux maximal est réduit à `2e-5`. Les données sont tirées selon le mélange documenté : 60 % du corpus v0.9, 15 % Wikipedia FR, 5 % Wikipedia EN, 10 % mathématiques, 5 % code/agents et 5 % cybersécurité défensive.

Les validations restent séparées par domaine. Le checkpoint n'est promu que si le score pondéré baisse et si la loss du corpus général v0.9 n'augmente pas de plus de `0.03`. Aucun fichier test n'est chargé par le programme.

Le premier palier s'arrêtait à 250 étapes. Le pilote a ensuite été poursuivi jusqu'à 1 000 étapes. La loss pondérée est passée de 3,1619 à 2,8577 et la garde du corpus général est restée valide, mais les générations qualitatives sont encore trop répétitives et fragiles pour un assistant général. La décision complète est consignée dans [DECISION_CPT_V10_STEP1000.md](DECISION_CPT_V10_STEP1000.md).
