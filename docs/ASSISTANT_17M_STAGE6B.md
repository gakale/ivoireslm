# Assistant 17M — Stage 6B ciblé

Le Stage 6 a amélioré son score de génération de 0,0331 à 0,1298 sans
effondrement ni perte de langue générale, mais son taux d'écho reste à 10,4 %
et sept compétences centrales restent à 0 % de réussite stricte.

Le Stage 6B est une nouvelle branche, jamais une continuation silencieuse. Il
part du meilleur checkpoint Stage 6 étape 250 et réutilise exactement le même
curriculum vérifié. Les dix corrections du contrôle humain restent hors de
l'entraînement et de la validation.

Le changement expérimental est limité : 25 % de répétition de langue générale
au lieu de 60 %, un taux d'apprentissage de `4e-6`, et davantage de poids pour
la conversation, l'identité, les connaissances, le dioula élémentaire, les
faits ivoiriens et les calculs. Le premier palier s'arrête à 125 étapes.

Le pilote n'est candidat au contrôle humain aveugle que si la qualité progresse,
le taux d'écho passe sous 5 %, la diversité ne s'effondre pas, la hausse de loss
générale reste au plus à 0,01 et au moins quatre des sept compétences centrales
obtiennent une réussite stricte non nulle. Aucun test final n'est créé ou ouvert.
