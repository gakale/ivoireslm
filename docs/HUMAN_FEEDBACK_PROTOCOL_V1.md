# Protocole de retours humains v1

La réussite de la validation automatique ne suffit pas à publier IvoireSLM 17M.
Le candidat stage 3 doit être testé sur des formulations humaines inédites.

Chaque retour conserve la question, la réponse, la route, le checkpoint exact, une
catégorie, un jugement et, en cas d'erreur, une correction. Aucun nom, adresse,
secret ou autre donnée personnelle ne doit être saisi.

Un retour n'est jamais ajouté automatiquement à l'entraînement. Avant un éventuel
stage 4, les lignes doivent être dédupliquées, vérifiées, réparties entre
entraînement et validation, puis figées avec leurs empreintes SHA256. Les
traductions dioula nécessitent en plus une validation par un locuteur compétent.

La publication du poids candidat exige au minimum :

- 30 questions humaines inédites et consenties ;
- au moins 24 réponses jugées acceptables ;
- aucune régression critique de sécurité ou d'intégrité ;
- une fiche modèle décrivant clairement la taille et les limites ;
- la conservation du test final scellé jusqu'à la décision de publication.
