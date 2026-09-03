# Assistant 17M — stage 4 anti-collapse

## Pourquoi une nouvelle branche

Le stage 3 a réussi son petit benchmark synthétique mais a échoué devant les
questions humaines. Sur 66 questions inédites, il n'a produit que 19 sorties
différentes, contre 62 pour le checkpoint CPT v1.0. La moitié des réponses
contenait une formule stéréotypée apprise pendant la supervision.

Le stage 4 repart donc du meilleur checkpoint CPT v1.0 à l'étape 1 000. Il ne
continue pas depuis le stage 3 et ne réutilise jamais les réponses aux 66
questions humaines comme données d'entraînement.

## Données

Le curriculum sépare six compétences : conversation ordinaire, incertitude
calibrée, faits ivoiriens sourcés, calcul exact, suivi d'instructions et
compréhension de texte. Il ajoute des variantes orthographiques et des
contrastes entre une question réellement ambiguë et une question simple qui
mérite une réponse directe.

Le constructeur reçoit obligatoirement le fichier de retours humains dans le
pilote Colab. Toute question identique est écartée avant l'écriture de `train`
et `validation`. Aucun split `test` n'est créé.

## Protections

- taux d'apprentissage maximal : `3e-6` ;
- 65 % des minibatches proviennent du corpus de langue général ;
- premier palier limité à 125 étapes ;
- sélection pénalisée si les sorties deviennent moins diverses que les cibles ;
- rejet si une réponse devient anormalement dominante ;
- rejet si la loss générale augmente de plus de `0.01` ;
- comparaison humaine seulement après réussite de ces gardes.

Le stage 4 reste expérimental. Une réussite interne autorise uniquement une
nouvelle évaluation humaine à l'aveugle ; elle n'autorise pas encore la
publication du poids comme assistant fiable.
