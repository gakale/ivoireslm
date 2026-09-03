# Assistant 17M — curriculum correctif v1.1 stage 2

Le checkpoint assistant à l'étape 500 conserve correctement la langue générale et
réussit la compréhension de lecture, mais il ne satisfait pas encore les critères
de publication. Il confond notamment l'identité, les limites, le refus d'inventer
et les transformations de texte.

Ce stage 2 repart du `best.pt` de l'étape 500 sans modifier l'expérience précédente.
Il crée une nouvelle branche d'entraînement et un nouveau dossier Drive.

## Données

- 480 exemples d'entraînement ;
- 132 exemples de validation ;
- aucun prompt commun entre entraînement et validation ;
- aucun split test créé ;
- davantage de variantes d'identité, de limites et d'incertitude ;
- 360 consignes de copie, minuscules et majuscules sur des phrases distinctes ;
- compréhension de lecture avec des noms et villes non vus à l'entraînement ;
- faits ivoiriens conservés et reformulés à partir des sources déjà contrôlées.

Les réponses attendues restent courtes afin de mesurer une compétence précise et
d'éviter de récompenser une génération longue mais incohérente.

## Entraînement

Le learning rate est abaissé à `7e-6`. La langue générale représente 35 % des lots
et sert aussi de garde de non-régression. Les consignes représentent 30 % des lots,
car c'est l'échec principal observé au palier 500.

Le premier palier s'arrête à 250 étapes locales. Il ne faut pas continuer tant que
la qualification stricte n'a pas mesuré séparément les cinq familles. Le test final
reste scellé.
