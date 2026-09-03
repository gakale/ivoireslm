# Assistant 17M — Stage 5B, correction ciblée

## Diagnostic du stage 5

Entre les étapes 125 et 250, le stage 5 améliore son score de génération de
`0,1914` à `0,2929`. Le taux d'écho tombe de 9 % à 2,2 %, la diversité reste à
76,9 % et la loss de langue générale reste protégée. Les connaissances
ivoiriennes atteignent toutefois seulement 12,5 %, le calcul exact 3,1 % et les
connaissances générales 0 %.

L'examen des sorties révèle deux causes. D'abord, les réponses de prudence et
le gabarit « Le résultat de… » contaminent des questions ordinaires. Ensuite,
la normalisation utilisée par le curriculum stage 5 supprimait les opérateurs
`+`, `−`, `×` et `÷` lors de la déduplication. Des exercices distincts pouvaient
donc recevoir la même clé textuelle et être écartés. Continuer mécaniquement
jusqu'à l'étape 375 n'aurait pas corrigé cette erreur de données.

## Correction

Le stage 5B repart du checkpoint stage 5 étape 250 dans un nouveau dossier. Il
ne modifie aucun ancien checkpoint. Son curriculum :

- conserve explicitement les opérateurs pendant la normalisation ;
- apprend les additions de 0 à 20, les tables de multiplication de 0 à 12 et
  les soustractions positives de 0 à 20 ;
- varie les formes de réponse mathématique afin de réduire la contamination
  par un gabarit unique ;
- renforce la conversation ordinaire et dix définitions générales ;
- ajoute davantage de contrastes entre une question répondable et une question
  réellement incertaine ;
- augmente le poids des faits ivoiriens et diminue celui des refus ;
- conserve 55 % de lots de langue générale pour limiter l'oubli.

Les formulations de validation restent absentes de l'entraînement, mais les
faits arithmétiques et publics peuvent se retrouver des deux côtés sous des
paraphrases différentes. Cette évaluation mesure donc l'acquisition et le
rappel des compétences ciblées, pas une généralisation mathématique hors
distribution.

Les réponses humaines ne sont jamais utilisées comme cibles. Les 66 questions
déjà consultées servent seulement à exclure leurs formulations exactes. Aucun
split test n'est créé et le test final reste scellé.

## Décision prévue

Le premier pilote s'arrête à 125 étapes. Une poursuite n'est autorisée que si
la qualité augmente, si les gardes anti-écho et anti-collapse passent, et si la
loss de langue générale n'augmente pas de plus de `0,01`. Les générations
doivent ensuite être examinées manuellement avant toute publication.
