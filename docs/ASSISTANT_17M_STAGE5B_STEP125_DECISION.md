# Décision — assistant 17M Stage 5B, étape 125

Date : 2026-09-02  
Statut : expérience conservée comme preuve, entraînement arrêté.

## Résultat

Le Stage 5B améliore le score interne, mais ne produit toujours pas un assistant
général fiable. Les gardes techniques passent : diversité 77 %, réponse dominante
2,6 %, 2 % d'échos, aucune réponse vide et test final toujours scellé.

Les compétences restent insuffisantes : conversation et prudence calibrée à 0 %,
connaissances générales à 5 %, connaissances ivoiriennes à 18,8 % et calcul à
18,8 %. L'examen manuel montre que certains calculs reçoivent une réponse constante
incorrecte ; ce score ne prouve donc pas un raisonnement mathématique acquis.

## Décision

1. Figer le checkpoint Stage 5B étape 125 et son `progress.json`.
2. Ne pas poursuivre ce SFT et ne pas publier ce poids comme assistant prêt à l'emploi.
3. Revenir au checkpoint CPT avant la prochaine continuation de préentraînement.
4. Auditer et rééquilibrer le corpus en donnant la priorité au français naturel,
   aux conversations ivoiriennes vérifiées et à une encyclopédie générale propre.
5. Garder les mathématiques, le code et la cybersécurité comme petites composantes.
6. Après le nouveau CPT, refaire un SFT court, puis une évaluation humaine aveugle.

