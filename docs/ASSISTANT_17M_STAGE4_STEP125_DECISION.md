# Décision — assistant 17M stage 4, étape 125

Date de l'évaluation : 2026-09-02  
Statut : candidat anti-collapse conservé, mais non qualifié comme assistant.

## Artefacts évalués

- Modèle : `microivoire_transformer_v1.2_17m_assistant_stage4`
- Étape : 125
- SHA256 du checkpoint : `ccb26306e027f2b99e3f5ce20e1cf08ad89ead96c0877645c7c0292e5213dbf8`
- SHA256 du feedback source : `6f97b00a90bf5f5121cfb9cddb6c6237bdeeee788f6b8baca7c8da672d0e674f`
- SHA256 du benchmark figé : `f3b078da75200e68af6b891abdf1f2ea94aecec4e3214034e1f506b45a8c726`
- SHA256 des prédictions : `953dab9174f66e540645cbc36014a8c0f48e92b95eb6ff214e312deef4024313`
- Questions humaines uniques : 66
- Entraînement effectué pendant l'évaluation : non
- Test final scellé ouvert : non

## Résultats automatiques

| Mesure | Stage 3 | Stage 4 étape 125 |
|---|---:|---:|
| Sorties différentes | 19/66 | 41/66 |
| Taux de diversité | 28,8 % | 62,1 % |
| Réponse dominante | 22,7 % | 10,6 % |
| Réponses stéréotypées | 50,0 % | 0,0 % |
| Terminaison EOS propre | 100,0 % | 100,0 % |
| Longueur moyenne | non consignée ici | 4,91 tokens |
| Correspondance exacte indicative | 3/66 | 0/66 |

Le stage 4 répare donc une grande partie de l'effondrement de diversité du
stage 3. Cette amélioration n'implique toutefois pas une amélioration de la
justesse.

## Examen qualitatif

Plusieurs sorties montrent un comportement d'écho ou une réponse incomplète :

- `2+2` produit `2+2` au lieu de `4` ;
- `C'est quoi Google ?` produit `Google` ;
- `Parle de la philosophie` produit `la philosophie` ;
- de nombreuses questions factuelles ivoiriennes produisent un nombre isolé,
  un morceau de la question ou une réponse sans rapport.

Le candidat a appris à s'arrêter proprement et à éviter une réponse unique
répétée, mais pas encore à transformer systématiquement une question en
réponse informative.

## Limite du benchmark humain

Les 66 questions sont désormais un jeu de développement, puisqu'elles ont été
consultées à plusieurs reprises. Elles ne doivent pas être présentées comme un
test final aveugle. Certaines corrections humaines sont en outre erronées,
ambiguës, dépendantes de la date ou incompatibles avec la question. Elles
doivent être auditées avant tout usage pédagogique. Elles ne sont pas injectées
dans l'entraînement du stage 4.

## Décision

1. Conserver et figer le checkpoint de l'étape 125 comme preuve de la réparation
   anti-collapse.
2. Ne pas poursuivre automatiquement le stage 4 jusqu'à 250.
3. Ne pas publier ce checkpoint comme assistant prêt à l'emploi.
4. Construire un stage suivant centré sur la réponse directe, la complétude,
   le calcul exact et des faits ivoiriens vérifiés, avec pénalité d'écho.
5. Utiliser les 66 questions uniquement pour le diagnostic de développement ;
   créer plus tard un nouveau test final, indépendant et scellé.

