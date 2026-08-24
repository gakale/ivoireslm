# Investigation — routage des questions mathématiques naturelles

## Symptôme

Dans Gradio, `combien font 2+2` était envoyé au Transformer général et produisait
une suite incohérente au lieu de la réponse 4.

## Cause racine

Le routeur ne détectait que des mots comme `calculer` et `résoudre`. Le solveur
n'acceptait que les formulations structurées des générateurs de benchmarks.
La phrase naturelle ne correspondait donc à aucun patron et tombait dans la
génération libre du Transformer caractère 5M.

## Correction

- reconnaissance de `combien font`, `combien fait`, `ça fait combien`,
  `quel est le résultat de` et des expressions numériques seules ;
- évaluation par arbre syntaxique fermé, sans `eval` ;
- opérateurs autorisés : addition, soustraction, multiplication, division,
  parenthèses et petites puissances ;
- refus des appels de code, divisions par zéro, exposants hors limites et
  résultats démesurés ;
- avertissement visible : le Transformer 5M reste expérimental et n'est pas un
  assistant conversationnel général.

## Preuves

- reproduction originale : `combien font 2+2` → route déterministe → réponse 4 ;
- 70 tests automatisés réussis ;
- benchmark de non-régression : 1 000/1 000 exacts ;
- cas dangereux testés et refusés.

## Limite d'architecture

Cette correction améliore le routage et le calcul, pas l'intelligence générale
du Transformer. Pour obtenir un véritable assistant libre, il faudra changer la
phase modèle : tokenizer sous-mots, données instructionnelles et modèle plus
grand ou adaptation d'un petit modèle préentraîné.
