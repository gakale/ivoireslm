# IvoireSLM — curriculum mathématique v0.3.2 « tables »

## Pourquoi une nouvelle branche ?

Le curriculum v0.3.1 a amélioré les additions, mais ses résultats ont oscillé :
le meilleur score est resté à l'étape 500, avec 13,3 % d'additions exactes et
0 % de multiplications exactes. À l'étape 750, l'addition a reculé à 11,1 % et
la multiplication est remontée à seulement 4,2 %.

Cette expérience sépare donc deux questions qui étaient mélangées :

1. le modèle peut-il **mémoriser et rappeler** les tables de 2 à 12 ?
2. peut-il **généraliser** à des produits jamais enseignés, de 13 à 20 ?

Le premier objectif sert à sélectionner le modèle. Le second est uniquement un
diagnostic et n'influence ni le gradient ni le choix du meilleur checkpoint.

## Données

### Entraînement — 1 850 exemples

- 882 révisions d'addition ;
- 484 questions directes sur les tables 2–12 ;
- 242 indices par addition répétée ;
- 242 indices par décomposition autour de 10.

### Validation de rappel — 562 exemples

- 441 additions avec une formulation absente de l'entraînement ;
- 121 produits couvrant toutes les tables 2–12, avec une formulation absente
  de l'entraînement.

Les faits 2–12 apparaissent volontairement en entraînement et en validation :
on mesure le rappel des tables, pas la généralisation algorithmique.

### Diagnostic inédit — 64 exemples

Les 64 produits ordonnés de 13×13 à 20×20 ne sont jamais utilisés pour le
gradient ni pour sélectionner le meilleur modèle.

## Sélection

Le score est composé de 85 % de rappel des multiplications et 15 % de révision
des additions. Un checkpoint ne peut devenir le meilleur que si la loss de
langue générale reste dans une marge de +2 % par rapport au parent.

Seuils du pilote :

- rappel des tables ≥ 60 % ;
- addition ≥ 20 % ;
- garde de langue générale respectée.

## Parent et isolation

Le parent attendu est le meilleur checkpoint du curriculum v0.3.1, étape 500.
La sortie v0.3.2 doit aller dans un nouveau dossier. Aucun test scellé n'est
créé ou ouvert.

## Limite assumée

Réussir les tables 2–12 ne prouve pas que le modèle sait multiplier n'importe
quels nombres. C'est précisément pourquoi le diagnostic 13–20 est séparé et
rapporté sans intervenir dans la sélection.
