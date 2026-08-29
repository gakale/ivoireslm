# Expérience IvoireSLM Transformer 17M v0.3

Date de conception : 29 août 2026

## Objectif

Construire un modèle dense d'environ 17 millions de paramètres qui améliore
la modélisation du français naturel et ivoirien sans confondre taille du modèle
et qualité. L'expérience doit rester reproductible, explicable et comparable à
des baselines réentraînées sur les mêmes données.

## Hypothèses testées

1. Un corpus moins répétitif et plus naturel améliore davantage les sorties
   humaines que l'augmentation isolée du nombre de paramètres.
2. Un tokenizer BPE de 8 192 unités utilise mieux une fenêtre de 512 tokens que
   le tokenizer caractère de la v0.2.
3. Une architecture dense moderne d'environ 17M paramètres peut apprendre des
   dépendances plus longues que le modèle 4,76M, tout en restant entraînable sur
   un GPU T4.
4. Les gains de perplexité doivent être confirmés par des évaluations séparées
   de français, de faits ivoiriens, de génération et de mathématiques.

## Corpus v0.7 visé

Le corpus v0.7 part de la v0.6 sans l'écraser. Il ajoute un snapshot Wikimedia
français diversifié et attribué. Les cibles de qualité sont :

- au moins 40 millions de caractères pour le premier candidat ;
- au moins 45 % de texte naturel encyclopédique ;
- 5 à 15 % de mathématiques ;
- au plus 12 % de dictionnaire ;
- au plus 25 % de sources factuelles structurées ;
- aucune fuite de page ou de groupe entre train, validation et test ;
- déduplication exacte globale et empreintes SHA-256 complètes.

Cette taille est suffisante pour un **pilote**, mais elle reste inférieure au
volume idéal pour entraîner 17M paramètres depuis zéro. Une étape ultérieure
devra augmenter les tokens uniques si la courbe de validation justifie le coût.

## Tokenizer BPE v0.3

Le BPE apprend des sous-mots fréquents. Par exemple, plusieurs caractères de
`agriculture` peuvent devenir une seule unité. Il apporte trois avantages :

- davantage de texte visible dans 512 tokens ;
- unités plus proches des mots et morphèmes ;
- génération plus rapide, car moins de pas sont nécessaires.

Le tokenizer est entraîné uniquement sur `train`. Validation et test restent
fermés pendant l'apprentissage de son vocabulaire. Le vocabulaire de 8 192
unités tient dans des fichiers `uint16`.

## Architecture dense prévue

| Élément | Valeur | Rôle |
|---|---:|---|
| Vocabulaire | 8 192 | Sous-mots BPE |
| Contexte | 512 tokens | Plusieurs paragraphes courts |
| Dimension | 320 | Largeur des représentations |
| Blocs | 12 | Profondeur du raisonnement séquentiel |
| Têtes | 8 | Relations parallèles dans le contexte |
| FFN | 832 | Transformation SwiGLU |
| Normalisation | RMSNorm | Stabilité et simplicité |
| Position | RoPE | Position relative sans table apprise fixe |
| Embeddings | partagés | Réduit les paramètres inutiles |

Le nombre exact de paramètres doit être calculé et inscrit dans le rapport par
le code. Le nom `17M` est une classe de taille, pas une affirmation arrondie
sans vérification.

## Entraînement en paliers

L'entraînement complet n'est pas lancé aveuglément :

1. **Smoke test** : quelques étapes pour vérifier formes, GPU et sauvegarde.
2. **Pilote** : 5 200 étapes, environ 85 millions de tokens vus avec le batch
   effectif configuré.
3. **Décision** : comparer train/validation, échantillons et benchmarks.
4. **Extension** : reprendre jusqu'à 10 400 puis 20 800 étapes seulement si la
   validation continue de progresser sans mémorisation excessive.

Avec 20 800 étapes, batch 16, accumulation 2 et contexte 512, le modèle voit
environ 340,8 millions de positions d'entraînement. Les répétitions du corpus
seront comptées séparément des tokens uniques.

## Critères avant ouverture du test gelé

- toutes les empreintes des données sont correctes ;
- loss finie et gradients finis ;
- validation évaluée de façon déterministe ;
- meilleur checkpoint choisi uniquement sur validation ;
- test ouvert une seule fois après sélection ;
- checkpoints `best.pt` et `latest.pt` copiés dans Cloud Storage ;
- rapport contenant configuration, durée, tokens vus et SHA-256.

## Critères de réussite

Le modèle 17M ne sera pas déclaré meilleur sur sa seule perplexité. Il devra :

- battre une baseline 5M réentraînée sur le corpus/tokenizer v0.3 ;
- réduire les répétitions et dérives WDI observées ;
- produire du français plus cohérent sur des prompts jamais utilisés dans les
  exemples de démonstration ;
- mieux respecter la forme de questions simples, sans revendiquer une réponse
  exacte lorsqu'elle ne l'est pas ;
- conserver le routeur mathématique comme outil externe vérifié.

## Pourquoi le MoE est reporté

Un Mixture of Experts active seulement certains sous-réseaux pour chaque token.
Il permet d'augmenter le nombre total de paramètres sans augmenter le calcul
dans la même proportion. Mais il ajoute un routeur, une perte d'équilibrage, des
risques d'experts inutilisés et une évaluation plus difficile.

À 17M, un modèle dense constitue donc la meilleure expérience de contrôle. Le
MoE sera testé plus tard comme comparaison autour du projet 300M, après avoir
validé les données, les métriques et une baseline dense de calcul comparable.
