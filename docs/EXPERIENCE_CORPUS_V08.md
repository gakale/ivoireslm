# IvoireSLM — préparation du corpus v0.8

## Pourquoi une nouvelle version

Le Transformer v0.3 de 17 129 280 paramètres a été entraîné jusqu'à l'étape
10 400 sur le corpus v0.7 et le tokenizer BPE v0.3. La loss de validation est
passée de 9,0761 à 3,3701 et la perplexité de 8 743,9 à 29,08. Malgré cette
amélioration, les générations libres ont montré trois défauts :

- association excessive entre « Côte d'Ivoire » et les phrases WDI ;
- répétitions encyclopédiques et mélange d'entités ;
- reproduction du format mathématique sans calcul exact.

Cette expérience montre que la perplexité mesure la prédiction de la
distribution de validation, pas directement la factualité, le suivi
d'instructions ou le calcul.

## Diagnostic du corpus v0.7

- 50 706 925 caractères et 12 898 145 tokens BPE d'entraînement ;
- 67,56 % issus du snapshot Wikipédia diversifié v0.1 ;
- environ 9,8 % de français ivoirien naturel ;
- 8,56 % de mathématiques ;
- 11,11 % de données factuelles structurées ;
- plusieurs sources synthétiques regroupées dans de très gros documents.

WDI ne représente que 1,3 % des caractères, mais ses squelettes de phrases
répétés contiennent directement « Côte d'Ivoire ». Leur influence
conditionnelle est donc beaucoup plus visible que leur poids global.

## Changements v0.8

1. Le snapshot Wikipédia v0.2 vise plus de 50 000 pages candidates dans des
   catégories françaises variées, dont Côte d'Ivoire et mathématiques.
2. Les anciens documents sont forcés dans `train`; seuls de nouveaux groupes
   peuvent entrer dans `validation` ou `test`.
3. Les nombres et contenus entre guillemets sont masqués pour calculer une
   signature de gabarit.
4. Un même squelette synthétique est plafonné à 24 lignes.
5. Les gros fichiers sont découpés à 32 000 caractères maximum par document.
6. Les lignes exactes restent dédupliquées globalement.
7. Le corpus v0.7 et son test gelé ne sont jamais modifiés.

## Garde-fous candidats

- au moins 150 millions de caractères ;
- au moins 20 000 documents ;
- au moins 70 % de français naturel ;
- au moins 8 % de français ivoirien naturel ;
- entre 7 % et 13 % de mathématiques ;
- au plus 3 % de dictionnaire ;
- au plus 5 % de données factuelles structurées ;
- au moins 3 millions de caractères dans chaque split gelé ;
- aucune fuite de groupe, aucun ancien document dans les splits gelés ;
- aucun caractère de contrôle.

Ces seuils sont évalués avant publication. Un candidat qui échoue est conservé
sous le suffixe `.rejected` pour diagnostic, sans devenir le corpus officiel.

## Étape suivante

Après validation du corpus, le tokenizer BPE v0.4 est entraîné uniquement sur
`train`. Le même Transformer 17M sera ensuite réentraîné depuis zéro afin
d'isoler l'effet du changement de données.
