# Assistant 17M — Stage 5 « réponse directe »

## Pourquoi ce stage existe

Le stage 4 a corrigé une partie du collapse observé au stage 3 : sur les 66
questions humaines, la diversité est remontée de 28,8 % à 62,1 % et les
réponses stéréotypées sont passées de 50 % à 0 %. L'examen manuel montre
cependant que le modèle produit encore surtout des fragments. Par exemple, il
peut répondre `Google` à « C'est quoi Google ? » ou recopier un calcul sans le
résoudre. Le checkpoint stage 4 n'est donc pas publiable comme assistant.

Le stage 5 est une nouvelle branche expérimentale depuis le meilleur
checkpoint du stage 4, à l'étape 125. Il cherche à apprendre une propriété
simple et mesurable : répondre par une phrase complète qui contient la bonne
information.

## Données

Le curriculum contient uniquement des splits `train` et `validation` et sept
familles : conversation ordinaire, connaissances générales, faits ivoiriens
vérifiés, calcul exact, suivi d'instructions, compréhension d'un contexte et
incertitude calibrée.

Les faits ivoiriens stables sont accompagnés de leur source institutionnelle.
Les calculs sont générés et vérifiés par programme. Les réponses du formulaire
humain ne sont jamais importées comme cibles d'entraînement : elles peuvent
être erronées. Le fichier de retours humains est lu uniquement pour exclure
les formulations exactes des 66 questions de développement.

Ces 66 questions ont déjà été inspectées et ne constituent plus un test à
l'aveugle. Un chevauchement sémantique avec un fait public stable, comme la
capitale politique, est accepté et déclaré ; la formulation exacte reste
exclue. Le test final demeure absent et scellé.

## Entraînement pilote

Le pilote effectue 125 étapes avec un faible taux d'apprentissage de `8e-6`.
Une moitié environ des lots conserve l'apprentissage de langue générale afin
de limiter l'oubli. L'autre moitié échantillonne les sept familles de manière
équilibrée, indépendamment de leur taille brute.

Le pilote repart du checkpoint stage 4 étape 125 et écrit dans un nouveau
dossier. Il refuse d'écraser une expérience existante.

## Mesures de décision

La sélection ne se limite pas à la loss supervisée. Les générations gloutonnes
sont vérifiées selon la famille : présence des éléments factuels requis,
dernier entier correct pour les calculs, exactitude des transformations et
similarité pour la conversation. Une pénalité est ajoutée pour :

- la répétition d'une même sortie ;
- les réponses dominantes ;
- la simple recopie d'un morceau de la question ;
- les sorties vides ou réduites à un seul mot.

Le checkpoint n'est candidat à une comparaison humaine que si sa qualité
augmente, si les gardes de réponse directe et de diversité passent, et si la
loss de langue générale n'augmente pas de plus de `0,01`.

La réussite de ce pilote ne vaut pas validation finale. Elle autorise seulement
une nouvelle comparaison sur le jeu de développement, suivie d'un véritable
test humain inédit avant toute publication Hugging Face.

## Sources factuelles principales

- Présentation institutionnelle de la Côte d'Ivoire :
  <https://diplomatie.gouv.ci/informations-utiles/presentation-de-la-c%C3%B4te-d-ivoire>
- Fiche signalétique de la Côte d'Ivoire :
  <https://oif.diplomatie.gouv.ci/fiche_signaletique.php>
- Histoire du franc CFA, BCEAO :
  <https://www.bceao.int/fr/content/histoire-du-franc-cfa>
- Présentation du Groupe de la Banque africaine de développement :
  <https://www.afdb.org/en/about-us>

