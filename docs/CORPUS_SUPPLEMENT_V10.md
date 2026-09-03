# IvoireSLM — supplément de corpus v1.0

Ce lot transforme une sélection prudente des recherches archivées dans Google Drive en un supplément reproductible pour le futur corpus IvoireSLM.

## Pourquoi un supplément séparé

Le corpus v0.9 contient déjà le socle ivoirien, français et multilingue. Les nouvelles données sont fortement orientées vers les mathématiques, le code et la cybersécurité. Une concaténation directe déséquilibrerait un modèle de 17 millions de paramètres et augmenterait ses répétitions. Le nouveau lot reste donc séparé jusqu'à ce qu'un mélange d'entraînement soit mesuré et validé.

## Sources admises au pilote

- Wikipedia français et anglais : texte naturel, conservé dans une couche ShareAlike distincte.
- GSM8K : split train uniquement.
- AQuA : train et développement uniquement ; test exclu.
- DeepSeek Harness : documentation et outils sous licence MIT ; fichier de licence non utilisé comme donnée d'apprentissage.
- CISA KEV : descriptions et mesures de remédiation défensives uniquement.

## Sources exclues

- Tous les benchmarks de test, notamment MATH-500.
- Les contenus non commerciaux et les licences à vérifier.
- Les gros flux NVD et OSV pendant ce pilote.
- Les binaires, charges d'exploitation et preuves de concept.

## Garanties

- Aucun split test créé.
- Split train/validation déterministe par groupe.
- Déduplication exacte globale après normalisation.
- Détection et masquage des formes courantes de secrets et des adresses électroniques.
- Provenance, licence, classification et SHA256 conservés.
- Refus d'écraser un dataset déjà construit.

## Suite scientifique

Après la construction, on mesure la répartition par domaine et par langue. On ne relance pas immédiatement un préentraînement. Le pilote proposé conserve 60 % du corpus v0.9, puis échantillonne 15 % de Wikipedia français, 5 % de Wikipedia anglais, 10 % de raisonnement mathématique, 5 % de code/agents et 5 % de cybersécurité défensive. Ces poids sont dans `configs/corpus_mix_v10.json`.

On entraîne ensuite un petit palier comparatif avec une validation générale, ivoirienne, mathématique et cybersécurité. Le test final reste scellé.
