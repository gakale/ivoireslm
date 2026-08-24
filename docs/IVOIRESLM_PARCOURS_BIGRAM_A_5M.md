# IvoireSLM — du Bigram au Transformer 5M

> Journal technique, preuves du parcours et leçons apprises
>
> Période documentée : 13 au 24 août 2026
>
> Dernière mise à jour : 24 août 2026
>
> Statut : parcours expérimental reproductible ; prototype de recherche, pas encore assistant conversationnel général

## 1. Résumé exécutif

IvoireSLM est parti d'un corpus ivoirien audité et d'un modèle Bigram très simple. Le projet a ensuite progressé vers un MLP contextuel, un Transformer de 1,10 million de paramètres, puis un Transformer de 4,76 millions de paramètres entraîné sur un corpus rééquilibré. Un entraînement supervisé mathématique a enfin été testé, puis complété par un moteur déterministe pour garantir les calculs pris en charge.

Le résultat le plus important n'est pas seulement le passage de 1,30 à 4,76 millions de paramètres. C'est la mise en place d'une démarche scientifique : données versionnées, splits gelés, empreintes SHA-256, baselines, checkpoints, évaluations séparées, tests de non-régression et conservation des échecs.

Les principales conclusions sont les suivantes :

1. Le Bigram a fourni une référence minimale reproductible.
2. Ajouter du contexte ne suffit pas : le MLP a fait moins bien que le Bigram sur le test gelé.
3. L'attention a aidé : le Transformer 1,10M a battu les deux premières baselines.
4. La qualité et l'équilibre du corpus comptent autant que sa taille brute.
5. Le Transformer 4,76M a fortement amélioré la perplexité sur ses propres splits, sans devenir un bon assistant.
6. Une faible perplexité ne prouve ni le raisonnement, ni l'exactitude factuelle, ni la capacité à suivre une instruction.
7. Le SFT mathématique a appris le format des réponses, mais presque pas l'algorithme de calcul.
8. Le moteur mathématique atteint 100 % uniquement sur les familles et benchmarks couverts par ses règles. Ce score ne doit jamais être présenté comme l'intelligence générale du Transformer.
9. Le prototype hybride sait distinguer une réponse vérifiée d'une génération libre non vérifiée.
10. Pour obtenir un assistant réellement utile, la voie la plus réaliste est l'adaptation d'un modèle préentraîné plus grand, tout en conservant le modèle 5M comme piste de recherche pédagogique.

## 2. Ce que nous cherchions à démontrer

Le projet avait quatre objectifs complémentaires :

- construire un corpus propre, traçable et majoritairement pertinent pour la Côte d'Ivoire ;
- apprendre chaque étape de la construction d'un modèle de langage depuis zéro ;
- comparer les modèles sur des mesures objectives plutôt que sur quelques textes générés ;
- conserver tous les artefacts nécessaires pour reproduire ou auditer les expériences.

Le chemin suivi peut être résumé ainsi :

```text
Sources ouvertes et autorisées
        ↓
Nettoyage, provenance, déduplication et splits gelés
        ↓
Tokenizer caractère
        ↓
Bigram → MLP contextuel → Transformer 1,10M
        ↓
Corpus v0.6 rééquilibré + tokenizer v0.2
        ↓
Transformer 4,76M
        ↓
Benchmark mathématique → SFT → diagnostic d'échec
        ↓
Moteur mathématique déterministe + routeur hybride
```

## 3. Principes de preuve et de reproductibilité

Le projet ne repose pas uniquement sur des captures d'écran. Les éléments suivants constituent les preuves techniques :

- chaque version importante possède un rapport Markdown dans `reports/` ;
- les données volumineuses et checkpoints sont conservés sur la VM ou dans Cloud Storage ;
- les fichiers critiques possèdent une empreinte SHA-256 ;
- le tokenizer est entraîné uniquement sur `train` ;
- les groupes de sources ne traversent pas les splits ;
- le test gelé n'est pas utilisé pour choisir les hyperparamètres ;
- le code est versionné dans Git ;
- les sorties négatives sont documentées au lieu d'être masquées.

Une empreinte SHA-256 agit comme une carte d'identité du fichier. Si un seul octet change, son empreinte change. Elle permet donc de prouver quel corpus ou checkpoint a réellement été utilisé.

## 4. Chronologie synthétique

| Étape | Date | Résultat principal | Décision |
|---|---:|---|---|
| Corpus v0.1.0 | 13 août | 6,26 M caractères, 13 documents | Première base validée |
| Corpus v0.3.0 | 13 août | 17,19 M caractères | Cible de 17 M caractères atteinte |
| Corpus v0.4.0 | 13 août | 32,53 M caractères, français ouvert ajouté | Diversifier le français |
| Corpus v0.5.0 | 13 août | 36,15 M caractères, 10,0039 % de maths | Préparer les premières expériences |
| Bigram-CI v0.1 | 20 août | PPL test 15,5437 | Baseline minimale établie |
| MLP-CI v0.1 | 22 août | PPL test 16,3381 | Échec utile ; passer à l'attention |
| Transformer v0.1 | 22 août | 1,10M ; PPL test 14,3929 | Première victoire sur le Bigram |
| Corpus v0.6.0 | 23 août | 14,32 M caractères mieux équilibrés | Privilégier qualité et diversité |
| Transformer v0.2 5M | 23 août | 4,76M ; PPL test 3,0791 | Évaluer le raisonnement séparément |
| SFT math v0.1 | 23–24 août | Format 100 %, exactitude 0,2–0,3 % | Ne plus poursuivre ce seul objectif |
| Outil math + hybride | 24 août | 100 % sur benchmarks couverts | Séparer modèle probabiliste et calcul exact |

`PPL` signifie perplexité. Plus elle est basse, mieux le modèle prédit le token suivant sur le jeu évalué. Elle ne mesure pas directement la vérité, le raisonnement ou la qualité d'une conversation.

## 5. Construction progressive du corpus

### 5.1 Corpus v0.1.0 — fondation ivoirienne contrôlée

La première livraison validée contenait :

- 13 documents ou groupes ;
- 34 482 phrases factuelles ;
- 46 791 faits atomiques ;
- environ 1 054 825 mots ;
- 6 262 547 caractères.

Les splits étaient séparés par groupe de source : 9 groupes pour l'entraînement, 2 pour la validation et 2 pour le test. Le test comprenait notamment le RGPH 2021 et les résultats du BAC. Aucun groupe ne traversait deux splits.

Les contrôles ont trouvé zéro doublon exact de document, zéro doublon exact de phrase normalisée, zéro fuite entre groupes, zéro adresse courriel et zéro numéro de téléphone. Les ressources sous droit d'auteur ou à licence incertaine ont été gardées hors du corpus d'entraînement.

Archive : `ivoireslm_corpus_v0.1.0.tar.gz`

SHA-256 : `2ebf8b63428199aea6d0c7e266f1770bdcba659e2e2741eda3f0e011566050e2`

**Leçon :** accumuler des fichiers n'est pas encore construire un corpus. Il faut connaître leur origine, leur licence, leurs doublons et leur destination dans les splits.

### 5.2 Corpus v0.3.0 — cible de 17 millions de caractères

La v0.3 a atteint :

- 17 192 011 caractères ;
- 2 781 979 mots ;
- 91 870 phrases ;
- 104 179 faits atomiques ;
- 15 documents ou groupes.

Les apports majeurs étaient les World Development Indicators et FAOSTAT. WDI représentait 40 477 observations et 1 423 indicateurs couvrant 1960 à 2025. FAOSTAT apportait 16 911 observations retenues sur 121 produits.

Le seuil des 17 millions a été atteint à 101,13 %. Il s'agissait de **caractères-tokens**, car le tokenizer visé était caractère. Ce nombre ne doit pas être confondu avec 17 millions de tokens BPE ou SentencePiece.

SHA-256 de l'archive : `ee0c471573738b1f47358d50bb636371aac2c2f08013be5b43d93d5f6374d5c8`

**Leçon :** la cible de volume a été atteinte, mais une grande partie du texte était structurée et répétitive. Plus de volume ne signifiait donc pas automatiquement plus de langage naturel.

### 5.3 Corpus v0.4.0 — diversification du français

La v0.4 a atteint 32 534 930 caractères et 5 001 595 mots. Le noyau ivoirien factuel représentait 52,84 % des caractères et les nouvelles ressources françaises ouvertes 47,16 %.

Deux familles ont été ajoutées :

- Wiktionnaire français : 48 690 lemmes, 63 240 sens, 12 086 605 caractères ;
- documentation Python française : 17 258 blocs, 3 256 314 caractères.

SHA-256 de l'archive : `019a8ccb61441a3c826dae2fa468baaad5ee6d698a1596352777f442269fe113`

**Leçon :** le dictionnaire améliore la couverture lexicale et la documentation apporte du français technique, mais ces textes ne remplacent pas les conversations et récits naturels ivoiriens.

### 5.4 Corpus v0.5.0 — objectif de 10 % de mathématiques

La v0.5 a atteint :

- 36 151 493 caractères ;
- 5 605 013 mots ;
- 184 155 lignes ;
- 210 483 faits ;
- 19 documents et 9 domaines.

Les mathématiques représentaient 3 616 563 caractères, soit 10,0039 % du corpus. Elles provenaient principalement de Wikilivres et de Wikipédia en français, avec 13 725 formules LaTeX repérées.

SHA-256 de l'archive : `6a45c958497b7b64fb5bef09dd1f14187b6c15acb5843556da77c86e33a7e5b1`

**Leçon :** lire des articles mathématiques ne suffit pas à apprendre à résoudre des exercices. Le corpus contenait des explications et des formules, mais pas encore assez de paires structurées `problème → méthode → solution → réponse`.

### 5.5 Corpus v0.6.0 — moins volumineux, mais mieux équilibré

La v0.6 a été reconstruite pour corriger les répétitions :

- 25 documents ;
- 14 316 283 caractères ;
- 2 217 561 mots ;
- 83 983 lignes ;
- 1 623 209 caractères de maths, soit 11,3382 % ;
- 585 064 caractères de français naturel ivoirien ajoutés, soit 4,0867 % ;
- 6 400 exercices mathématiques structurés et vérifiés ;
- 179 articles naturels liés à la Côte d'Ivoire.

Les motifs WDI ont été réduits de 40 477 à 3 808 lignes. Le Wiktionnaire a été plafonné à environ 4 millions de caractères. Les 6 400 exercices couvrent le format problème, méthode, solution et réponse ; 100 % de leurs réponses ont été vérifiées par programme.

| Split v0.6 | Documents | Caractères | Mots |
|---|---:|---:|---:|
| Train | 17 | 14 060 827 | 2 177 978 |
| Validation | 4 | 183 073 | 27 715 |
| Test | 4 | 72 383 | 11 868 |

Les 36 empreintes attendues ont été validées, sans fuite de groupe, PII détectée ni caractère de contrôle.

Archive Cloud Storage : `gs://legbairai-ivoireslm-artifacts-20260822/datasets/character_v0.2/ivoireslm_character_v0.2.tar.gz`

**Leçon :** la v0.6 est plus petite que la v0.5 volontairement. La réduction des répétitions et le meilleur équilibre rendent la comparaison « nombre brut de caractères » trompeuse. La qualité utile était devenue prioritaire.

## 6. Tokenizers caractère

### 6.1 Tokenizer v0.1

Le premier tokenizer a été entraîné uniquement sur le split `train` de la v0.5 :

- vocabulaire de 1 142 tokens, dont `<PAD>`, `<UNK>`, `<BOS>` et `<EOS>` ;
- 35 895 978 tokens train ;
- 136 182 tokens validation ;
- 119 352 tokens test ;
- taux de tokens inconnus : 0 % sur les trois splits ;
- stockage compact en entiers `uint16`.

SHA-256 du tokenizer : `d0ca1d6064d8ceb1b06c7b3b6518bbeed01af86221fb116d51fba110602185f4`

### 6.2 Tokenizer v0.2

Le tokenizer v0.2 accompagne le corpus v0.6 :

- vocabulaire de 722 tokens ;
- 14 060 844 tokens train ;
- 183 077 tokens validation ;
- 72 387 tokens test ;
- taux de tokens inconnus : 0 % ;
- test aller-retour texte → tokens → texte validé.

SHA-256 du tokenizer : `e3a6c91d7b4d766573346d363d486e60170f30081848b95e404d0a10afb97f95`

**Ce que signifie réellement 0 % de `<UNK>` :** le tokenizer sait représenter tous les caractères rencontrés. Cela ne signifie pas que le modèle comprend tous les mots. Un tokenizer caractère découpe un mot en nombreuses unités et utilise vite le contexte disponible. Avec une fenêtre de 256 tokens, le Transformer v0.2 ne voit qu'environ 256 caractères, pas 256 mots.

## 7. Bigram-CI v0.1 — la baseline minimale

### 7.1 Principe

Le Bigram estime le prochain caractère uniquement à partir du caractère courant. Il apprend par exemple que `q` est presque toujours suivi de `u`, mais ne peut pas comprendre une phrase entière.

```text
caractère actuel → table de probabilités → prochain caractère
```

### 7.2 Configuration et résultats

- type : maximum de vraisemblance sur bigrammes caractère ;
- vocabulaire : 1 142 ;
- paramètres de transition : 1 304 164 ;
- transitions train : 35 895 977 ;
- lissage de Laplace : 0,1 ;
- seed : 20260820 ;
- ajustement : split `train` uniquement.

| Split | Loss en nats | Perplexité |
|---|---:|---:|
| Train | 2,4472 | 11,5561 |
| Validation | 2,4735 | 11,8639 |
| Test gelé | 2,7437 | 15,5437 |

SHA-256 du checkpoint : `a296df9576c0dde6bf25a0d3f7de3b2f4e9a6e0726669a5480dfcecb45d5b303`

### 7.3 Ce que nous avons appris

La génération incohérente du Bigram était normale. Il avait appris l'orthographe locale de quelques transitions, pas la grammaire à longue distance, les faits ou le raisonnement. Son rôle était de fixer un seuil mesurable : un futur modèle devait faire mieux que 15,5437 de perplexité sur le même test.

## 8. Contextual MLP-CI v0.1 — premier contexte appris

Le MLP utilisait une fenêtre fixe de 16 caractères :

- 249 526 paramètres ;
- embeddings de dimension 32 ;
- couche cachée de 128 unités avec GELU ;
- dropout 0,1 ;
- batch 256, AdamW, taux d'apprentissage 0,001 ;
- 10 000 étapes et 2,56 millions d'exemples ;
- entraînement CPU en 167 secondes ;
- meilleur checkpoint à l'étape 4 400.

| Split | Loss en nats | Perplexité |
|---|---:|---:|
| Train échantillonné | 1,1997 | 3,3192 |
| Validation | 2,2587 | 9,5706 |
| Test gelé | 2,7935 | 16,3381 |

SHA-256 : `e574d62a6d46924161ba0a64a8c02e9e1e387d39a2a55b2f8439bce606fa9916`

Malgré sa bonne validation, le MLP était 5,11 % moins bon que le Bigram sur le test gelé. Il mémorisait facilement les patrons répétitifs de train, alors que le test contenait des groupes différents.

**Leçon :** plus de paramètres et une meilleure validation ne garantissent pas une meilleure généralisation. Les résultats négatifs doivent être conservés : ils justifient le passage à un modèle avec attention.

## 9. MicroIvoire Transformer v0.1 — première victoire sur les baselines

### 9.1 Architecture

- Transformer causal caractère, decoder-only ;
- 1 100 032 paramètres ;
- contexte de 128 caractères ;
- dimension d'embedding 128 ;
- 4 têtes d'attention ;
- 4 blocs ;
- dropout 0,1 ;
- entraînement sur GPU T4.

### 9.2 Entraînement et résultat

L'entraînement a duré 10 000 étapes et environ 295,3 secondes. Le meilleur checkpoint a été obtenu à l'étape 8 250.

| Split | Loss en nats | Perplexité |
|---|---:|---:|
| Validation | 1,7415 | 5,7057 |
| Test gelé | 2,6667 | 14,3929 |

Il a battu le Bigram de 15,5437 et le MLP de 16,3381 sur le même test. L'amélioration face au Bigram était d'environ 7,4 %.

Artefacts : `gs://legbairai-ivoireslm-artifacts-20260822/checkpoints/microivoire_transformer_v0.1/`

### 9.3 Limites observées

Le modèle produisait des séquences plus structurées, mais inventait encore des faits et des nombres, dérivait vers les patrons WDI ou Wiktionnaire et ne résolvait pas réellement les mathématiques.

**Leçon :** l'attention améliore la prédiction dans le contexte, mais un modèle de 1,10M entraîné au prochain caractère ne devient pas automatiquement un assistant.

## 10. MicroIvoire Transformer v0.2 5M

### 10.1 Architecture exacte

Le nom « 5M » désigne une classe de taille. Le nombre exact est :

- 4 758 144 paramètres ;
- contexte de 256 caractères ;
- dimension d'embedding 192 ;
- 6 têtes d'attention ;
- 10 blocs Transformer ;
- dropout 0,15 ;
- architecture decoder-only caractère.

### 10.2 Entraînement

- corpus : IvoireSLM v0.6.0 ;
- tokenizer : `ivoireslm_character_v0.2` ;
- GPU : NVIDIA T4 ;
- étapes : 10 000 ;
- meilleur checkpoint : étape 9 250 ;
- durée : 2 500,79 secondes, soit environ 41 min 41 s.

### 10.3 Résultats

| Split | Tokens évalués | Loss en nats | Perplexité |
|---|---:|---:|---:|
| Validation | 183 040 | 1,2184 | 3,3816 |
| Test gelé | 72 192 | 1,1247 | 3,0791 |

SHA-256 du meilleur checkpoint : `ca1dfc4639f2a9887a7dadaaddeac3ffbd28da85f279039cb5093a80b56374f4`

Ces nombres ne doivent pas être comparés directement à ceux du Transformer v0.1 : le corpus, le vocabulaire et surtout les splits ont changé. Une comparaison scientifique directe exigerait de réentraîner les modèles sur les mêmes données.

### 10.4 Ce que la génération a montré

Le modèle produisait mieux la forme des exercices et des phrases. Cependant, il pouvait :

- continuer un patron mémorisé au lieu de répondre à la question ;
- fabriquer des nombres ou des personnes ;
- mélanger des fragments WDI, dictionnaire et mathématiques ;
- échouer sur une opération simple formulée naturellement ;
- donner une phrase grammaticalement plausible mais fausse.

L'exemple `Qui est Gnakale Hacker ?` illustre une limite normale : la personne et une réponse de référence n'existent pas nécessairement dans le corpus. Le modèle génère alors la suite statistiquement probable au lieu de reconnaître honnêtement qu'il ne sait pas.

## 11. Benchmark mathématique : l'épreuve qui a révélé la limite

Un benchmark gelé de 1 000 exercices a été construit, avec 100 exercices dans chacune de dix familles. Le décodage était déterministe et les réponses finales étaient contrôlées exactement.

### 11.1 Transformer 5M avant SFT

- réponses au bon format : 400/1 000, soit 40 % ;
- réponses exactes : 0/1 000, soit 0 % ;
- durée : environ 928,39 secondes.

Le modèle pouvait imiter la présentation d'une solution, mais ne calculait pas correctement.

### 11.2 Données SFT mathématiques

Un jeu supervisé distinct a été construit :

- 50 000 exercices uniques et vérifiés ;
- 48 052 exemples train ;
- 1 948 exemples validation ;
- 10 familles et 4 niveaux de difficulté ;
- zéro chevauchement avec les benchmarks ;
- séquences limitées à 256 caractères.

Le meilleur checkpoint SFT a été obtenu à l'étape 3 000 après environ 1 014,97 secondes. Sa loss cible de validation était 0,1619, soit une perplexité de 1,1757.

SHA-256 du checkpoint SFT : `b542d40be5e13bc39560a273aaece1746b330c5d6f464dc7559432583921cb02`

### 11.3 Résultat après SFT

| Évaluation | Format valide | Réponses exactes |
|---|---:|---:|
| Benchmark étendu | 1 000/1 000 | 2/1 000, soit 0,2 % |
| Diagnostic proche du SFT | 1 000/1 000 | 3/1 000, soit 0,3 % |

Les niveaux de difficulté 3 et 4 ont obtenu zéro réponse exacte dans le diagnostic.

### 11.4 Interprétation honnête

Le SFT a très bien appris **le format** `Problème / Méthode / Solution / Réponse`. Il n'a pas appris un algorithme général de calcul. La loss supervisée très basse reflétait en grande partie la capacité à reproduire des patrons vus, pas à généraliser les opérations.

**Décision :** arrêter de dépenser du GPU sur le même objectif et tester une architecture hybride.

## 12. Moteur mathématique déterministe et assistant hybride

### 12.1 Pourquoi un outil déterministe

Un Transformer choisit des tokens probables. Un calculateur applique des règles et peut garantir le résultat. Le prototype hybride utilise donc trois routes :

```text
Question utilisateur
        ↓
Routeur
 ├─ exercice reconnu → moteur mathématique → réponse vérifiée
 ├─ math non couverte → refus contrôlé → réponse vérifiée
 └─ texte général → Transformer 5M → génération non vérifiée
```

### 12.2 Résultats du moteur

Le moteur couvre dix familles d'exercices. Il lit l'énoncé et ne consulte pas la réponse de référence.

| Benchmark | Exactitude | Erreurs de routage |
|---|---:|---:|
| Développement principal | 1 000/1 000 | 0 |
| Diagnostic SFT | 1 000/1 000 | 0 |
| Test final scellé | 1 000/1 000 | 0 |

Le test final a duré environ 0,056 seconde. Son SHA-256 est `b2ac4391d8d352280c783e0a8a51152853c1d7a91001c76a6b4831ac7f574c7e`.

Ce 100 % est la performance du **moteur de règles sur les patrons couverts**, pas celle du Transformer. Il ne faut pas dire « le modèle 5M sait toutes les maths ».

### 12.3 Correction du langage naturel

Lors d'un test humain, la question `combien font 2+2` était initialement envoyée au Transformer. La cause était un routeur trop dépendant des formulations des exemples. Une analyse arithmétique sûre par arbre syntaxique et de nouveaux motifs d'intention ont été ajoutés.

Après correction :

- 70 tests automatisés passaient ;
- le benchmark de non-régression restait à 1 000/1 000 ;
- les expressions naturelles sûres étaient routées vers le calcul exact.

Cette correction améliore l'**inférence hybride**, pas les poids du Transformer.

### 12.4 Interface Gradio

Une petite interface publique temporaire permet de saisir une question et d'afficher :

- la réponse ;
- la route choisie ;
- le statut vérifié ou non vérifié ;
- une explication du statut.

Les liens Gradio temporaires ne constituent pas un déploiement permanent : ils expirent lorsque la session s'arrête. L'archive durable de l'application est stockée dans :

`gs://legbairai-ivoireslm-artifacts-20260822/inference/hybrid_assistant_v0.1/hybrid_assistant_v0.1.tar.gz`

SHA-256 de l'archive après correction du routeur : `99887a8f4afd2b6958a19f124e86639dadd2b2a11b37312c5f1e562d8c4581fd`

## 13. Tableau comparatif des modèles

| Modèle | Paramètres | Contexte | Données | PPL test | Conclusion |
|---|---:|---:|---|---:|---|
| Bigram-CI v0.1 | 1 304 164 transitions | 1 caractère | v0.5 | 15,5437 | Baseline minimale |
| MLP-CI v0.1 | 249 526 | 16 caractères | v0.5 | 16,3381 | Ne généralise pas mieux |
| Transformer v0.1 | 1 100 032 | 128 caractères | v0.5 | 14,3929 | Bat les baselines |
| Transformer v0.2 5M | 4 758 144 | 256 caractères | v0.6 | 3,0791 | Meilleure modélisation sur nouveaux splits |
| Transformer 5M + SFT | 4 758 144 | 256 caractères | v0.6 + SFT | maths : 0,2–0,3 % exact | Apprend surtout le format |

Les perplexités v0.1 et v0.2 ne sont pas directement comparables, car les données et splits diffèrent.

## 14. Pourquoi le modèle paraît encore « très petit »

Les observations des tests humains sont cohérentes avec l'architecture :

1. **4,76M de paramètres est minuscule pour un assistant général.** Les assistants modernes possèdent généralement beaucoup plus de capacité et bénéficient d'un long préentraînement.
2. **Le corpus v0.6 contient environ 2,2 millions de mots.** C'est utile pour une expérience, mais insuffisant pour couvrir le monde, la conversation et des connaissances rares.
3. **Le tokenizer caractère est exact mais peu efficace sémantiquement.** Le mot est fragmenté en caractères et la fenêtre de 256 se remplit vite.
4. **L'objectif principal était la prédiction du caractère suivant.** Il n'enseigne pas directement à répondre, refuser, citer ou dialoguer.
5. **Le corpus reste en partie structuré.** Le modèle apprend des patrons WDI, dictionnaire et exercices.
6. **Les faits privés ou absents du corpus sont inconnus.** Il n'existe aucun mécanisme interne garantissant « je ne sais pas ».
7. **Un nom ou une identité n'apparaît pas magiquement à l'inférence.** Le modèle connaît son nom seulement si cette identité a été apprise dans les données/SFT, injectée dans le prompt système ou imposée par l'application.

## 15. Les leçons scientifiques du parcours

### Leçon 1 — Séparer les données, le modèle et l'inférence

- Le corpus détermine ce que le modèle peut observer.
- L'entraînement modifie les poids du modèle.
- L'inférence choisit comment utiliser ces poids et peut appeler des outils externes.
- Une règle ajoutée au routeur n'est pas une nouvelle connaissance apprise par le Transformer.

### Leçon 2 — Une métrique isolée peut tromper

La perplexité mesure la prédiction séquentielle. Le benchmark mathématique mesure l'exactitude. Les tests humains mesurent l'utilité perçue. Aucun de ces indicateurs ne remplace les autres.

### Leçon 3 — L'échec du MLP était utile

Il a montré le décalage entre validation et test et a empêché de conclure trop vite que davantage de contexte fixe suffisait.

### Leçon 4 — Un format correct peut cacher une réponse fausse

Le SFT mathématique est passé à 100 % de format valide, mais seulement 0,2 à 0,3 % de réponses exactes. La présentation ne doit jamais être confondue avec le raisonnement.

### Leçon 5 — Les benchmarks doivent être gelés

Un test utilisé pendant les réglages cesse d'être une mesure neutre. C'est pourquoi le projet utilise un test final scellé et enregistre son empreinte.

### Leçon 6 — Un 100 % doit être expliqué

Le moteur déterministe obtient 100 % parce que le benchmark couvre des familles formalisées que le moteur sait analyser. Le score prouve la correction dans ce périmètre, pas une capacité générale.

### Leçon 7 — La diversité vaut mieux que la répétition

Réduire WDI et plafonner le Wiktionnaire a diminué le volume de la v0.6, mais a rendu le corpus plus équilibré. Un modèle entraîné sur trop de gabarits les répète ensuite.

### Leçon 8 — Toujours conserver les artefacts hors de Colab

Le disque `/content` de Colab est temporaire. Les corpus, checkpoints, rapports et notebooks importants doivent être copiés dans Cloud Storage. Une déconnexion Colab ne doit pas détruire l'expérience.

## 16. Questions possibles et réponses courtes

### Pourquoi avoir commencé par un Bigram ?

Pour disposer d'une baseline simple, rapide et reproductible. Sans baseline, on ne peut pas prouver qu'un modèle plus complexe apporte une amélioration.

### Pourquoi le MLP a-t-il échoué malgré une bonne validation ?

Il a surtout appris les patrons répétitifs du train. Le test contenait d'autres groupes de sources, ce qui a révélé une généralisation insuffisante.

### Qu'a apporté l'attention ?

Le Transformer v0.1 peut pondérer plusieurs positions du contexte et a obtenu une meilleure perplexité test que le Bigram et le MLP.

### Le Transformer 5M est-il vraiment un modèle de cinq millions ?

Oui comme catégorie arrondie. Il possède exactement 4 758 144 paramètres.

### Pourquoi sa perplexité est-elle bonne alors que ses réponses sont mauvaises ?

Parce que la perplexité mesure la probabilité du prochain caractère sur un corpus. Répondre à une question exige aussi compréhension de l'instruction, rappel fiable, raisonnement, calibration et parfois des outils.

### Pourquoi zéro token inconnu ne suffit-il pas ?

Le tokenizer peut représenter chaque caractère sans que le modèle comprenne chaque mot ou chaque concept.

### Le modèle sait-il calculer après le SFT ?

Très peu. Il a surtout appris le format : 100 % de sorties formatées, mais seulement 0,2 à 0,3 % exactes sur les diagnostics.

### D'où vient alors le 100 % mathématique ?

Du moteur déterministe, pas du Transformer. Le routeur reconnaît les familles couvertes et le moteur calcule la réponse avec des règles testées.

### Si on ajoute son nom dans l'interface, le modèle l'a-t-il appris ?

Non. Une identité ajoutée au prompt ou au routeur existe seulement pendant l'inférence. Pour l'intégrer aux poids, il faut l'inclure dans des données d'entraînement ou de SFT, puis vérifier sa généralisation.

### Peut-on donner le lien Gradio à d'autres personnes ?

Oui tant que la session qui héberge l'interface reste active et que le lien public n'a pas expiré. Il faut présenter clairement le système comme prototype expérimental.

### Pourquoi les amis obtiennent-ils parfois des phrases absurdes ?

Leur question sort souvent des patrons étroits appris. Le modèle 5M a peu de capacité, peu de données conversationnelles, un contexte court et un objectif de prochain caractère. Il complète alors statistiquement au lieu de répondre comme un assistant moderne.

## 17. État honnête au 24 août 2026

### Ce qui est validé

- pipeline de corpus versionné et audité ;
- corpus ivoirien/français/mathématique reproductible ;
- tokenizers caractère avec 0 % d'inconnus sur les splits ;
- Bigram, MLP, Transformer 1,10M et Transformer 4,76M entraînés ;
- checkpoints et rapports sauvegardés ;
- évaluations gelées ;
- diagnostic clair de l'échec mathématique du modèle ;
- moteur déterministe et interface hybride testés ;
- distinction visible entre réponse vérifiée et génération non vérifiée.

### Ce qui n'est pas encore validé

- assistant conversationnel général ;
- connaissance fiable de personnes ou faits absents du corpus ;
- raisonnement mathématique général appris par le Transformer ;
- refus fiable en dehors des règles de l'application ;
- compréhension longue ;
- déploiement public permanent et robuste.

## 18. Prochaine direction recommandée

Le projet peut continuer sur deux pistes distinctes.

### Piste recherche depuis zéro

- remplacer le tokenizer caractère par un BPE ou SentencePiece de 8 000 à 16 000 unités ;
- augmenter le contexte à 512 ou 1 024 tokens ;
- ajouter davantage de français naturel ivoirien autorisé ;
- créer des dialogues, questions-réponses, refus, résumés et tâches instructionnelles ;
- évaluer séparément langage, faits ivoiriens, raisonnement et sécurité ;
- tester un modèle plus grand seulement après validation du nouveau corpus et des baselines.

### Piste produit utile

- partir d'un modèle instruct préentraîné d'environ 0,5 à 1,5 milliard de paramètres ;
- l'adapter par LoRA/QLoRA sur les données ivoiriennes et instructionnelles ;
- conserver le moteur déterministe pour les calculs ;
- ajouter une recherche documentaire pour répondre avec des sources ;
- garder le modèle 5M comme démonstrateur pédagogique et banc d'essai reproductible.

Ces pistes ne s'opposent pas : l'une montre comment un modèle se construit, l'autre vise plus rapidement une expérience utilisateur crédible.

## 19. Registre des principales preuves

| Preuve | Emplacement dans le dépôt |
|---|---|
| Corpus v0.1.0 | `reports/data/CORPUS_V0.1.0_RELEASE.md` |
| Corpus v0.3.0 à v0.6.0 | `reports/data/CORPUS_V0.*_RELEASE.md` |
| Tokenizer v0.1 | `reports/training/TOKENIZER_CHARACTER_V0.1.md` |
| Bigram | `reports/training/BIGRAM_CI_V0.1.md` |
| MLP contextuel | `reports/training/CONTEXTUAL_MLP_CI_V0.1.md` |
| Transformer 1,10M | `reports/training/MICROIVOIRE_TRANSFORMER_V0.1.md` |
| Transformer 4,76M | `reports/training/MICROIVOIRE_TRANSFORMER_V0.2_5M.md` |
| Benchmark et SFT math | `reports/evaluation/MATH_REASONING_V0.1.md` |
| Diagnostic SFT et outil | `reports/evaluation/MATH_SFT_AND_TOOL_V0.1.md` |
| Assistant hybride | `reports/inference/HYBRID_ASSISTANT_V0.1.md` |
| Interface Gradio | `reports/inference/GRADIO_HYBRID_V0.1.md` |
| Correction du routage naturel | `reports/investigations/NATURAL_MATH_ROUTING_V0.1.md` |

## 20. Conclusion

IvoireSLM n'est pas encore un concurrent des grands assistants. En revanche, le projet a déjà démontré toute une chaîne de travail réelle : collecte responsable, nettoyage, audits, tokenisation, baselines, entraînement GPU, checkpoints, évaluation, analyse d'échec, inférence hybride et sauvegarde durable.

Le progrès le plus solide est d'avoir appris à ne pas confondre :

- volume et qualité ;
- vocabulaire couvert et compréhension ;
- perplexité et intelligence ;
- format correct et réponse exacte ;
- routeur d'inférence et connaissance apprise ;
- démonstration expérimentale et produit prêt pour le public.

Cette distinction, appuyée par les rapports, les checkpoints et les empreintes SHA-256, constitue la preuve principale du parcours accompli.
