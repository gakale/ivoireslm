---
language:
- fr
- en
library_name: pytorch
license: other
tags:
- ivoire
- cote-divoire
- french
- causal-lm
- research
- experimental
---

# MicroIvoire Transformer v1.0 17M CPT Pilot

Checkpoint expérimental d'un Transformer causal dense de **17 129 280 paramètres**, développé dans le cadre du projet [IvoireSLM](https://github.com/gakale/ivoireslm).

> **Ce modèle n'est pas encore un assistant conversationnel fiable.** Il peut produire du texte répétitif, incohérent ou factuellement faux. Il ne doit pas être utilisé pour prendre une décision médicale, juridique, financière, scolaire ou de sécurité.

## Ce qui est publié

Cette version est un pilote de continuation de préentraînement de 1 000 étapes à partir du meilleur checkpoint IvoireSLM v0.4 17M. Le checkpoint est publié pour la recherche, la reproductibilité et l'apprentissage, pas comme produit fini.

- architecture : Transformer causal dense, RMSNorm, RoPE et SwiGLU ;
- vocabulaire : BPE de 8 192 tokens ;
- contexte maximal : 512 tokens ;
- dimension d'embedding : 320 ;
- têtes d'attention : 8 ;
- blocs Transformer : 12 ;
- dimension feed-forward : 832 ;
- poids d'embedding et de sortie liés ;
- paramètres : 17 129 280.

## Mélange du pilote

| Domaine | Part |
|---|---:|
| Corpus général IvoireSLM v0.9 | 60 % |
| Wikipédia français | 15 % |
| Wikipédia anglais | 5 % |
| Raisonnement mathématique | 10 % |
| Code et agents | 5 % |
| Cybersécurité défensive | 5 % |

Le test est resté scellé. Les chiffres ci-dessous proviennent uniquement des validations séparées par domaine.

## Résultats

| Domaine | Avant | Étape 1 000 | Évolution |
|---|---:|---:|---:|
| Loss pondérée | 3,1619 | 2,8577 | -0,3042 |
| Corpus général v0.9 | 2,8881 | 2,8951 | +0,0070 |
| Wikipédia français | 3,0599 | 2,9059 | -0,1540 |
| Wikipédia anglais | 4,0119 | 3,6850 | -0,3268 |
| Raisonnement mathématique | 3,8734 | 2,6384 | -1,2351 |
| Code et agents | 4,1901 | 3,3140 | -0,8761 |
| Cybersécurité défensive | 3,4523 | 1,4177 | -2,0345 |

La garde de rétention du corpus général autorisait une hausse maximale de `0,03`. La hausse observée est de `0,0070`.

## Limites constatées

Le diagnostic qualitatif déterministe à l'étape 1 000 montre encore :

- des répétitions importantes ;
- des affirmations géographiques ou factuelles inventées ;
- un mauvais suivi des questions simples ;
- des sorties de code incorrectes ;
- une baisse de loss qui ne suffit pas à garantir une meilleure expérience utilisateur.

Le modèle ne doit donc pas être présenté comme un chatbot général ni comme une source de connaissances. Une interface peut l'utiliser uniquement comme démonstration expérimentale, avec un avertissement visible et, pour les tâches calculables, un moteur déterministe séparé.

## Fichiers attendus

- `best.pt` : poids du meilleur checkpoint de validation, sans état d'optimiseur ;
- `tokenizer.json` : tokenizer BPE v0.4 ;
- `config.json` : architecture et provenance du checkpoint ;
- `modeling_microivoire.py` : définition PyTorch du modèle ;
- `generate.py` : exemple minimal de chargement et génération ;
- `progress.json` : métriques finales du pilote ;
- `QUALITATIVE_DIAGNOSTIC_STEP1000.json` : sorties de contrôle ;
- `SHA256SUMS` : empreintes des artefacts publiés.

Le format du checkpoint est un dictionnaire PyTorch personnalisé, pas un modèle `transformers.AutoModelForCausalLM`.

## Utilisation responsable

Vérifiez systématiquement les sorties. N'utilisez pas ce checkpoint pour imiter une personne, produire de la désinformation, automatiser des décisions à fort impact ou générer du code exécuté sans revue humaine.

## Reproductibilité

Le code d'entraînement, les contrôles, les décisions et les scripts de préparation sont versionnés dans le dépôt IvoireSLM. La [décision scientifique complète](https://github.com/gakale/ivoireslm/blob/main/docs/DECISION_CPT_V10_STEP1000.md) documente le résultat.

Empreintes connues avant assemblage final :

- `best.pt` : `3dbd076f7df86fedad4f4b47a657f042dd156de05d8b37d9c46319811fde7072` ;
- `latest.pt` : `c652acc835bce9d54994f7dbac083fcaa03e9540ff38b411017b6fb99b8d21ba` ;
- diagnostic qualitatif : `032fe4cc986b20b1f22dff436a1d773842252015db1aaf8770d59c990dcca0a9`.

## Licence et données

Le code du dépôt est sous licence MIT. La licence du checkpoint est indiquée comme `other` tant que l'inventaire complet des licences et conditions de redistribution des données du mélange n'a pas été finalisé. Ne réutilisez pas les poids comme si toutes les données étaient sous licence MIT.
