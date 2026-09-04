# IvoireSLM

IvoireSLM est un projet de recherche consacré aux petits modèles de langage adaptés au français et aux contextes ivoiriens. Le dépôt contient toute la chaîne de travail : collecte et audit des données, entraînement, évaluation, inférence et interface de démonstration.

Le modèle actuel compte **17 129 280 paramètres**. Il peut être utilisé seul pour observer ce que les poids ont réellement appris, ou derrière un assistant outillé qui confie les réponses sensibles à des composants déterministes et à des sources identifiées.

## État actuel

Le checkpoint de référence est le CPT v1.1.1 à l’étape 500 :

| Élément | Valeur |
|---|---|
| Identifiant | `microivoire_transformer_v1.4_17m_cpt_v111_pilot` |
| Paramètres | 17 129 280 |
| Tokenizer | BPE v0.4, vocabulaire de 8 192 tokens |
| Fenêtre de contexte | 512 tokens |
| Corpus de continuation | `ivoireslm_pretraining_mix_v1.1.1_pilot` |
| Étape sélectionnée | 500 |
| Test final | toujours fermé |

Ce checkpoint améliore les validations de français naturel, de conversation et de données ivoiriennes, tout en conservant la qualité mesurée sur le corpus historique. Cela ne suffit pas à en faire un bon chatbot : en génération libre, il répète encore certaines phrases, suit mal des questions simples et peut inventer des faits.

Il est donc présenté comme un **checkpoint expérimental**, pas comme un assistant généraliste.

## Architecture du modèle 17M

IvoireSLM 17M est un Transformer causal dense écrit en PyTorch. Il prédit le token suivant à partir des tokens précédents.

| Composant | Configuration actuelle |
|---|---:|
| Blocs Transformer | 12 |
| Dimension des embeddings | 320 |
| Têtes d’attention | 8 |
| Dimension par tête | 40 |
| Dimension du réseau feed-forward | 832 |
| Vocabulaire | 8 192 tokens |
| Contexte maximal | 512 tokens |
| Normalisation | RMSNorm |
| Encodage des positions | RoPE |
| Activation du feed-forward | SwiGLU |
| Dropout | 0,10 |
| Embedding et tête de sortie | poids partagés |

Chaque bloc applique deux résidus :

```text
tokens
  │
  ▼
embedding partagé (8 192 × 320)
  │
  ▼
12 × [RMSNorm → attention causale + RoPE → résidu
      RMSNorm → SwiGLU (320 → 832 → 320) → résidu]
  │
  ▼
RMSNorm finale
  │
  ▼
projection vers les 8 192 tokens
```

L’attention utilise l’implémentation `scaled_dot_product_attention` de PyTorch. Le modèle n’est ni un MoE ni une adaptation d’un modèle téléchargé : son architecture et sa boucle d’entraînement sont maintenues dans ce dépôt.

Le code principal se trouve dans [`scripts/training/train_transformer_ci_v03_17m.py`](scripts/training/train_transformer_ci_v03_17m.py). La configuration v0.4 utilisée comme base est définie dans [`scripts/training/train_transformer_ci_v04_17m.py`](scripts/training/train_transformer_ci_v04_17m.py).

## Entraînement actuel

Le checkpoint v1.1.1 prolonge le modèle v0.4 avec un faible taux d’apprentissage. Le mélange reste volontairement majoritaire en données historiques afin de limiter l’oubli.

| Domaine | Part du mélange |
|---|---:|
| Corpus IvoireSLM v0.9 | 70 % |
| Français naturel ouvert | 20 % |
| Conversation française ouverte | 5 % |
| Données ivoiriennes officielles et vérifiées | 5 % |

Résultats au palier 500 :

| Validation | Avant CPT | Étape 500 |
|---|---:|---:|
| Loss pondérée | 2,8930 | 2,8593 |
| Corpus général v0.9 | 2,8881 | 2,8867 |
| Français naturel | 3,0028 | 2,9581 |
| Conversation française | 3,2449 | 3,0027 |
| Données ivoiriennes vérifiées | 2,1713 | 1,9380 |

Le détail du protocole est disponible dans [`docs/CPT_V111_17M_PILOT.md`](docs/CPT_V111_17M_PILOT.md).

## Modèle seul et assistant outillé

L’interface expose deux modes distincts.

### Modèle seul

La question est envoyée directement au Transformer 17M. Aucun calculateur, aucune base documentaire et aucune recherche Internet ne corrigent sa sortie. Ce mode sert à évaluer honnêtement le checkpoint. Les réponses peuvent être incomplètes, répétitives ou fausses.

### Assistant outillé

Un routeur examine d’abord la demande, puis choisit le composant adapté :

- calculateur déterministe pour les opérations prises en charge ;
- fiche locale pour l’identité, la version et les capacités du système ;
- base vérifiée pour quelques faits institutionnels ivoiriens ;
- petit lexique dioula validé ;
- garde de fraîcheur pour les demandes qui exigent une information actuelle ;
- recherche Wikipédia en français lorsque l’utilisateur autorise Internet ;
- modèle 17M uniquement lorsqu’aucune route plus fiable ne convient.

L’interface affiche la route réellement utilisée, le niveau de confiance et les sources disponibles. Une réponse issue d’un outil ne prouve pas que le modèle 17M connaissait lui-même l’information.

La conception complète de cette séparation est décrite dans [`docs/TOOL_ASSISTANT_17M_V1.md`](docs/TOOL_ASSISTANT_17M_V1.md).

## Lancer la démonstration

Le lancement Colab de l’assistant outillé utilise :

```bash
python -u scripts/colab/run_tool_assistant_v1.py
```

Le script vérifie les empreintes du checkpoint et de l’archive contenant le tokenizer avant de démarrer l’interface Gradio.

Pour travailler directement dans le dépôt :

```bash
python -m pip install torch tokenizers gradio
python scripts/inference/gradio_tool_assistant_v1.py \
  --checkpoint /chemin/vers/best.pt \
  --data-dir /chemin/vers/bpe_v0.4 \
  --model-script scripts/training/train_transformer_ci_v04_17m.py \
  --feedback-file /chemin/vers/feedback.jsonl
```

## Publication du checkpoint

Un candidat Hugging Face est préparé dans [`releases/huggingface/ivoireslm-17m-cpt-v1.1.1-step500`](releases/huggingface/ivoireslm-17m-cpt-v1.1.1-step500). Les poids sont convertis en `safetensors` et le dépôt est créé en privé par défaut.

La publication publique reste bloquée tant que les conditions suivantes ne sont pas remplies :

- inventaire complet des licences des données ;
- au moins 90 % de réponses sans boucle répétitive ;
- réussite des seuils d’identité, de prudence, de compréhension et de suivi de consignes ;
- au moins 24 réponses acceptables sur 30 pendant le test humain libre.

Les seuils exacts sont définis dans [`docs/ASSISTANT_17M_QUALIFICATION_V1.md`](docs/ASSISTANT_17M_QUALIFICATION_V1.md).

## Organisation du dépôt

```text
configs/               configurations des expériences
data/                  données locales non versionnées
docs/                  protocoles, audits et décisions
releases/huggingface/  fichiers préparés pour les publications
scripts/colab/         lanceurs destinés à Google Colab
scripts/data/          collecte, nettoyage et construction des corpus
scripts/evaluation/    évaluations automatiques et benchmarks
scripts/inference/     génération et interfaces Gradio
scripts/tokenizer/     entraînement et audit des tokenizers
scripts/training/      architectures et boucles d’entraînement
```

## Limites

IvoireSLM 17M reste un petit modèle de recherche. Il ne doit pas être utilisé comme source unique d’information ni pour prendre des décisions médicales, juridiques, financières, administratives ou de sécurité. Toute sortie du modèle brut doit être vérifiée.

Le code du projet est distribué sous licence MIT. La licence des poids reste distincte tant que l’inventaire des données n’est pas terminé.
