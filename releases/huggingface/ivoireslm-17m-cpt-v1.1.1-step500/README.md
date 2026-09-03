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

# IvoireSLM 17M — CPT v1.1.1, étape 500

Checkpoint expérimental d’un Transformer causal dense de **17 129 280 paramètres**, développé dans le cadre du projet [IvoireSLM](https://github.com/gakale/ivoireslm).

> **Ce checkpoint n’est pas un assistant conversationnel fiable.** Les essais qualitatifs montrent encore des répétitions, des échos de question et des réponses incorrectes. Il est publié pour la recherche et la reproductibilité, pas pour fournir des informations fiables ni prendre des décisions.

## Version publiée

- identifiant interne : `microivoire_transformer_v1.4_17m_cpt_v111_pilot` ;
- mélange : `ivoireslm_pretraining_mix_v1.1.1_pilot` ;
- étape sélectionnée : 500 ;
- tokenizer : BPE v0.4, 8 192 tokens ;
- contexte maximal : 512 tokens ;
- dimension : 320 ;
- têtes d’attention : 8 ;
- blocs Transformer : 12 ;
- dimension feed-forward : 832 ;
- architecture : RMSNorm, RoPE, SwiGLU et poids d’embedding/sortie liés ;
- test final ouvert : non.

## Mélange de continuation de préentraînement

| Domaine | Part |
|---|---:|
| Corpus IvoireSLM v0.9 | 70 % |
| Français naturel ouvert | 20 % |
| Conversation française ouverte | 5 % |
| Données ivoiriennes officielles ancrées | 5 % |

## Résultats de validation

| Domaine | Avant | Étape 500 | Évolution |
|---|---:|---:|---:|
| Loss pondérée | 2,8930 | 2,8593 | -0,0337 |
| Corpus général v0.9 | 2,8881 | 2,8867 | -0,0014 |
| Français naturel ouvert | 3,0028 | 2,9581 | -0,0447 |
| Conversation française ouverte | 3,2449 | 3,0027 | -0,2422 |
| Données ivoiriennes vérifiées | 2,1713 | 1,9380 | -0,2333 |

La garde de rétention du corpus général est passée. Une baisse de loss ne démontre toutefois pas qu’un modèle répond correctement en conversation.

## Limites constatées

- boucles répétitives et échos du prompt ;
- suivi fragile des questions et consignes ;
- faits susceptibles d’être inventés ;
- absence de connaissance fiable de sa propre identité ou version ;
- absence de garantie sur les informations récentes ;
- français, dioula et autres langues encore très limités.

L’interface « assistant outillé » du projet masque certaines de ces limites grâce à des routes déterministes et des bases vérifiées. Ces capacités appartiennent à l’application et **ne sont pas contenues dans ce checkpoint**.

## Utilisation

Le dépôt utilise une architecture PyTorch personnalisée, pas `transformers.AutoModelForCausalLM`.

```bash
pip install torch tokenizers safetensors
python generate.py "La Côte d’Ivoire est" --max-new-tokens 48
```

Chaque sortie doit être considérée comme non vérifiée.

## Fichiers

- `model.safetensors` : poids seuls dans un format sans pickle ;
- `tokenizer.json` : tokenizer BPE v0.4 ;
- `config.json` : architecture et provenance ;
- `modeling_microivoire.py` : définition PyTorch ;
- `generate.py` : exemple minimal d’inférence ;
- `progress.json` : métriques du pilote ;
- `checkpoint_metadata.json` : provenance du checkpoint converti ;
- `SHA256SUMS` : empreintes des fichiers publiés.

## Utilisation responsable

N’utilisez pas ce modèle pour des décisions médicales, juridiques, financières, scolaires, administratives ou de sécurité. Ne l’utilisez pas pour imiter une personne, produire de la désinformation ou exécuter automatiquement du code généré.

## Licence et données

Le code IvoireSLM est sous licence MIT. La licence des poids reste `other` tant que l’inventaire des licences et des conditions de redistribution de toutes les données du mélange n’est pas finalisé. La publication de ce candidat doit donc rester privée jusqu’à validation de cet inventaire.

