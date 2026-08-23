# Feuille de route IvoireSLM

La roadmap maître complète est conservée sur la VM à l'emplacement :

`/home/gnakalehacker/ivoireslm-storage/incoming/mac_2026-08-12/project_context/IVOIRESLM_ROADMAP.md`

## État au 20 août 2026

| Phase | Sujet | État | Preuve |
|---:|---|---|---|
| 0 | Cadrage et souveraineté | 🟩 Validé | Roadmap maître |
| 1 | Organisation et reproductibilité | 🟩 Validé | Dépôt, stockage séparé et scripts versionnés |
| 2 | Inventaire, provenance et droits | 🟨 En cours global / validé pour v0.1.0 | Manifestes officiel et quarantaine |
| 3 | Extraction et originaux | 🟨 En cours global / validé pour les 13 sources | Snapshots et sorties factuelles traçables |
| 4 | Nettoyage et déduplication | 🟩 Validé pour v0.1.0 | `quality_report.json` |
| 5 | Analyse et splits | 🟩 Validé pour v0.1.0 | Splits par groupe, dataset card et audit |
| 6 | Tokenizer caractère | 🟩 Validé | 1 142 tokens, round-trip et couverture validés sur v0.5.0 |
| 7 | Bigram-CI | 🟩 Validé | Loss validation 2,4735, checkpoint et rapport enregistrés |
| 8 | Embeddings et réseau contextuel | 🟩 Expérience validée | MLP CPU reproductible ; PPL test 16,3381, baseline conservée car Bigram = 15,5437 |
| 9 | MicroIvoire Transformer v0.1 | 🟩 Validé | 1 100 032 paramètres ; PPL test 14,3929 ; reprise Cloud Storage vérifiée |
| 10 | Corpus v0.6 et tokenizer v0.2 | 🟩 Validé | Corpus rééquilibré, mathématiques vérifiées, nouveau test gelé et 0 % UNK |
| 11 | MicroIvoire Transformer v0.2 5M | 🟩 Validé | 4 758 144 paramètres ; meilleur pas 9 250 ; PPL validation 3,3816 et test gelé 3,0791 |

## Corpus officiel actuel

Version : `ivoireslm_corpus_v0.6.0`  
Documents : 25  
Caractères : 14 316 283, dont 11,3382 % de mathématiques vérifiées et 4,0867 % de français ivoirien naturel ajouté  
Lignes : 83 983  
Tokenizer : `ivoireslm_character_v0.2`, vocabulaire 722, 0 % UNK  
Quality gate : réussi  

Voir [le rapport de livraison](reports/data/CORPUS_V0.6.0_RELEASE.md).

## Prochaine action

Construire une évaluation automatique dédiée aux mathématiques, entraîner les baselines sur les mêmes splits v0.2 et créer un jeu de corrections humaines. La perplexité globale ne suffit pas à prouver que les réponses factuelles ou mathématiques sont correctes.
