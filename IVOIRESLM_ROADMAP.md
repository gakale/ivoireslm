# Feuille de route IvoireSLM

La roadmap maître complète est conservée sur la VM à l'emplacement :

`/home/gnakalehacker/ivoireslm-storage/incoming/mac_2026-08-12/project_context/IVOIRESLM_ROADMAP.md`

## État au 13 août 2026

| Phase | Sujet | État | Preuve |
|---:|---|---|---|
| 0 | Cadrage et souveraineté | 🟩 Validé | Roadmap maître |
| 1 | Organisation et reproductibilité | 🟩 Validé | Dépôt, stockage séparé et scripts versionnés |
| 2 | Inventaire, provenance et droits | 🟨 En cours global / validé pour v0.1.0 | Manifestes officiel et quarantaine |
| 3 | Extraction et originaux | 🟨 En cours global / validé pour les 13 sources | Snapshots et sorties factuelles traçables |
| 4 | Nettoyage et déduplication | 🟩 Validé pour v0.1.0 | `quality_report.json` |
| 5 | Analyse et splits | 🟩 Validé pour v0.1.0 | Splits par groupe, dataset card et audit |
| 6 | Tokenizer caractère | 🟦 Prochaine étape | À exécuter sur le split train uniquement |
| 7 | Bigram-CI | 🟦 Préparé | À exécuter après validation du tokenizer |

## Corpus officiel actuel

Version : `ivoireslm_corpus_v0.1.0`  
Documents : 13  
Phrases : 34 482  
Faits atomiques : 46 791  
Quality gate : réussi  

Voir [le rapport de livraison](reports/data/CORPUS_V0.1.0_RELEASE.md).

## Prochaine action

Entraîner et évaluer le tokenizer caractère de référence uniquement sur `train.txt`. Le split `test` reste gelé. En parallèle, poursuivre l'acquisition de textes naturels ivoiriens dont les droits sont explicitement compatibles.
