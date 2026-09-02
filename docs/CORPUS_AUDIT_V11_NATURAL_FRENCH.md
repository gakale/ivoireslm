# Audit corpus v1.1 — priorité au français naturel

Date : 2026-09-02  
Décision : aucun nouvel entraînement n'est autorisé avec le supplément v1.0 seul.

## Constat mesuré

Le supplément `ivoireslm_corpus_supplement_v1.0.0` contient 108 554 194
caractères, mais sa composition est très déséquilibrée :

| Domaine | Caractères | Part |
|---|---:|---:|
| Raisonnement mathématique | 97 428 589 | 89,75 % |
| Code et agents | 3 722 158 | 3,43 % |
| Anglais naturel | 3 309 385 | 3,05 % |
| Français naturel | 3 110 725 | 2,87 % |
| Cybersécurité défensive | 983 337 | 0,91 % |

Une continuation sur ce lot renforcerait précisément les symptômes observés :
réponses qui mélangent les exercices, formules hors contexte, faible conversation et
connaissances générales fragiles. Le lot reste utile comme réservoir spécialisé,
mais il ne doit pas être concaténé directement au corpus principal.

## Ce qui est réellement exploitable dans Drive

- Wikipedia français et anglais : licence ShareAlike, provenance conservée ; le lot
  actuel ne contient que 500 articles par langue.
- GSM8K train et AQuA train/dev : ouverts, mais à plafonner fortement et à tenir
  hors de toute évaluation finale.
- DeepSeek Harness : documentation MIT, utile en faible proportion.
- CISA KEV : données publiques défensives, utiles en faible proportion.
- Les benchmarks de test, les données non commerciales, les licences inconnues,
  les binaires et les preuves de concept restent exclus.
- Les contenus Facebook/X ne sont pas admis sans consentement ou licence de
  réutilisation explicite ; leur accessibilité publique ne suffit pas.

## Recette candidate v1.1

La recette est décrite dans `configs/corpus_mix_v11_candidate.json`. Elle conserve
65 % du corpus ivoirien v0.9, vise 20 % de nouveau français naturel, 5 % de
conversation ivoirienne vérifiée, 3 % de langues ivoiriennes, 5 % d'anglais et
seulement 2 % de STEM/code/cybersécurité au total.

Avant entraînement, le supplément doit contenir au minimum 30 millions de caractères
de français naturel et 5 millions de caractères de conversation ivoirienne vérifiée.
Les données présentes aujourd'hui ne satisfont pas ces seuils. Le script
`scripts/data/audit_corpus_mix_v11.py` rend ce blocage automatique et reproductible.

## Ordre de travail

1. Inventorier les sources textuelles françaises et ivoiriennes déjà présentes.
2. N'admettre que les documents avec provenance et licence vérifiables.
3. Nettoyer, dédupliquer et séparer les groupes train/validation.
4. Construire le supplément v1.1 et exécuter l'audit automatique.
5. Lancer un CPT court uniquement si toutes les gardes passent.
6. Comparer le CPT au parent sur langue générale, Côte d'Ivoire, répétition et faits.
7. Faire ensuite un SFT court ; garder le test final scellé jusqu'au candidat final.
