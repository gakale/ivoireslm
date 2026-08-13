# IvoireSLM corpus v0.5.0

Date de construction : 13 août 2026.

## Résultat

- 36 151 493 caractères-tokens avec le tokenizer caractère de la roadmap.
- 5 605 013 mots approximatifs.
- 184 155 lignes et 210 483 faits atomiques déclarés.
- 19 documents : 15 train, 2 validation et 2 test.
- 9 domaines, avec ajout du domaine `mathematics`.
- 19 documents sur 19 classés `A_REDISTRIBUTABLE`, chacun conservant sa licence et son attribution.

La branche mathématique apporte 3 616 563 caractères, soit **10,0039 %** du corpus final. La version v0.4.0 et tous ses artefacts restent conservés séparément.

## Mathématiques françaises

Sources : Wikilivres en français et complément encyclopédique de Wikipédia en français, figés avec identifiants de révision, URL permanente et historique d'attribution par page.

- 704 pages Wikilivres inspectées ; 353 pages utilisables retenues.
- 236 pages Wikipédia inspectées ; 178 pages complémentaires retenues.
- 2 335 666 caractères Wikilivres et 1 280 897 caractères Wikipédia après assemblage final.
- 13 725 formules LaTeX conservées : 8 088 depuis Wikilivres et 5 637 depuis Wikipédia.
- Couverture Wikipédia retenue : algèbre 28 pages, analyse 30, géométrie 27, logique mathématique 32, probabilités 33 et statistiques 30. Une page peut relever de plusieurs catégories.
- Sections bibliographiques, références, navigation et éléments non imprimables exclus.
- MathML rendu converti en une représentation LaTeX unique ; aucune balise MathML résiduelle.
- 3 550 lignes répétées écartées à la source : 2 840 dans Wikilivres et 710 dans Wikipédia.
- Sélection du complément Wikipédia par classement SHA-256 déterministe jusqu'à la cible de 10 %.

Licence : CC BY-SA 4.0, avec GFDL comme option alternative lorsqu'elle s'applique. Les fichiers d'attribution du paquet associent chaque page à sa révision, son URL permanente et son historique.

## Limites

- La proportion de 10 % est mesurée en caractères, conformément au tokenizer caractère prévu par la roadmap ; elle ne correspondra pas exactement à 10 % avec un tokenizer sous-mots.
- Les textes Wikimedia peuvent contenir des erreurs pédagogiques ou encyclopédiques et ne remplacent pas une validation par un spécialiste.
- Le corpus apporte des explications et des formules, mais pas encore un grand jeu dédié de démonstrations formelles, de problèmes corrigés gradués ou de données mathématiques ivoiriennes.
- Toutes les données mathématiques sont placées dans `train` sous un même `group_id`, afin d'éviter une contamination des évaluations existantes.

## Contrôle qualité final

- 25 tests unitaires réussis.
- Quality gate et audit indépendant réussis.
- Aucun hash source ou d'attribution invalide.
- Aucun document, ligne normalisée ou document quasi dupliqué.
- Aucune fuite de groupe entre train, validation et test.
- Aucune adresse électronique ou numéro de téléphone détecté.
- Aucun marqueur suspect, caractère de contrôle ou balise MathML résiduelle.

## Artefacts serveur

Corpus :

`/home/gnakalehacker/ivoireslm-storage/corpora/ivoireslm_corpus_v0.5.0`

Archive :

`/home/gnakalehacker/ivoireslm-storage/corpora/ivoireslm_corpus_v0.5.0.tar.gz`

SHA-256 :

`6a45c958497b7b64fb5bef09dd1f14187b6c15acb5843556da77c86e33a7e5b1`

Snapshots mathématiques :

`/home/gnakalehacker/ivoireslm-storage/snapshots/wikibooks_math_fr_2026-08-13_v0.3`

`/home/gnakalehacker/ivoireslm-storage/snapshots/wikipedia_math_fr_2026-08-13_v0.1`

## Reproduction depuis les snapshots figés

```bash
cd ~/ivoireslm
source ~/venv/bin/activate
python3 scripts/data/build_wikibooks_math_fr_v01.py
python3 scripts/data/build_wikipedia_math_fr_v01.py
python3 scripts/data/build_corpus_v05.py
python3 scripts/data/audit_corpus_v05.py
```

Pour recréer les snapshots à partir des API Wikimedia avant la construction :

```bash
python3 scripts/data/snapshot_wikibooks_math_fr.py
python3 scripts/data/snapshot_wikipedia_math_fr.py
```

## Prochaine priorité

La cible mathématique est atteinte. La prochaine augmentation doit privilégier le français naturel ivoirien explicitement autorisé, les ressources éducatives locales, la santé publique, les textes administratifs ouverts, le nouchi et les langues ivoiriennes avec consentement et provenance ; la qualité et l'équilibre des domaines doivent rester prioritaires sur le volume brut.
