# IvoireSLM corpus v0.4.0

Date de construction : 13 août 2026.

## Résultat

- 32 534 930 caractères-tokens avec le tokenizer caractère de la roadmap.
- 5 001 595 mots approximatifs.
- 157 818 lignes et 184 677 faits atomiques déclarés.
- 17 documents : 13 train, 2 validation et 2 test.
- 8 domaines, avec ajout de la lexicographie et de la documentation technique.
- 17 documents sur 17 classés `A_REDISTRIBUTABLE`, chacun conservant sa licence propre.

Le noyau ivoirien factuel v0.3 représente 17 192 011 caractères, soit 52,84 % du nouveau corpus. Les deux ressources françaises ouvertes apportent 15 342 919 caractères, soit 47,16 %. Cette répartition diversifie fortement le français sans effacer le noyau ivoirien.

## Dictionnaire français

Source : Wiktionnaire français, extraction structurée Wiktextract/kaikki.org du 11 août 2026 à partir du dump Wikimedia du 4 août 2026.

- 2 108 619 entrées source inspectées.
- 48 690 lemmes retenus par échantillonnage SHA-256 déterministe et quotas grammaticaux.
- 63 240 sens définis.
- 12 086 605 caractères et 1 722 115 mots.
- Simples flexions, renvois, citations et exemples exclus.
- Deux rendus identiques exclus avant assemblage.
- Marques d'usage conservées lorsqu'elles sont fournies par la source.

Licence : CC BY-SA 4.0, avec GFDL comme option alternative. Le paquet contient `attributions/frwiktionary_definitions_v0.1.jsonl`, qui associe chaque lemme à son article et à son historique de contributeurs.

Le dictionnaire peut contenir du vocabulaire historique, sensible, péjoratif ou offensant. Les marques d'usage sont conservées lorsqu'elles existent, mais elles ne sont pas garanties pour toutes les entrées. Cette branche doit être évaluée spécifiquement avant entraînement d'un modèle destiné au public.

## Documentation technique française

Source : traduction officielle française de la documentation Python 3.14, commit `a02a5710f906eb93fae4ce3a0a0cbed91243c5d9`.

- 522 catalogues de traduction inspectés.
- 17 258 blocs substantiels retenus après nettoyage du balisage reStructuredText.
- 3 256 314 caractères et 497 501 mots.
- Blocs contenant des adresses électroniques exclus.
- Aucun balisage de rôle ou directive reStructuredText résiduel détecté lors de l'inspection finale.

Licence : Python Software Foundation License Version 2 ; contributions de traduction sous CC0 1.0.

## Contrôle qualité final

- 21 tests unitaires réussis.
- Quality gate et audit indépendant réussis.
- Aucun hash source ou d'attribution invalide.
- Aucun document, ligne normalisée ou document quasi dupliqué.
- Aucune fuite de groupe entre train, validation et test.
- Aucune adresse électronique ou numéro de téléphone détecté.
- Aucun marqueur suspect ni caractère de contrôle.

## Artefacts serveur

Corpus :

`/home/gnakalehacker/ivoireslm-storage/corpora/ivoireslm_corpus_v0.4.0`

Archive :

`/home/gnakalehacker/ivoireslm-storage/corpora/ivoireslm_corpus_v0.4.0.tar.gz`

SHA-256 :

`019a8ccb61441a3c826dae2fa468baaad5ee6d698a1596352777f442269fe113`

Snapshot des sources françaises :

`/home/gnakalehacker/ivoireslm-storage/snapshots/french_open_2026-08-13_v0.1`

## Reproduction

```bash
cd ~/ivoireslm
source ~/venv/bin/activate
python3 scripts/data/snapshot_french_open_sources.py
python3 scripts/data/build_python_docs_fr_v01.py
python3 scripts/data/build_wiktionary_fr_v01.py
python3 scripts/data/build_corpus_v04.py
python3 scripts/data/audit_corpus_v04.py
```

## Prochaine priorité

Le corpus possède maintenant davantage de français général et technique. Le manque principal reste le français naturel ivoirien explicitement autorisé : textes administratifs ouverts, supports éducatifs, santé publique, médias sous licence claire, littérature du domaine public ou autorisée, nouchi et langues ivoiriennes documentées avec consentement et provenance.
