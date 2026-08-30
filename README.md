# IvoireSLM

IvoireSLM est un projet d'entraînement et d'évaluation de petits modèles de langage adaptés aux contextes ivoiriens.

## Structure

- `configs/` : configurations d'expériences.
- `data/` : données brutes, préparées et tokenisées (non versionnées).
- `src/` : code source du pipeline.
- `notebooks/` : expérimentations pédagogiques.
- `checkpoints/`, `reports/`, `logs/` : sorties d'entraînement et d'évaluation.

Consultez [la feuille de route](IVOIRESLM_ROADMAP.md) pour les prochaines étapes.

Le journal technique [du Bigram au Transformer 5M](docs/IVOIRESLM_PARCOURS_BIGRAM_A_5M.md) rassemble la chronologie, les résultats, les preuves et les leçons apprises.

La conception de la prochaine phase est décrite dans [l'expérience Transformer 17M v0.3](docs/EXPERIENCE_TRANSFORMER_17M_V03.md).

La correction des biais observés est détaillée dans
[l'expérience corpus v0.8](docs/EXPERIENCE_CORPUS_V08.md) : collecte française
élargie, plafonnement des gabarits répétitifs et nouveaux contrôles de
composition.
