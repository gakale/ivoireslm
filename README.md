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

## État du modèle 17M

Le pilote de diversification v1.0 a amélioré les losses de validation en
mathématiques, code, anglais et cybersécurité défensive, sans dépasser la garde
de rétention du corpus général. Les générations restent néanmoins trop
répétitives et factuellement fragiles pour présenter ce checkpoint comme un
assistant conversationnel fiable.

- [Décision scientifique à l'étape 1 000](docs/DECISION_CPT_V10_STEP1000.md)
- [Protocole de continuation de préentraînement](docs/CONTINUATION_PRETRAINING_V10_17M.md)
- [Fiche de publication Hugging Face](releases/huggingface/microivoire-transformer-v1.0-17m-cpt-pilot/README.md)

La publication des poids est désormais suspendue pendant la qualification du
micro-assistant. Le protocole, les seuils et la séparation entre le modèle seul
et le moteur hybride sont décrits dans
[la qualification du micro-assistant 17M](docs/ASSISTANT_17M_QUALIFICATION_V1.md).
Les poids ne seront publiés qu'après réussite des contrôles automatiques puis
d'un test humain de 30 questions.

Le test humain du stage 3 a révélé un collapse vers quelques réponses fixes.
La correction expérimentale repart du checkpoint CPT v1.0 et est décrite dans
[le stage 4 anti-collapse](docs/ASSISTANT_17M_STAGE4_ANTI_COLLAPSE.md). L'audit
du corpus qui motive ce choix est conservé dans
[l'audit de données du stage 4](docs/CORPUS_AUDIT_ASSISTANT_STAGE4.md).

Le stage 4 a restauré une partie de la diversité sans produire des réponses
assez complètes ni assez justes. La décision à l'étape 125 est consignée dans
[son rapport de décision](docs/ASSISTANT_17M_STAGE4_STEP125_DECISION.md). Le
[stage 5 « réponse directe »](docs/ASSISTANT_17M_STAGE5_DIRECT_ANSWER.md)
introduit un curriculum vérifié, une mesure anti-écho et une garde contre les
réponses d'un seul mot. Il reste expérimental et ne justifie pas encore la
publication des poids.

Le diagnostic de l'étape 250 a ensuite révélé une suppression accidentelle des
opérateurs pendant la déduplication des exercices. La nouvelle branche
[stage 5B ciblée](docs/ASSISTANT_17M_STAGE5B_TARGETED_RECOVERY.md) corrige cette
cause, rééquilibre conversation, savoirs et prudence, puis limite son premier
pilote à 125 étapes.
