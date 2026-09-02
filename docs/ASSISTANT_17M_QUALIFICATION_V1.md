# Qualification du micro-assistant IvoireSLM 17M v1

## Décision

Le modèle 17M doit être rendu utile et mesuré avant toute publication de poids et avant le lancement d'un modèle 100M. Il n'est pas réaliste d'exiger d'un modèle de 17 millions de paramètres la polyvalence d'un grand GPT. La cible est un micro-assistant spécialisé, honnête et démontrable.

## Capacités ciblées

1. saluer et se présenter comme IvoireSLM ;
2. répondre brièvement en français à une question simple ;
3. reconnaître une information manquante et éviter d'inventer ;
4. répondre à un petit ensemble de faits ivoiriens stables et sourcés ;
5. extraire une réponse explicite depuis un court contexte ;
6. suivre des transformations simples : majuscules, minuscules, choix oui/non et extraction ;
7. transmettre les calculs exacts au moteur mathématique dans l'interface hybride.

Le calcul fiable n'est pas une condition model-only pour ce palier : les expériences précédentes ont montré qu'un Transformer 17M pouvait apprendre la forme d'un exercice sans apprendre l'algorithme arithmétique.

## Garde de publication

Les poids restent privés tant que les conditions suivantes ne sont pas toutes remplies :

| Mesure | Seuil minimal |
|---|---:|
| réponses terminées sans boucle répétitive | 90 % |
| identité et conversation de base | 80 % |
| incertitude et refus d'inventer | 80 % |
| faits ivoiriens contrôlés | 70 % |
| compréhension avec contexte | 70 % |
| suivi de consignes simples | 70 % |
| hausse de loss du français général | au plus +0,03 |
| test humain libre | 24 réponses acceptables sur 30 |

Une baisse de loss ne suffit jamais à promouvoir un checkpoint.

## Protocole

1. partir du meilleur checkpoint CPT 17M validé ;
2. entraîner 250 étapes avec un taux d'apprentissage faible ;
3. évaluer les générations déterministes par capacité ;
4. conserver le checkpoint seulement si le score progresse et si la garde de langue passe ;
5. poursuivre par paliers 250 → 500 → 750 → 1 000 ;
6. ouvrir ensuite l'interface à deux modes : `modèle seul` et `assistant hybride` ;
7. faire le test humain de 30 questions avant toute publication Hugging Face.

## Lancement du premier palier sur Colab

Le script `scripts/colab/run_assistant_v1_pilot.py` exige deux chemins explicites
afin de ne jamais choisir le mauvais checkpoint :

```python
import os

os.environ["IVOIRESLM_ASSISTANT_PARENT"] = "/content/drive/MyDrive/.../best.pt"
os.environ["IVOIRESLM_BPE_DIR"] = "/content/.../bpe_v0.4"
%run /content/ivoireslm/scripts/colab/run_assistant_v1_pilot.py
```

Le checkpoint parent doit avoir l'empreinte
`3dbd076f7df86fedad4f4b47a657f042dd156de05d8b37d9c46319811fde7072`.
Le pilote écrit dans un nouveau dossier `v1.1` de Google Drive et refuse tout
écrasement. Après l'étape 250, le script d'évaluation automatique décide si le
checkpoint est prêt pour l'interface et le test humain.

## Faits sourcés du premier curriculum

Les formulations du dataset sont rédigées pour le projet ; les sources servent à vérifier les faits et ne sont pas copiées comme textes d'entraînement.

- présentation officielle de la Côte d'Ivoire : `https://www.diplomatie.gouv.ci/informations-utiles/presentation-de-la-c%C3%B4te-d-ivoire` ;
- portail économique officiel : `https://www.eco.diplomatie.gouv.ci/` ;
- histoire et pays utilisateurs du franc CFA : `https://www.bceao.int/fr/content/histoire-du-franc-cfa` ;
- proclamation de l'indépendance : `https://www.gouv.ci/index.php/actualite/50eme-anniversaire-de-lindependance-voici-le-message-du-chef-de-letat-au-peuple-ivoirien-2609`.

## Passage au 100M

Le 100M ne commence qu'après réussite du 17M. Son corpus devra être plus grand, plus naturel et mieux équilibré, avec un inventaire de licences complet. Une nouvelle architecture de type GPT pourra être étudiée à ce moment-là ; une architecture différente ne compensera pas des données faibles.
