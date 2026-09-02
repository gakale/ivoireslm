# Audit de données avant le stage 4 assistant

Date : 2026-09-02

## Constat sur le supplément CPT v1.0

Le supplément contient 206 037 documents, 108 554 194 caractères et aucun
split test. Les sources admises conservent leur licence et leur empreinte.

| Domaine | Caractères | Part approximative |
|---|---:|---:|
| mathématiques (AQuA + GSM8K) | 97 428 589 | 89,75 % |
| code et agents | 3 722 158 | 3,43 % |
| Wikipédia anglais | 3 309 385 | 3,05 % |
| Wikipédia français | 3 110 725 | 2,87 % |
| cybersécurité défensive | 983 337 | 0,91 % |

Le supplément est donc utile comme ensemble de domaines séparés, mais il n'est
pas un corpus conversationnel équilibré. Il ne doit pas être concaténé
directement au corpus général. Le pilote CPT v1.0 l'a correctement utilisé par
échantillonnage pondéré, avec 60 % de corpus général et seulement 10 % de ce
bloc mathématique.

## Licences relevées

- Wikipédia français et anglais : CC BY-SA 3.0 / GFDL, couche ShareAlike ;
- GSM8K train : MIT ;
- AQuA train/dev : Apache-2.0 ;
- documentation DeepSeek admise : MIT ;
- CISA KEV : données publiques du gouvernement américain, usage défensif.

Les splits de test, MATH-500, les sources non commerciales, les éléments en
quarantaine, les charges d'exploitation et les preuves de concept ont été
exclus du supplément.

## Décision pour le stage 4

Le stage 4 ne lance pas une nouvelle continuation de préentraînement. Il repart
du CPT v1.0 déjà diversifié et applique une supervision courte, équilibrée et
protégée contre le collapse. Les 66 questions humaines servent uniquement de
benchmark tenu à l'écart. Leur contenu ne devient pas un raccourci
d'entraînement.

Le corpus Drive contient aussi un kit de collecte des langues ivoiriennes. Il
ne constitue pas à lui seul un dataset validé. Toute donnée dioula ajoutée à
une future version devra conserver sa licence et être contrôlée par un locuteur
compétent avant entraînement.
