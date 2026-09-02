# Fichiers de la publication Hugging Face

La publication n'est complète que lorsque les fichiers suivants sont réunis dans le dépôt de modèle.

| Fichier | Source | Obligatoire |
|---|---|---:|
| `README.md` | ce dossier | oui |
| `config.json` | ce dossier | oui |
| `best.pt` | checkpoint Drive du pilote CPT v1.0 | oui |
| `tokenizer.json` | données BPE v0.4 | oui |
| `modeling_microivoire.py` | copie autonome de l'architecture v0.4 | oui |
| `generate.py` | exemple autonome d'inférence | oui |
| `progress.json` | dossier final du pilote | oui |
| `QUALITATIVE_DIAGNOSTIC_STEP1000.json` | dossier final du pilote | oui |
| `SHA256SUMS` | calculé après assemblage | oui |
| `latest.pt` | checkpoint avec optimiseur pour reprise | non, trop volumineux pour l'usage normal |

Avant l'envoi public :

1. vérifier chaque SHA256 ;
2. vérifier que `best.pt` est bien l'étape sélectionnée par la validation ;
3. exécuter `generate.py` dans un environnement propre ;
4. conserver l'avertissement « modèle expérimental » dans la fiche ;
5. terminer l'inventaire des licences des données avant de remplacer `license: other`.
