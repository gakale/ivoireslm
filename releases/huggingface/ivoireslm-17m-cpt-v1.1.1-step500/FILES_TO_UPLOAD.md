# Fichiers de la publication Hugging Face

Le script `scripts/colab/publish_huggingface_cpt_v111_step500.py` assemble et vérifie ces fichiers avant tout envoi.

| Fichier | Obligatoire |
|---|---:|
| `README.md` | oui |
| `config.json` | oui |
| `model.safetensors` | oui |
| `tokenizer.json` | oui |
| `modeling_microivoire.py` | oui |
| `generate.py` | oui |
| `progress.json` | oui |
| `checkpoint_metadata.json` | oui |
| `SHA256SUMS` | oui |

Le dépôt est créé en mode privé. Ne le rendre public qu’après l’inventaire des licences et la qualification humaine prévue dans `docs/ASSISTANT_17M_QUALIFICATION_V1.md`.
