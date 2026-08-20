# Tokenizer caractère IvoireSLM v0.1

Date d'exécution : 20 août 2026.

## Résultat

- Entraîné exclusivement sur le split `train` du corpus v0.5.0.
- Vocabulaire : 1 142 tokens, dont quatre tokens spéciaux.
- Train : 35 895 978 tokens, aucun token inconnu.
- Validation : 136 182 tokens, aucun token inconnu.
- Test : 119 352 tokens, aucun token inconnu.
- Round-trip `encode`/`decode` validé.
- Encodage binaire `uint16`, deux octets par token.

Artefacts :

`/home/gnakalehacker/ivoireslm-storage/tokenizers/character_v0.1`

SHA-256 du tokenizer :

`d0ca1d6064d8ceb1b06c7b3b6518bbeed01af86221fb116d51fba110602185f4`

## Reproduction

```bash
cd ~/ivoireslm
source ~/venv/bin/activate
python3 scripts/tokenizer/train_character_v01.py
```

Le split test n'a servi ni à construire le vocabulaire ni à ajuster un paramètre.
