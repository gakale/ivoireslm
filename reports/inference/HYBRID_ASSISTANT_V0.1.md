# IvoireSLM — interface hybride v0.1

## Objectif

L'interface `chat_hybrid_v01.py` reçoit une requête unique et choisit une route :

1. `deterministic_math_tool_v0.1` pour les dix familles vérifiées ;
2. `unsupported_math_guard` pour une demande mathématique non prise en charge ;
3. `microivoire_transformer_v0.2_5m` pour la génération générale.

La propriété JSON `verified` indique si le contenu est garanti par une règle
déterministe. Elle vaut `true` pour un calcul exact ou un refus contrôlé, et
`false` pour une génération libre du Transformer.

## Checkpoint général

L'interface utilise le checkpoint de préentraînement général, et non le
checkpoint SFT mathématique :

- modèle : `microivoire_transformer_v0.2_5m` ;
- étape : 9 250 ;
- SHA256 : `ca1dfc4639f2a9887a7dadaaddeac3ffbd28da85f279039cb5093a80b56374f4` ;
- tokenizer : `character_v0.2` ;
- appareil : CPU ou CUDA, choisi automatiquement.

## Contrôles réalisés

- 56 tests automatisés passent ;
- un calcul reconnu retourne une réponse exacte sans charger le Transformer ;
- une intégrale non prise en charge est arrêtée par le garde-fou ;
- une requête générale charge le Transformer et génère correctement sur CPU ;
- 120 nouveaux caractères sont générés en environ 5,2 secondes sur la VM CPU.

Les sorties générales restent des continuations probabilistes d'un petit modèle
de 4,76 millions de paramètres. Elles ne doivent pas être présentées comme des
faits vérifiés.
