# IvoireSLM — interface Gradio hybride v0.1

## Fonctionnement

L'écran affiche la question, la réponse, la route choisie et le statut de
vérification. Le checkpoint Transformer est chargé paresseusement : une question
mathématique reconnue n'utilise pas le modèle et répond immédiatement.

## Préparation dans Colab

Après authentification Google Cloud, télécharger l'archive applicative, le
checkpoint général et le tokenizer caractère v0.2 dans `/content`. Aucun de ces
fichiers n'est copié sur l'ordinateur de l'utilisateur.

L'interface est lancée avec `--share`. Gradio affiche alors une URL temporaire
accessible dans le navigateur. Cette URL doit rester privée et disparaît quand
le runtime Colab est arrêté.

## Lecture des statuts

- `✅ réponse vérifiée` : résultat produit par une règle mathématique exacte ;
- `⚠️ génération libre non vérifiée` : texte probabiliste du Transformer ;
- `unsupported_math_guard` : calcul détecté mais refusé, car absent des dix
  familles actuellement prises en charge.

## Validation

- contrôleur testé sans dépendre de l'interface graphique ;
- calcul exact, génération générale et entrée vide testés ;
- suite complète : 59 tests réussis.
