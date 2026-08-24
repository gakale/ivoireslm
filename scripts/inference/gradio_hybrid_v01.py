#!/usr/bin/env python3
"""Interface web Gradio pour l'assistant hybride IvoireSLM v0.1."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chat_hybrid_v01 import TransformerGenerator
from inference.hybrid_router import route_request


def build_controller(
    checkpoint: Path,
    data_dir: Path,
    model_script: Path,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
):
    """Construit le contrôleur indépendamment de Gradio pour pouvoir le tester."""
    generation_args = SimpleNamespace(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )
    generator = TransformerGenerator(checkpoint, data_dir, model_script, generation_args)

    def answer(request: str) -> tuple[str, str, str, str]:
        if not request or not request.strip():
            return "Écris d’abord une question.", "aucune", "—", ""
        try:
            result = route_request(request.strip(), generator)
        except Exception as exc:  # Afficher une erreur lisible dans l'interface.
            return (
                "Une erreur technique empêche la génération.",
                "erreur",
                "❌ non vérifiée",
                f"{type(exc).__name__}: {exc}",
            )
        status = "✅ réponse vérifiée" if result.verified else "⚠️ génération libre non vérifiée"
        detail = result.detail or (
            "Calcul déterministe contrôlé." if result.verified else
            "Texte produit par le Transformer 5M : vérifier les faits importants."
        )
        return result.response, result.route, status, detail

    return answer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.temperature <= 0 or args.top_k <= 0 or args.max_new_tokens <= 0:
        raise ValueError("les paramètres de génération doivent être strictement positifs")
    for path, label in (
        (args.checkpoint, "checkpoint"),
        (args.data_dir / "tokenizer.json", "tokenizer"),
        (args.model_script, "script du modèle"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} manquant : {path}")

    import gradio as gr

    answer = build_controller(
        args.checkpoint,
        args.data_dir,
        args.model_script,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )

    with gr.Blocks(title="IvoireSLM — Assistant hybride") as demo:
        gr.Markdown(
            "# IvoireSLM — Assistant hybride v0.1\n"
            "Les exercices pris en charge sont calculés par un moteur exact. "
            "Les autres textes proviennent du petit Transformer 5M et doivent être vérifiés."
        )
        request = gr.Textbox(
            label="Ta question",
            lines=4,
            placeholder="Exemple : Calculer 18 + 24.",
        )
        generate = gr.Button("Répondre", variant="primary")
        response = gr.Textbox(label="Réponse", lines=10)
        with gr.Row():
            route = gr.Textbox(label="Route utilisée", interactive=False)
            status = gr.Textbox(label="Statut", interactive=False)
        detail = gr.Textbox(label="Explication du statut", interactive=False)
        gr.Examples(
            examples=[
                ["Calculer 18 + 24."],
                ["Soit x un nombre réel. Résoudre l’équation x² − 5x + 6 = 0."],
                ["À Adjamé, Awa achète 7 paniers à 1250 FCFA et paie 10000 FCFA. Monnaie ?"],
                ["Calculer l’intégrale de sin(x)."],
                ["La Côte d’Ivoire"],
            ],
            inputs=request,
        )
        generate.click(answer, inputs=request, outputs=[response, route, status, detail])
        request.submit(answer, inputs=request, outputs=[response, route, status, detail])

    demo.queue(default_concurrency_limit=1).launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.server_port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
