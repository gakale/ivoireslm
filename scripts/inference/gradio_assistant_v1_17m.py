#!/usr/bin/env python3
"""Interface de qualification du 17M : modèle seul ou assistant hybride."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inference.hybrid_router import route_request
from inference.transformer17m_runtime import Transformer17MRuntime


def build_controller(runtime):
    def answer(request: str, mode: str) -> tuple[str, str, str]:
        if not request or not request.strip():
            return "Écris d’abord une question.", "aucune", "—"
        if mode == "Modèle seul":
            generation = runtime.generate(request.strip())
            status = "✅ fin EOS" if generation.stopped_on_eos else "⚠️ limite de longueur atteinte"
            return generation.text, runtime.model_id, status
        result = route_request(request.strip(), runtime)
        status = "✅ réponse calculée et vérifiée" if result.verified else "⚠️ réponse du modèle à vérifier"
        return result.response, result.route, status

    return answer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    runtime = Transformer17MRuntime(args.checkpoint, args.data_dir, args.model_script)
    answer = build_controller(runtime)
    import gradio as gr

    with gr.Blocks(title="IvoireSLM 17M — Qualification") as demo:
        gr.Markdown(
            "# IvoireSLM 17M — Banc d’essai\n"
            "**Modèle seul** montre exactement ce que le réseau a appris. "
            "**Assistant hybride** confie les calculs reconnus au moteur exact. "
            "Les autres réponses restent expérimentales et doivent être vérifiées."
        )
        mode = gr.Radio(
            ["Modèle seul", "Assistant hybride"], value="Modèle seul", label="Mode"
        )
        request = gr.Textbox(label="Question", lines=3)
        button = gr.Button("Répondre", variant="primary")
        response = gr.Textbox(label="Réponse", lines=8)
        with gr.Row():
            route = gr.Textbox(label="Route réellement utilisée", interactive=False)
            status = gr.Textbox(label="Statut", interactive=False)
        gr.Examples(
            [
                ["Comment t’appelles-tu ?", "Modèle seul"],
                ["Quelle est la capitale de la Côte d’Ivoire ?", "Modèle seul"],
                ["Que fais-tu lorsque tu ne sais pas ?", "Modèle seul"],
                ["Contexte : Awa habite à Bouaké. Où habite Awa ?", "Modèle seul"],
                ["combien font 12 + 7 ?", "Assistant hybride"],
            ],
            inputs=[request, mode],
        )
        button.click(answer, [request, mode], [response, route, status])
        request.submit(answer, [request, mode], [response, route, status])
    demo.queue(default_concurrency_limit=1).launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.server_port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
