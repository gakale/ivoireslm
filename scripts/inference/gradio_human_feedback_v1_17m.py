#!/usr/bin/env python3
"""Interface humaine enregistrant questions, réponses, jugements et corrections."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.human_feedback import CATEGORIES, RATINGS, append_record, create_record
from gradio_assistant_v1_17m import build_controller
from inference.transformer17m_runtime import Transformer17MRuntime


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument("--feedback-file", type=Path, required=True)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-port", type=int, default=7862)
    return parser.parse_args()


def main():
    args = arguments()
    runtime = Transformer17MRuntime(args.checkpoint, args.data_dir, args.model_script)
    controller = build_controller(runtime)
    checkpoint_sha = sha256(args.checkpoint)
    import gradio as gr

    def generate(question, mode):
        response, route, status = controller(question, mode)
        interaction = {
            "question": question,
            "response": response,
            "mode": mode,
            "route": route,
            "status": status,
            "model_id": runtime.model_id,
            "checkpoint_step": runtime.checkpoint_step,
        }
        return response, route, status, interaction, "Réponse prête à être évaluée."

    def save(interaction, rating, correction, category, consent):
        try:
            record = create_record(
                interaction or {}, rating=rating, correction=correction or "",
                category=category, consent=consent, checkpoint_sha256=checkpoint_sha,
            )
            count = append_record(args.feedback_file, record)
        except ValueError as exc:
            return f"❌ {exc}"
        return f"✅ Retour enregistré dans Drive. Total : {count}"

    with gr.Blocks(title="IvoireSLM 17M — Retours humains") as demo:
        gr.Markdown(
            "# IvoireSLM 17M — Banc de retours humains\n"
            "Teste surtout **Modèle seul**. N'écris aucune donnée personnelle ou secrète. "
            "Les retours sont conservés séparément et ne réentraînent jamais automatiquement le modèle."
        )
        interaction = gr.State({})
        mode = gr.Radio(["Modèle seul", "Assistant hybride"], value="Modèle seul", label="Mode")
        question = gr.Textbox(label="Question inédite", lines=3)
        generate_button = gr.Button("Générer", variant="primary")
        response = gr.Textbox(label="Réponse du système", lines=7, interactive=False)
        with gr.Row():
            route = gr.Textbox(label="Route utilisée", interactive=False)
            generation_status = gr.Textbox(label="Statut technique", interactive=False)
        gr.Markdown("## Évaluation humaine")
        category = gr.Dropdown(list(CATEGORIES), value="conversation_ordinaire", label="Catégorie")
        rating = gr.Radio(list(RATINGS), label="Ton jugement")
        correction = gr.Textbox(
            label="Réponse corrigée (obligatoire si incorrecte ou partielle)", lines=4
        )
        consent = gr.Checkbox(
            label=(
                "Je confirme que ce retour ne contient aucune donnée personnelle ou secrète, "
                "et j’autorise son utilisation pour la recherche et l’entraînement IvoireSLM."
            )
        )
        save_button = gr.Button("Enregistrer ce retour")
        feedback_status = gr.Textbox(label="Enregistrement", interactive=False)
        generate_button.click(
            generate,
            [question, mode],
            [response, route, generation_status, interaction, feedback_status],
        )
        question.submit(
            generate,
            [question, mode],
            [response, route, generation_status, interaction, feedback_status],
        )
        save_button.click(
            save,
            [interaction, rating, correction, category, consent],
            feedback_status,
        )
    demo.queue(default_concurrency_limit=1).launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.server_port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
