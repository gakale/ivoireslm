#!/usr/bin/env python3
"""Interface du 17M outillé : modèle brut, outils, Internet et retours humains."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.human_feedback import CATEGORIES, RATINGS, append_record, create_record
from inference.tool_assistant_v1 import WikipediaFrenchSearch, route_assistant
from inference.transformer17m_runtime import Transformer17MRuntime


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_controller(runtime, web_search=None):
    search = web_search or WikipediaFrenchSearch()

    def answer(request: str, mode: str, internet_enabled: bool):
        if not request or not request.strip():
            return "Écris d’abord une question.", "aucune", "—", "", {}
        if mode == "Modèle seul":
            generation = runtime.generate(request.strip())
            status = "✅ fin EOS" if generation.stopped_on_eos else "⚠️ limite de longueur atteinte"
            result = (generation.text, runtime.model_id, status, "")
        else:
            routed = route_assistant(
                request.strip(), runtime,
                internet_enabled=bool(internet_enabled), web_search=search,
            )
            sources = "\n".join(f"- {source}" for source in routed.sources)
            result = (routed.response, routed.route, routed.status, sources)
        interaction = {
            "question": request.strip(),
            "response": result[0],
            "mode": mode,
            "route": result[1],
            "status": result[2],
            "sources": result[3],
            "internet_enabled": bool(internet_enabled),
            "model_id": runtime.model_id,
            "checkpoint_step": runtime.checkpoint_step,
        }
        return *result, interaction

    return answer


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-script", type=Path, required=True)
    parser.add_argument("--feedback-file", type=Path, required=True)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main():
    args = arguments()
    runtime = Transformer17MRuntime(args.checkpoint, args.data_dir, args.model_script)
    controller = build_controller(runtime)
    checkpoint_sha = sha256(args.checkpoint)
    import gradio as gr

    def save(interaction, rating, correction, category, consent):
        try:
            record = create_record(
                interaction or {}, rating=rating, correction=correction or "",
                category=category, consent=consent, checkpoint_sha256=checkpoint_sha,
            )
            record["sources"] = (interaction or {}).get("sources", "")
            record["internet_enabled"] = bool((interaction or {}).get("internet_enabled"))
            count = append_record(args.feedback_file, record)
        except ValueError as exc:
            return f"❌ {exc}"
        return f"✅ Retour enregistré séparément dans Drive. Total : {count}"

    with gr.Blocks(title="IvoireSLM 17M — Assistant outillé") as demo:
        gr.Markdown(
            "# IvoireSLM 17M — Assistant expérimental outillé\n"
            "**Modèle seul** expose les poids sans aide. **Assistant outillé** utilise "
            "une calculatrice, une fiche d’identité, une base ivoirienne vérifiée, un "
            "petit lexique dioula et, si activé, une recherche Wikipédia. La route et "
            "le niveau de confiance sont toujours affichés."
        )
        interaction = gr.State({})
        with gr.Row():
            mode = gr.Radio(
                ["Assistant outillé", "Modèle seul"],
                value="Assistant outillé", label="Mode",
            )
            internet_enabled = gr.Checkbox(
                value=False,
                label="Autoriser la recherche Internet pour les questions factuelles inconnues",
            )
        question = gr.Textbox(label="Ta question", lines=3)
        generate = gr.Button("Répondre", variant="primary")
        response = gr.Markdown(label="Réponse")
        with gr.Row():
            route = gr.Textbox(label="Route réellement utilisée", interactive=False)
            status = gr.Textbox(label="Statut de confiance", interactive=False)
        sources = gr.Markdown(label="Sources")
        gr.Examples(
            [
                ["Comment t’appelles-tu ?", "Assistant outillé", False],
                ["Quelle est la capitale de la Côte d’Ivoire ?", "Assistant outillé", False],
                ["Comment dit-on bonjour le matin en dioula ?", "Assistant outillé", False],
                ["Combien font 17 × 8 ?", "Assistant outillé", False],
                ["Cherche sur Internet qui est Marie Curie.", "Assistant outillé", True],
                ["Parle-moi librement de la Côte d’Ivoire.", "Modèle seul", False],
            ],
            inputs=[question, mode, internet_enabled],
        )
        generate.click(
            controller, [question, mode, internet_enabled],
            [response, route, status, sources, interaction],
        )
        question.submit(
            controller, [question, mode, internet_enabled],
            [response, route, status, sources, interaction],
        )

        gr.Markdown(
            "## Retour humain\nLes retours ne réentraînent jamais automatiquement le modèle. "
            "N’écris aucune donnée personnelle, aucun mot de passe et aucun secret."
        )
        with gr.Row():
            category = gr.Dropdown(
                list(CATEGORIES), value="conversation_ordinaire", label="Catégorie"
            )
            rating = gr.Radio(list(RATINGS), label="Ton jugement")
        correction = gr.Textbox(
            label="Réponse corrigée si la réponse est fausse ou partielle", lines=3
        )
        consent = gr.Checkbox(
            label=(
                "Je confirme que ce retour ne contient aucune donnée personnelle ou "
                "secrète et j’autorise son utilisation pour la recherche IvoireSLM."
            )
        )
        save_button = gr.Button("Enregistrer le retour")
        feedback_status = gr.Textbox(label="Enregistrement", interactive=False)
        save_button.click(
            save, [interaction, rating, correction, category, consent], feedback_status
        )

    demo.queue(default_concurrency_limit=2).launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.server_port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
