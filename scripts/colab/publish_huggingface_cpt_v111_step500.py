#!/usr/bin/env python3
"""Assemble, vérifie et publie en privé le CPT v1.1.1 étape 500."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import torch


PROJECT = Path(__file__).resolve().parents[2]
TEMPLATE = (
    PROJECT
    / "releases/huggingface/ivoireslm-17m-cpt-v1.1.1-step500"
)
DEFAULT_CHECKPOINT = Path(
    "/content/drive/MyDrive/IvoireSLM/v1.4/checkpoints_17m_cpt_v111_pilot/best.pt"
)
DEFAULT_PROGRESS = Path(
    "/content/drive/MyDrive/IvoireSLM/v1.4/checkpoints_17m_cpt_v111_pilot/progress.json"
)
DEFAULT_TOKENIZER = Path("/content/ivoireslm_v04_base_restore/bpe_v0.4/tokenizer.json")
EXPECTED_CHECKPOINT_SHA256 = (
    "f97f0077cbbae78723284cf59f5f0193192a94b71e4d68bc47f86876dcb3d9e5"
)
EXPECTED_MODEL_ID = "microivoire_transformer_v1.4_17m_cpt_v111_pilot"
EXPECTED_MIXTURE_ID = "ivoireslm_pretraining_mix_v1.1.1_pilot"
EXPECTED_STEP = 500
EXPECTED_PARAMETER_COUNT = 17_129_280


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trusted_checkpoint(path: Path) -> dict:
    observed = sha256_file(path)
    if observed != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "checkpoint refusé : SHA256 observé "
            f"{observed}, attendu {EXPECTED_CHECKPOINT_SHA256}"
        )
    # Ce chargement pickle n'est effectué que sur le checkpoint local dont
    # l'empreinte est figée. Seuls les tenseurs sont ensuite publiés.
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_checkpoint(checkpoint: dict) -> dict[str, torch.Tensor]:
    cpt_config = checkpoint.get("cpt_config", {})
    if cpt_config.get("model_id") != EXPECTED_MODEL_ID:
        raise ValueError(f"model_id inattendu : {cpt_config.get('model_id')}")
    if cpt_config.get("mixture_id") != EXPECTED_MIXTURE_ID:
        raise ValueError(f"mixture_id inattendu : {cpt_config.get('mixture_id')}")
    if checkpoint.get("step") != EXPECTED_STEP:
        raise ValueError(f"étape inattendue : {checkpoint.get('step')}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("model_state_dict absent ou vide")
    count = sum(tensor.numel() for tensor in state.values())
    # Le state dict contient deux noms pour le poids lié embedding/sortie.
    tied_size = state["token_embedding.weight"].numel()
    if count - tied_size != EXPECTED_PARAMETER_COUNT:
        raise ValueError(f"nombre de paramètres inattendu : {count - tied_size}")
    return state


def write_sha256s(directory: Path) -> None:
    target = directory / "SHA256SUMS"
    files = sorted(
        path for path in directory.iterdir() if path.is_file() and path != target
    )
    target.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


def assemble(
    checkpoint_path: Path,
    tokenizer_path: Path,
    progress_path: Path,
    output_dir: Path,
) -> None:
    from safetensors.torch import load_file, save_file

    for path in (checkpoint_path, tokenizer_path, progress_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not TEMPLATE.is_dir():
        raise FileNotFoundError(TEMPLATE)
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} existe déjà ; utilise un nouveau dossier d'assemblage"
        )

    checkpoint = load_trusted_checkpoint(checkpoint_path)
    state = validate_checkpoint(checkpoint)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("model_id") != EXPECTED_MODEL_ID:
        raise ValueError("progress.json ne correspond pas au modèle attendu")
    if progress.get("test_opened") is not False:
        raise ValueError("le test final ne doit pas être ouvert")

    shutil.copytree(TEMPLATE, output_dir)
    shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")
    shutil.copy2(progress_path, output_dir / "progress.json")

    # Cloner supprime les stockages partagés non pris en charge par save_file.
    safe_state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in state.items()
    }
    weights_path = output_dir / "model.safetensors"
    save_file(safe_state, str(weights_path))

    # Vérification indépendante du fichier publié et de son architecture.
    sys.path.insert(0, str(output_dir))
    from modeling_microivoire import MicroIvoireTransformer17M, TrainingConfig

    model = MicroIvoireTransformer17M(TrainingConfig())
    model.load_state_dict(load_file(str(weights_path), device="cpu"))
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise AssertionError("nombre de paramètres du modèle reconstruit inattendu")

    metadata = {
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "source_step": checkpoint.get("step"),
        "model_id": EXPECTED_MODEL_ID,
        "mixture_id": EXPECTED_MIXTURE_ID,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "published_weights_format": "safetensors",
        "test_opened": False,
        "assistant_qualified": False,
    }
    (output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_sha256s(output_dir)


def upload_private(output_dir: Path, repo_id: str, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=False)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(output_dir),
        commit_message="Publish private IvoireSLM 17M CPT v1.1.1 step 500 candidate",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", help="Dépôt Hugging Face, par exemple compte/nom")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/ivoireslm_hf_release"))
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="prépare et vérifie sans créer de dépôt ni envoyer de fichier",
    )
    args = parser.parse_args()

    assemble(args.checkpoint, args.tokenizer, args.progress, args.output_dir)
    print(f"Candidat assemblé et vérifié : {args.output_dir}")
    print((args.output_dir / "SHA256SUMS").read_text(encoding="utf-8"))

    if args.assemble_only:
        return
    if not args.repo_id:
        raise ValueError("--repo-id est obligatoire pour publier")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("secret HF_TOKEN absent de l'environnement")
    upload_private(args.output_dir, args.repo_id, token)
    print(f"Publication privée terminée : https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
