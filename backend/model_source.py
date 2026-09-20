"""Resolve modelos locais ou baixa somente os folds pedidos do Hugging Face Hub."""
from pathlib import Path
import os
import hashlib


DEFAULT_REPO_ID = 'gcouzzi/jurisclass-legalbertpt'
DEFAULT_REVISION = 'f1b1130e7d7dd59c286c4928a5ca90b893528ed4'

MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt", "added_tokens.json")


def resolve_model_directory(model_dir, repo_id=None, revision=None, folds=(1, 2, 3, 4, 5)):
    if not repo_id:
        return Path(model_dir)
    if not revision:
        raise ValueError("Informe HF_MODEL_REVISION/--revision para fixar a versão do modelo do Hub.")
    from huggingface_hub import snapshot_download
    # HF_TOKEN ou login local são tratados pela biblioteca; não registrar tokens.
    # local_dir evita depender de privilégios de links simbólicos no Windows.
    cache_root = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    cache_key = hashlib.sha256(f"{repo_id}@{revision}".encode()).hexdigest()
    snapshot = snapshot_download(
        repo_id=repo_id, revision=revision,
        local_dir=cache_root / "jurisclass" / cache_key,
        allow_patterns=[f"fold_{fold}/best_model/{name}" for fold in folds for name in MODEL_FILES],
    )
    return Path(snapshot)
