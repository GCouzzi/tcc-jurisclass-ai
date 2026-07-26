"""
Classificador de documentos jurídicos.
Suporta modo single-fold e ensemble (soft voting).
"""

import os
import logging
from typing import Optional

import torch
import numpy as np
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (
    MODEL_DIR,
    USE_ENSEMBLE,
    FALLBACK_FOLD,
    HF_MODEL_NAME,
    LABEL_DISPLAY,
)

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Classificador com suporte a ensemble (soft voting) ou modelo único."""

    def __init__(self):
        self.models: list = []
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2label: dict = {}
        self.mode: str = "not_loaded"

    def load(self) -> None:
        """Carrega modelo(s) na memória. Chamado no startup do FastAPI."""
        logger.info("Device: %s", self.device)
        logger.info("MODEL_DIR: %s", MODEL_DIR)
        logger.info("USE_ENSEMBLE: %s | FALLBACK_FOLD: %d", USE_ENSEMBLE, FALLBACK_FOLD)

        if USE_ENSEMBLE:
            self._load_ensemble()
        else:
            self._load_single_fold()

        if not self.models:
            logger.error("Nenhum modelo carregado!")
            raise RuntimeError("Nenhum modelo disponível para classificação.")

        logger.info(
            "Classificador pronto — modo: %s | %d modelo(s) carregado(s)",
            self.mode, len(self.models)
        )

    def _find_available_folds(self) -> list[str]:
        """Encontra folds que possuem model.safetensors."""
        available = []
        for fold_num in range(1, 6):
            fold_path = os.path.join(MODEL_DIR, f"fold_{fold_num}", "best_model")
            model_file = os.path.join(fold_path, "model.safetensors")
            if os.path.isfile(model_file):
                available.append(fold_path)
                logger.info("  ✓ Fold %d encontrado: %s", fold_num, fold_path)
            else:
                logger.warning("  ✗ Fold %d sem model.safetensors", fold_num)
        return available

    def _load_model_from_path(self, path: str) -> AutoModelForSequenceClassification:
        """Carrega um modelo de um diretório local."""
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(self.device)
        model.eval()
        return model

    def _load_ensemble(self) -> None:
        """Carrega todos os folds disponíveis para ensemble."""
        fold_paths = self._find_available_folds()

        if len(fold_paths) < 2:
            logger.warning(
                "Apenas %d fold(s) disponível(is) — insuficiente para ensemble. "
                "Usando fallback para fold %d.",
                len(fold_paths), FALLBACK_FOLD
            )
            self._load_single_fold()
            return

        # Carrega tokenizer do primeiro fold disponível
        self.tokenizer = AutoTokenizer.from_pretrained(fold_paths[0])

        for path in fold_paths:
            model = self._load_model_from_path(path)
            self.models.append(model)
            logger.info("  Modelo carregado: %s", path)

        # Extrai mapeamento de labels do primeiro modelo
        self.id2label = self.models[0].config.id2label
        self.mode = f"ensemble_{len(self.models)}_folds"

    def _load_single_fold(self) -> None:
        """Carrega apenas o fold especificado (ou fallback HuggingFace)."""
        fold_path = os.path.join(MODEL_DIR, f"fold_{FALLBACK_FOLD}", "best_model")
        model_file = os.path.join(fold_path, "model.safetensors")

        if os.path.isfile(model_file):
            logger.info("Carregando modelo do fold %d: %s", FALLBACK_FOLD, fold_path)
            self.tokenizer = AutoTokenizer.from_pretrained(fold_path)
            model = self._load_model_from_path(fold_path)
            self.models.append(model)
            self.id2label = model.config.id2label
            self.mode = f"single_fold_{FALLBACK_FOLD}"
        else:
            logger.error("Fold %d não encontrado em %s.", FALLBACK_FOLD, fold_path)
            raise RuntimeError(
                f"Modelo do fold {FALLBACK_FOLD} não encontrado no caminho: {fold_path}. "
                "Certifique-se de que os pesos treinados (model.safetensors) estão disponíveis."
            )

    def classify(self, text: str) -> dict:
        """
        Classifica um texto e retorna label, confiança e scores.

        Args:
            text: Texto pré-processado do documento jurídico.

        Returns:
            {
                "label": "Acórdão",
                "confidence": 0.94,
                "scores": [
                    {"name": "Acórdão", "score": 0.94},
                    ...
                ]
            }
        """
        # Tokenizar
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inferência
        all_probs = []

        with torch.no_grad():
            for model in self.models:
                outputs = model(**inputs)
                probs = softmax(outputs.logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        # Soft voting (média das probabilidades)
        if len(all_probs) > 1:
            avg_probs = np.mean(all_probs, axis=0)
        else:
            avg_probs = all_probs[0]

        avg_probs = avg_probs[0]  # Remove batch dimension

        # Resultado
        top_idx = int(np.argmax(avg_probs))
        top_label_internal = self.id2label.get(str(top_idx), self.id2label.get(top_idx))
        top_label_display = LABEL_DISPLAY.get(top_label_internal, top_label_internal)
        confidence = float(avg_probs[top_idx])

        scores = []
        for i, prob in enumerate(avg_probs):
            internal_name = self.id2label.get(str(i), self.id2label.get(i))
            display_name = LABEL_DISPLAY.get(internal_name, internal_name)
            scores.append({
                "name": display_name,
                "score": round(float(prob), 4),
            })

        # Ordenar por score decrescente
        scores.sort(key=lambda x: x["score"], reverse=True)

        return {
            "label": top_label_display,
            "confidence": round(confidence, 4),
            "scores": scores,
        }


# Instância global — carregada no startup do FastAPI
classifier = DocumentClassifier()
