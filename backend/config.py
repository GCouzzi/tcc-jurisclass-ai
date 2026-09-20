"""
Configurações do backend JurisClass AI.
Valores configuráveis via variáveis de ambiente.
"""

import os
from decision import validate_threshold
from model_source import DEFAULT_REPO_ID, DEFAULT_REVISION

# ─── Paths dos Modelos ────────────────────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "results/results_cv"))
USE_ENSEMBLE = os.getenv("USE_ENSEMBLE", "true").lower() == "true"
FALLBACK_FOLD = int(os.getenv("FALLBACK_FOLD", "5"))

# Limiar escolhido experimentalmente; ausente => classificação normal, sem abstenção.
_threshold = os.getenv("CONFIDENCE_THRESHOLD", "").strip()
CONFIDENCE_THRESHOLD = validate_threshold(_threshold) if _threshold else None

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", DEFAULT_REPO_ID).strip() or None
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", DEFAULT_REVISION).strip() or None

# ─── Extração de texto ────────────────────────────────────────────────────────
MIN_TEXT_LENGTH = 50
MAX_FILE_SIZE_MB = 20
OCR_LANG = "por"
OCR_DPI = 300

# ─── Mapeamento de labels internas → nomes do frontend ───────────────────────
LABEL_DISPLAY = {
    "acordao_de_2_instancia": "Acórdão",
    "agravo_em_recurso_extraordinario": "Agravo (ARE)",
    "despacho_de_admissibilidade": "Despacho",
    "peticao_do_RE": "Recurso Extraordinário",
    "sentenca": "Sentença",
}

# ─── Descrições para o frontend ──────────────────────────────────────────────
LABEL_DESCRIPTIONS = {
    "Acórdão": "Decisão colegiada proferida por tribunal.",
    "Agravo (ARE)": "Impugnação ao recurso extraordinário inadmitido.",
    "Despacho": "Decisão interlocutória ou ordem judicial de andamento processual.",
    "Recurso Extraordinário": "Petição de recurso dirigida ao Supremo Tribunal Federal.",
    "Sentença": "Decisão judicial de primeira instância que encerra o processo.",
}

# ─── CORS ─────────────────────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
