"""
JurisClass AI — Backend FastAPI

Classificação automática de documentos jurídicos.
Recebe PDF ou texto, extrai conteúdo, classifica e retorna predição com confiança.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import CORS_ORIGINS, MIN_TEXT_LENGTH, MAX_FILE_SIZE_MB, CONFIDENCE_THRESHOLD
from preprocessing import clean_text
from ocr import PdfExtractionError, extract_text_from_pdf
from classifier import classifier

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jurisclass")


# ─── Lifespan (startup/shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelos no startup."""
    logger.info("=" * 60)
    logger.info(" JurisClass AI — Iniciando...")
    logger.info("=" * 60)
    try:
        classifier.load()
        logger.info("Modelos carregados com sucesso — modo: %s", classifier.mode)
    except Exception as e:
        logger.error("FALHA ao carregar modelos: %s", e)
        raise
    logger.info("=" * 60)
    logger.info(" JurisClass AI — Pronto para requisições!")
    logger.info("=" * 60)
    yield
    logger.info("JurisClass AI — Encerrando.")


# ─── App FastAPI ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="JurisClass AI",
    description="Classificação automática de documentos jurídicos brasileiros.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Rotas ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check com informações do modelo carregado."""
    return {
        "status": "ok",
        "model_mode": classifier.mode,
        "models_loaded": len(classifier.models),
        "device": str(classifier.device),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.post("/classify")
async def classify_document(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
):
    """
    Classifica um documento jurídico.

    Envie um arquivo PDF OU um texto via form-data.
    - Se enviar PDF: extrai texto via PyMuPDF/OCR, depois classifica.
    - Se enviar texto: classifica diretamente.

    Returns:
        JSON com label, confidence, scores e extraction_method.
    """
    extraction_method = "text_input"
    raw_text = ""

    # ─── Validação ────────────────────────────────────────────────────────
    if file is None and (text is None or text.strip() == ""):
        raise HTTPException(
            status_code=400,
            detail="Envie um arquivo PDF ou um texto para classificação.",
        )

    # ─── Extração de texto ────────────────────────────────────────────────
    if file is not None:
        # Validar tipo
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Apenas arquivos PDF são aceitos.",
            )

        # Ler conteúdo
        file_bytes = await file.read()

        # Validar tamanho
        if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"O arquivo excede o limite de {MAX_FILE_SIZE_MB} MB.",
            )

        logger.info("PDF recebido: %s (%d bytes)", filename, len(file_bytes))

        # Extrair texto
        try:
            raw_text, extraction_method = extract_text_from_pdf(file_bytes)
        except PdfExtractionError as e:
            raise HTTPException(status_code=422, detail=str(e))

        logger.info(
            "Texto extraído: %d chars via %s", len(raw_text), extraction_method
        )

    else:
        raw_text = text.strip()
        extraction_method = "text_input"
        logger.info("Texto recebido diretamente: %d chars", len(raw_text))

    # ─── Pré-processamento ────────────────────────────────────────────────
    cleaned = clean_text(raw_text)

    if len(cleaned) < MIN_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Texto insuficiente para classificação "
                f"({len(cleaned)} caracteres, mínimo {MIN_TEXT_LENGTH}). "
                f"O documento pode estar vazio ou ilegível."
            ),
        )

    # ─── Classificação ───────────────────────────────────────────────────
    logger.info("Classificando texto (%d chars)...", len(cleaned))
    result = classifier.classify(cleaned)

    logger.info(
        "Resultado: %s (%.1f%%)",
        result["label"],
        result["confidence"] * 100,
    )

    return {
        **result,
        "extraction_method": extraction_method,
    }
