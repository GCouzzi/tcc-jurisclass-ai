"""
Módulo de extração de texto de PDFs.
Estratégia: PyMuPDF (rápido) → fallback Tesseract OCR (para escaneados).
"""

import io
import logging

import fitz  # PyMuPDF

from config import MIN_TEXT_LENGTH, OCR_LANG, OCR_DPI

logger = logging.getLogger(__name__)


def _extract_with_pymupdf(file_bytes: bytes) -> str:
    """Extrai texto de PDF usando PyMuPDF (apenas PDFs com texto embutido)."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts).strip()


def _extract_with_tesseract(file_bytes: bytes) -> str:
    """Extrai texto de PDF escaneado usando Tesseract OCR."""
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError as e:
        logger.error("Dependências de OCR não instaladas: %s", e)
        raise RuntimeError(
            "OCR não disponível. Instale pytesseract e pdf2image."
        ) from e

    try:
        images = convert_from_bytes(file_bytes, dpi=OCR_DPI)
    except Exception as e:
        logger.error("Falha ao converter PDF em imagens: %s", e)
        raise RuntimeError(
            "Não foi possível processar o PDF para OCR. "
            "Verifique se o arquivo não está corrompido."
        ) from e

    text_parts = []
    for i, img in enumerate(images):
        try:
            text = pytesseract.image_to_string(img, lang=OCR_LANG)
            text_parts.append(text)
        except Exception as e:
            logger.warning("OCR falhou na página %d: %s", i + 1, e)
            continue

    return "\n".join(text_parts).strip()


def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, str]:
    """
    Extrai texto de um PDF usando a melhor estratégia disponível.

    1. Tenta PyMuPDF (rápido, para PDFs com texto digital)
    2. Se texto extraído < MIN_TEXT_LENGTH, assume PDF escaneado → Tesseract OCR

    Args:
        file_bytes: Conteúdo do arquivo PDF em bytes.

    Returns:
        Tupla (texto_extraído, método_utilizado).
        método pode ser: "pymupdf", "ocr", "pymupdf_insufficient"
    """
    # Etapa 1: PyMuPDF
    logger.info("Tentando extração com PyMuPDF...")
    text = _extract_with_pymupdf(file_bytes)

    if len(text) >= MIN_TEXT_LENGTH:
        logger.info("PyMuPDF extraiu %d caracteres — suficiente.", len(text))
        return text, "pymupdf"

    logger.info(
        "PyMuPDF extraiu apenas %d chars (< %d). Tentando OCR...",
        len(text), MIN_TEXT_LENGTH
    )

    # Etapa 2: Tesseract OCR
    try:
        ocr_text = _extract_with_tesseract(file_bytes)
        if len(ocr_text) >= MIN_TEXT_LENGTH:
            logger.info("OCR extraiu %d caracteres — sucesso.", len(ocr_text))
            return ocr_text, "ocr"
        else:
            logger.warning(
                "OCR extraiu apenas %d chars — insuficiente.", len(ocr_text)
            )
            # Retorna o melhor dos dois
            best = ocr_text if len(ocr_text) > len(text) else text
            return best, "ocr" if len(ocr_text) > len(text) else "pymupdf_insufficient"
    except RuntimeError:
        # OCR não disponível — retorna o que o PyMuPDF conseguiu
        logger.warning("OCR não disponível. Usando resultado do PyMuPDF.")
        return text, "pymupdf_insufficient"
