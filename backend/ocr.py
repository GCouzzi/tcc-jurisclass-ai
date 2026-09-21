"""Extração de texto de PDFs com Tesseract OCR."""

import io
import logging

import fitz  # PyMuPDF

from config import OCR_LANG, OCR_DPI

logger = logging.getLogger(__name__)


class PdfExtractionError(RuntimeError):
    """Erro de PDF que pode ser apresentado ao cliente da API."""


def _open_pdf(file_bytes: bytes) -> fitz.Document:
    """Abre e valida um PDF antes do OCR."""
    if not file_bytes:
        raise PdfExtractionError("O arquivo PDF está vazio.")

    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as error:
        logger.warning("PDF inválido ou corrompido: %s", error)
        raise PdfExtractionError(
            "Não foi possível abrir o PDF. Verifique se o arquivo é válido e não está corrompido."
        ) from error

    if document.needs_pass:
        document.close()
        raise PdfExtractionError(
            "O PDF está protegido por senha. Envie uma cópia desbloqueada."
        )
    if document.page_count == 0:
        document.close()
        raise PdfExtractionError("O PDF não contém páginas.")
    return document


def _extract_page_with_tesseract(page: fitz.Page) -> str:
    """Renderiza uma página com PyMuPDF e executa OCR com Tesseract."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as error:
        raise PdfExtractionError(
            "As dependências necessárias para OCR não estão instaladas."
        ) from error

    try:
        scale = OCR_DPI / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
            return pytesseract.image_to_string(image, lang=OCR_LANG).strip()
    except Exception as error:
        logger.warning("OCR falhou na página %d: %s", page.number + 1, error)
        raise PdfExtractionError(
            f"Não foi possível executar OCR na página {page.number + 1}."
        ) from error


def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, str]:
    """Renderiza todas as páginas e extrai todo o texto com Tesseract."""
    document = _open_pdf(file_bytes)
    text_parts: list[str] = []

    try:
        logger.info("Aplicando OCR nas %d página(s) do PDF.", document.page_count)
        for page in document:
            text_parts.append(_extract_page_with_tesseract(page))
        return "\n".join(text_parts).strip(), "ocr"
    finally:
        document.close()
