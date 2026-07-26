"""
Pré-processamento de texto para classificação.
Reutiliza a mesma lógica de limpeza do treinamento (pre_processamento.py).
"""

import re


def clean_text(text: str) -> str:
    """
    Limpa texto extraído de documentos jurídicos.
    Mesma função usada no treinamento para garantir consistência.
    """
    if not isinstance(text, str):
        return ""
    # Remove sequências repetitivas de caracteres decorativos
    text = re.sub(r'([_\-=\*#])\1{2,}', ' ', text)
    # Normaliza quebras de linha
    text = re.sub(r'\n+', ' ', text)
    # Remove caracteres especiais mantendo pontuação relevante
    text = re.sub(r'[^\w\s\.,;:!?\-\(\)\"\'\/]', ' ', text)
    # Remove espaços múltiplos
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()
