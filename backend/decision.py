"""Regra de abstenção compartilhada pelo painel e pela avaliação offline."""

import math


def validate_threshold(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("O limiar de confiança deve estar entre 0 e 1.")
    return value


def classification_decision(probabilities, labels, threshold=None):
    """Compara valores não arredondados; igualdade com o limiar é aceita."""
    threshold = validate_threshold(threshold) if threshold is not None else None
    probabilities = [float(p) for p in probabilities]
    labels = list(labels)
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("Probabilidades e classes devem ter o mesmo tamanho não nulo.")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Probabilidades devem ser finitas e estar entre 0 e 1.")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
        raise ValueError("As probabilidades devem somar 1.")
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    confidence = probabilities[index]
    review = threshold is None or confidence < threshold
    return {
        "label": "Não classificado" if review else labels[index],
        "suggested_label": labels[index],
        "confidence": confidence,
        "requires_review": review,
        "status": "manual_review" if review else "classified",
        "threshold": threshold,
        "message": ("Revisão manual necessária: limiar ainda não definido" if threshold is None
                    else "Revisão manual necessária" if review else "Classificação automática"),
    }
