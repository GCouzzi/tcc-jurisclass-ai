"""Compara o melhor fold e o ensemble diante de documentos Others.

Executa amostragem, inferência, escolha do limiar na validação e avaliação no
conjunto de teste. Grava somente um relatório JSON agregado.

Uso:
    python teste_others.py
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

from backend.model_source import DEFAULT_REPO_ID, DEFAULT_REVISION, resolve_model_directory
from backend.preprocessing import clean_text

OTHERS = "outros"
REVIEW = "revisao_manual"
BEST_FOLD = 5
ALL_FOLDS = (1, 2, 3, 4, 5)
CHECKPOINTS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 1.0)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def sample_indices(records, others_size, known_size, seed):
    """Amostra sem reposição; a parte conhecida é aproximadamente proporcional."""
    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[row["true_label"]].append(index)
    others = groups.pop(OTHERS, [])
    if len(others) < others_size:
        raise ValueError(f"Others insuficiente: {len(others)} disponível(is), {others_size} solicitado(s).")
    total_known = sum(len(indices) for indices in groups.values())
    if known_size > total_known or known_size < len(groups):
        raise ValueError("O tamanho conhecido deve incluir todas as classes e não exceder o disponível.")

    rng = random.Random(seed)
    selected = rng.sample(others, others_size)
    quotas = {label: 1 for label in groups}
    remaining = known_size - len(groups)
    capacities = {label: len(indices) - 1 for label, indices in groups.items()}
    total_capacity = sum(capacities.values())
    exact = {
        label: remaining * capacity / total_capacity if total_capacity else 0
        for label, capacity in capacities.items()
    }
    for label in groups:
        quotas[label] += int(exact[label])
    missing = known_size - sum(quotas.values())
    order = sorted(groups, key=lambda label: (-(exact[label] % 1), label))
    for label in order[:missing]:
        quotas[label] += 1
    for label in sorted(groups):
        selected.extend(rng.sample(groups[label], quotas[label]))
    return sorted(selected)


def load_sample(path, labels, others_size, known_size, seed):
    csv.field_size_limit(10**8)
    records = []
    texts = []
    excluded = Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"body", "document_type"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} deve conter as colunas body e document_type.")
        for row_index, row in enumerate(reader):
            label = row["document_type"].strip()
            if label.casefold() in {"others", "outros"}:
                label = OTHERS
            if label not in labels + [OTHERS]:
                raise ValueError(f"Classe desconhecida na linha {row_index + 2} de {path}: {label}")
            text = clean_text(row["body"])
            if len(text) < 50:
                excluded[label] += 1
                continue
            records.append({
                "row_index": row_index,
                "true_label": label,
                "process_id": row.get("process_id", ""),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            })
            texts.append(text)

    population = Counter(row["true_label"] for row in records)
    selected = sample_indices(records, others_size, known_size, seed)
    sampled_records = [records[index] for index in selected]
    sampled_texts = [texts[index] for index in selected]
    metadata = {
        "source": str(path.resolve()),
        "population_after_cleaning": dict(population),
        "sample_counts": dict(Counter(row["true_label"] for row in sampled_records)),
        "excluded_short_texts": dict(excluded),
    }
    return sampled_records, sampled_texts, metadata


def read_labels(model_dir):
    mappings = []
    for fold in ALL_FOLDS:
        config_path = model_dir / f"fold_{fold}" / "best_model" / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        mappings.append(config["id2label"])
    if any(mapping != mappings[0] for mapping in mappings[1:]):
        raise ValueError("Os folds possuem mapeamentos de classes diferentes.")
    labels = [mappings[0][str(index)] for index in range(len(mappings[0]))]
    if len(labels) != 5 or len(set(labels)) != 5 or OTHERS in labels:
        raise ValueError("Eram esperadas cinco classes conhecidas, sem Others.")
    return labels


def infer_all_folds(model_dir, split_texts, labels, batch_size, threads):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if batch_size < 1 or threads < 1:
        raise ValueError("batch-size e threads devem ser positivos.")
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = model_dir / "fold_1" / "best_model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    ensemble = {
        split: np.zeros((len(texts), len(labels)), dtype=np.float64)
        for split, texts in split_texts.items()
    }
    fold5 = {}
    started = time.monotonic()
    print(f"Dispositivo: {device}; páginas: " + ", ".join(
        f"{split}={len(texts)}" for split, texts in split_texts.items()
    ), flush=True)

    for fold in ALL_FOLDS:
        path = model_dir / f"fold_{fold}" / "best_model"
        model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(device)
        model.eval()
        model_labels = [model.config.id2label[index] for index in range(len(labels))]
        if model_labels != labels:
            raise ValueError(f"Mapeamento incompatível no fold {fold}.")
        for split, texts in split_texts.items():
            split_probs = np.zeros_like(ensemble[split])
            with torch.inference_mode():
                for start in range(0, len(texts), batch_size):
                    batch = tokenizer(
                        texts[start:start + batch_size],
                        truncation=True,
                        padding=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    batch = {key: value.to(device) for key, value in batch.items()}
                    precision = (
                        torch.autocast(device_type="cuda", dtype=torch.float16)
                        if device.type == "cuda"
                        else nullcontext()
                    )
                    with precision:
                        probabilities = torch.softmax(model(**batch).logits, dim=-1)
                    values = probabilities.float().cpu().numpy()
                    split_probs[start:start + len(values)] = values
                    if start % (batch_size * 25) == 0:
                        elapsed = time.monotonic() - started
                        print(f"fold {fold}/5, {split}: {start + len(values)}/{len(texts)} ({elapsed:.0f}s)", flush=True)
            ensemble[split] += split_probs / len(ALL_FOLDS)
            if fold == BEST_FOLD:
                fold5[split] = split_probs.copy()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "best_fold_5": fold5,
        "ensemble_5_folds": ensemble,
    }, {
        "device": str(device),
        "runtime_seconds": time.monotonic() - started,
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": __import__("transformers").__version__,
        },
    }


def predicted_index(probabilities):
    return max(range(len(probabilities)), key=lambda index: probabilities[index])


def confidence_diagnostics(records, probabilities, labels):
    import numpy as np

    groups = {"others": [], "known_correct": [], "known_incorrect": []}
    forced_labels = Counter()
    for row, probs in zip(records, probabilities):
        index = predicted_index(probs)
        confidence = float(probs[index])
        if row["true_label"] == OTHERS:
            groups["others"].append(confidence)
            forced_labels[labels[index]] += 1
        elif labels[index] == row["true_label"]:
            groups["known_correct"].append(confidence)
        else:
            groups["known_incorrect"].append(confidence)

    result = {"others_forced_labels": dict(forced_labels), "groups": {}}
    for name, values in groups.items():
        array = np.asarray(values, dtype=float)
        result["groups"][name] = {
            "count": len(values),
            "mean": float(array.mean()) if len(array) else None,
            "percentiles": {
                str(point): float(np.percentile(array, point)) if len(array) else None
                for point in (5, 25, 50, 75, 95)
            },
            "at_or_above": {
                str(threshold): {
                    "count": int((array >= threshold).sum()),
                    "rate": float((array >= threshold).mean()) if len(array) else None,
                }
                for threshold in (0.6, 0.8, 0.9, 0.95, 0.99)
            },
        }
    return result


def summarize(records, probabilities, labels, threshold):
    matrix = Counter()
    closed = Counter()
    for row, probs in zip(records, probabilities):
        index = predicted_index(probs)
        suggested = labels[index]
        confidence = float(probs[index])
        prediction = suggested if confidence >= threshold else REVIEW
        matrix[row["true_label"], prediction] += 1
        if row["true_label"] in labels:
            closed[row["true_label"], suggested] += 1

    known = sum(value for (truth, _), value in matrix.items() if truth in labels)
    others = sum(value for (truth, _), value in matrix.items() if truth == OTHERS)
    rejected_known = sum(matrix[label, REVIEW] for label in labels)
    accepted_known = known - rejected_known
    correct_accepted = sum(matrix[label, label] for label in labels)
    accepted_others = others - matrix[OTHERS, REVIEW]
    accepted_total = accepted_known + accepted_others
    f1_values = []
    per_class = {}
    for label in labels:
        support = sum(value for (truth, _), value in matrix.items() if truth == label)
        accepted = support - matrix[label, REVIEW]
        tp = closed[label, label]
        fp = sum(value for (truth, pred), value in closed.items() if pred == label and truth != label)
        fn = sum(value for (truth, pred), value in closed.items() if truth == label and pred != label)
        f1_values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        per_class[label] = {
            "support": support,
            "coverage": ratio(accepted, support),
            "accepted_accuracy": ratio(matrix[label, label], accepted),
        }

    metrics = {
        "threshold": float(threshold),
        "n_known": known,
        "n_others": others,
        "known_coverage": ratio(accepted_known, known),
        "known_rejection_rate": ratio(rejected_known, known),
        "others_rejection_rate": ratio(matrix[OTHERS, REVIEW], others),
        "others_false_acceptance_rate": ratio(accepted_others, others),
        "accepted_known_accuracy": ratio(correct_accepted, accepted_known),
        "accepted_total_risk": ratio(accepted_total - correct_accepted, accepted_total),
        "total_review_rate": ratio(rejected_known + matrix[OTHERS, REVIEW], len(records)),
        "closed_set_accuracy": ratio(sum(closed[label, label] for label in labels), known),
        "closed_set_f1_macro": sum(f1_values) / len(labels),
        "per_class": per_class,
    }
    confusion = {
        truth: {prediction: matrix[truth, prediction] for prediction in labels + [REVIEW]}
        for truth in labels + [OTHERS]
    }
    return metrics, confusion


def choose_threshold(records, probabilities, labels, min_known_coverage):
    import numpy as np

    known_confidences = []
    others_confidences = []
    for row, probs in zip(records, probabilities):
        confidence = float(max(probs))
        (others_confidences if row["true_label"] == OTHERS else known_confidences).append(confidence)
    known = np.asarray(known_confidences)
    others = np.asarray(others_confidences)
    candidates = np.unique(np.concatenate((
        np.asarray([0.0, 1.0]),
        known,
        np.nextafter(others, math.inf),
    )))
    feasible = []
    for threshold in candidates:
        coverage = float((known >= threshold).mean())
        if coverage + 1e-12 >= min_known_coverage:
            feasible.append({
                "threshold": float(threshold),
                "known_coverage": coverage,
                "others_rejection_rate": float((others < threshold).mean()),
            })
    return max(
        feasible,
        key=lambda item: (
            item["others_rejection_rate"],
            item["known_coverage"],
            -item["threshold"],
        ),
    )


def evaluate_configuration(records_by_split, probabilities_by_split, labels, min_known_coverage):
    validation_records = records_by_split["validation"]
    validation_probs = probabilities_by_split["validation"]
    selected = choose_threshold(validation_records, validation_probs, labels, min_known_coverage)
    threshold = selected["threshold"]
    validation_selected, _ = summarize(validation_records, validation_probs, labels, threshold)
    test_selected, test_confusion = summarize(
        records_by_split["test"], probabilities_by_split["test"], labels, threshold
    )
    checkpoints = sorted(set(CHECKPOINTS + (threshold,)))
    return {
        "validation": {
            "confidence_diagnostics": confidence_diagnostics(validation_records, validation_probs, labels),
            "selected_threshold": selected,
            "selected_threshold_metrics": validation_selected,
            "checkpoint_metrics": [
                summarize(validation_records, validation_probs, labels, point)[0]
                for point in checkpoints
            ],
        },
        "test": {
            "confidence_diagnostics": confidence_diagnostics(
                records_by_split["test"], probabilities_by_split["test"], labels
            ),
            "selected_threshold_metrics": test_selected,
            "confusion_with_manual_review": test_confusion,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=Path("dataset_victor/validation_small.csv"))
    parser.add_argument("--test", type=Path, default=Path("dataset_victor/test_small.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/rejection/resultado_others.json"))
    parser.add_argument("--others-sample", type=int, default=2129)
    parser.add_argument("--known-sample", type=int, default=2129)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--min-known-coverage", type=float, default=0.95)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--local-models", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=Path("results/results_cv"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        parser.error(f"A saída já existe: {args.output}. Use --overwrite para substituí-la.")
    if args.others_sample < 1 or args.known_sample < 5:
        parser.error("A amostra deve ter ao menos 1 Others e 5 páginas conhecidas.")
    if not 0 < args.min_known_coverage <= 1:
        parser.error("--min-known-coverage deve estar em (0, 1].")
    if args.validation.resolve() == args.test.resolve():
        parser.error("Validação e teste devem ser CSVs diferentes.")

    model_dir = resolve_model_directory(
        args.model_dir,
        None if args.local_models else args.repo_id,
        None if args.local_models else args.revision,
        ALL_FOLDS,
    )
    labels = read_labels(model_dir)
    records_by_split = {}
    texts_by_split = {}
    sample_metadata = {}
    for offset, (split, path) in enumerate((
        ("validation", args.validation),
        ("test", args.test),
    )):
        records, texts, metadata = load_sample(
            path, labels, args.others_sample, args.known_sample, args.seed + offset
        )
        records_by_split[split] = records
        texts_by_split[split] = texts
        sample_metadata[split] = metadata

    overlap = (
        {row["process_id"] for row in records_by_split["validation"]}
        & {row["process_id"] for row in records_by_split["test"]}
    ) - {"", None}
    if overlap:
        raise ValueError("Há processos compartilhados entre as amostras de validação e teste.")

    probabilities, runtime = infer_all_folds(
        model_dir, texts_by_split, labels, args.batch_size, args.threads
    )
    configurations = {
        name: evaluate_configuration(
            records_by_split, split_probabilities, labels, args.min_known_coverage
        )
        for name, split_probabilities in probabilities.items()
    }
    report = {
        "experiment": {
            "purpose": "Medir a confiança em Others e determinar limiares sem usar o teste na escolha.",
            "unit": "page",
            "configurations": {
                "best_fold_5": [5],
                "ensemble_5_folds": [1, 2, 3, 4, 5],
            },
            "best_fold_basis": "Fold 5 teve o maior F1-macro interno entre os folds.",
            "repo_id": None if args.local_models else args.repo_id,
            "revision": None if args.local_models else args.revision,
            "sampling": {
                "others_per_split": args.others_sample,
                "known_per_split": args.known_sample,
                "validation_seed": args.seed,
                "test_seed": args.seed + 1,
                "reference": "Sentença: 2.129 páginas no treino original, terceira classe conhecida.",
            },
            "threshold_rule": (
                "Maximizar a rejeição de Others mantendo ao menos a cobertura mínima das classes "
                "conhecidas na validação; desempatar por maior cobertura e menor limiar."
            ),
            "min_known_coverage": args.min_known_coverage,
            "test_was_not_used_to_select_threshold": True,
            "others_is_not_a_trained_output_class": True,
        },
        "labels": labels,
        "samples": sample_metadata,
        "runtime": runtime,
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Relatório: {args.output.resolve()}")
    for name, result in configurations.items():
        threshold = result["validation"]["selected_threshold"]["threshold"]
        rejected = result["test"]["selected_threshold_metrics"]["others_rejection_rate"]
        print(f"{name}: limiar={threshold:.8f}; Others rejeitados no teste={rejected:.2%}")


if __name__ == "__main__":
    main()
