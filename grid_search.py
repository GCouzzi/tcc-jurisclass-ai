"""Executa no Google Colab a busca em grade completa de 24 configurações descrita no TCC."""

import argparse
import gc
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "raquelsilveira/legalbertpt_fp"
GRID_SEARCH_SAMPLE = 3000
SEED = 42
EPOCHS = (3, 5)
BATCH_SIZES = (8, 16, 32)
LEARNING_RATES = (2e-5, 5e-5)
WARMUP_STEPS = (100, 500)


class LegalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {
            key: torch.tensor(values[index])
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_function = nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )
        loss = loss_function(
            outputs.logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(prediction: EvalPrediction):
    predicted = np.argmax(prediction.predictions, axis=1)
    return {
        "accuracy": accuracy_score(prediction.label_ids, predicted),
        "f1_macro": f1_score(prediction.label_ids, predicted, average="macro"),
    }


def find_input(explicit, candidates, description):
    if explicit is not None:
        path = explicit
        if not path.is_file():
            raise FileNotFoundError(f"{description} não encontrado: {path}")
        return path
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    options = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{description} não encontrado. Caminhos verificados: {options}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, help="CSV de treinamento limpo.")
    parser.add_argument("--validation", type=Path, help="CSV de validação limpo.")
    parser.add_argument("--class-weights", type=Path, help="JSON com pesos das classes.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/grid_search_results.csv"),
        help="CSV de saída; é atualizado após cada configuração.",
    )
    parser.add_argument("--sample-size", type=int, default=GRID_SEARCH_SAMPLE)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=2,
        help="Microlote por dispositivo; a acumulação reproduz os lotes 8, 16 e 32.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite substituir um CSV de resultados existente.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A busca completa requer GPU CUDA. No Colab, selecione Ambiente de execução > "
            "Alterar tipo de ambiente de execução > GPU."
        )
    if args.sample_size < 5:
        raise ValueError("--sample-size deve comportar as cinco classes.")
    if args.micro_batch_size < 1:
        raise ValueError("--micro-batch-size deve ser positivo.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"A saída já existe: {args.output}. Use --overwrite ou escolha outro caminho."
        )

    train_path = find_input(
        args.train,
        (Path("dataset_clean/train_clean.csv"), Path("train_clean.csv"), Path("/content/train_clean.csv")),
        "CSV de treinamento",
    )
    validation_path = find_input(
        args.validation,
        (Path("dataset_clean/val_clean.csv"), Path("val_clean.csv"), Path("/content/val_clean.csv")),
        "CSV de validação",
    )
    weights_path = find_input(
        args.class_weights,
        (Path("results/class_weights.json"), Path("class_weights.json"), Path("/content/class_weights.json")),
        "JSON de pesos",
    )

    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {properties.total_memory / 1024**3:.1f} GiB")
    print(f"Treino: {train_path}")
    print(f"Validação: {validation_path}")
    print(f"Pesos: {weights_path}")

    df_train = pd.read_csv(train_path)
    df_validation = pd.read_csv(validation_path)
    required_columns = {"body", "document_type"}
    for name, frame in (("treino", df_train), ("validação", df_validation)):
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"CSV de {name} sem colunas obrigatórias: {sorted(missing)}")

    with weights_path.open(encoding="utf-8") as stream:
        class_weights_dict = json.load(stream)
    classes = sorted(class_weights_dict)
    if len(classes) != 5:
        raise ValueError(f"Eram esperadas cinco classes; recebidas: {classes}")
    unknown_train = set(df_train["document_type"]) - set(classes)
    unknown_validation = set(df_validation["document_type"]) - set(classes)
    if unknown_train or unknown_validation:
        raise ValueError(
            f"Classes sem peso: treino={sorted(unknown_train)}, "
            f"validação={sorted(unknown_validation)}"
        )

    if len(df_train) > args.sample_size:
        df_train, _ = train_test_split(
            df_train,
            train_size=args.sample_size,
            stratify=df_train["document_type"],
            random_state=SEED,
        )
    print(f"Amostra de treino: {len(df_train)} páginas")
    print(df_train["document_type"].value_counts().sort_index())
    print(f"Validação: {len(df_validation)} páginas")

    label2id = {label: index for index, label in enumerate(classes)}
    id2label = {index: label for index, label in enumerate(classes)}
    class_weights = torch.tensor(
        [class_weights_dict[label] for label in classes], dtype=torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizando treino e validação...")
    train_encodings = tokenizer(
        df_train["body"].astype(str).tolist(),
        truncation=True,
        padding=True,
        max_length=512,
    )
    validation_encodings = tokenizer(
        df_validation["body"].astype(str).tolist(),
        truncation=True,
        padding=True,
        max_length=512,
    )
    train_dataset = LegalDataset(
        train_encodings,
        [label2id[label] for label in df_train["document_type"]],
    )
    validation_dataset = LegalDataset(
        validation_encodings,
        [label2id[label] for label in df_validation["document_type"]],
    )

    combinations = list(
        itertools.product(EPOCHS, BATCH_SIZES, LEARNING_RATES, WARMUP_STEPS)
    )
    print(f"Busca completa: {len(combinations)} configurações")
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for number, (epochs, effective_batch, learning_rate, warmup) in enumerate(combinations, 1):
        set_seed(SEED)
        micro_batch = min(args.micro_batch_size, effective_batch)
        if effective_batch % micro_batch:
            raise ValueError(
                f"O lote {effective_batch} não é divisível pelo microlote {micro_batch}."
            )
        accumulation = effective_batch // micro_batch
        print(
            f"\n[{number}/{len(combinations)}] épocas={epochs}, lote={effective_batch}, "
            f"lr={learning_rate}, warmup={warmup}, microlote={micro_batch}, "
            f"acumulação={accumulation}"
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(classes),
            id2label=id2label,
            label2id=label2id,
        ).to(device)
        training_args = TrainingArguments(
            output_dir=str(
                Path("results_gs")
                / f"ep{epochs}_bs{effective_batch}_lr{learning_rate}_wu{warmup}"
            ),
            num_train_epochs=epochs,
            per_device_train_batch_size=micro_batch,
            per_device_eval_batch_size=micro_batch * 2,
            gradient_accumulation_steps=accumulation,
            learning_rate=learning_rate,
            warmup_steps=warmup,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            disable_tqdm=False,
            report_to="none",
            fp16=True,
            gradient_checkpointing=True,
            seed=SEED,
            data_seed=SEED,
        )
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
        )
        trainer.train()
        metrics = trainer.evaluate()
        results.append({
            "epochs": epochs,
            "batch_size": effective_batch,
            "learning_rate": learning_rate,
            "warmup_steps": warmup,
            "val_loss": metrics["eval_loss"],
            "val_accuracy": metrics["eval_accuracy"],
            "val_f1_macro": metrics["eval_f1_macro"],
        })
        pd.DataFrame(results).sort_values(
            "val_f1_macro", ascending=False
        ).to_csv(args.output, index=False)
        print(f"F1-macro={metrics['eval_f1_macro']:.6f}; parcial salvo em {args.output}")

        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

    final = pd.DataFrame(results).sort_values("val_f1_macro", ascending=False)
    final.to_csv(args.output, index=False)
    print("\nBusca em grade concluída.")
    print(final.head().to_string(index=False))
    print(f"Resultado completo: {args.output.resolve()}")


if __name__ == "__main__":
    main()
