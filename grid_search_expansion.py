"""Executa o experimento de expansão de fronteira do Grid Search para o TCC.

Treina apenas as 3 novas combinações nos limites superiores sugeridos pelo orientador:
1) 8 épocas, lote 16, lr 5e-5, warmup 500
2) 5 épocas, lote 16, lr 8e-5, warmup 500
3) 8 épocas, lote 16, lr 8e-5, warmup 500

Mantém estritamente o mesmo protocolo de grid_search.py:
- Amostra estratificada de 3.000 registros
- Mesmo split e semente (SEED = 42)
- Avaliação no conjunto de validação (val_clean.csv)
- Pesos das classes (class_weights.json)
"""

import argparse
import gc
import json
import random
import time
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
BENCHMARK_F1 = 0.8642347606721715  # 86,42% (melhor do Grid Search original)

# As 3 novas combinações de fronteira definidas com o orientador
NEW_CONFIGURATIONS = (
    # (epochs, batch_size, learning_rate, warmup_steps, descricao)
    (8, 16, 5e-5, 500, "Expansao de epocas (8 epocas, lr 5e-5)"),
    (5, 16, 8e-5, 500, "Expansao de taxa (5 epocas, lr 8e-5)"),
    (8, 16, 8e-5, 500, "Fronteira conjunta (8 epocas, lr 8e-5)"),
)


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
        default=Path("results/grid_search_expansion_results.csv"),
        help="CSV de saída para as novas configurações.",
    )
    parser.add_argument(
        "--original-results",
        type=Path,
        default=Path("results/grid_search_results.csv"),
        help="CSV dos resultados do Grid Search original para comparação e unificação.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=Path("results/grid_search_combined_results.csv"),
        help="CSV com as 24 configurações originais + as 3 novas combinadas em ranking único.",
    )
    parser.add_argument("--sample-size", type=int, default=GRID_SEARCH_SAMPLE)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=2,
        help="Microlote por dispositivo; a acumulação reproduz o lote efetivo 16.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permite rodar em CPU se GPU não estiver disponível (apenas para testes).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite substituir arquivos de saída existentes.",
    )
    return parser.parse_args()


def display_decision_summary(best_new_f1, best_new_config, benchmark_f1=BENCHMARK_F1):
    print("\n" + "=" * 80)
    print("ANÁLISE DE DECISÃO PARA O TCC (SEÇÃO 5.1.2)")
    print("=" * 80)
    print(f"Melhor F1-macro anterior (5 ep, lote 16, lr 5e-5, wu 500): {benchmark_f1 * 100:.2f}%")
    print(f"Melhor F1-macro das novas configurações:                 {best_new_f1 * 100:.2f}%")
    print(f"Configuração vencedora da expansão:                      {best_new_config}")

    diff = (best_new_f1 - benchmark_f1) * 100
    print(f"Diferença relativa:                                      {diff:+.2f}%")

    if best_new_f1 <= benchmark_f1:
        print("\n-> CONCLUSÃO: NENHUMA NOVA CONFIGURAÇÃO SUPEROU A ANTERIOR.")
        print("  1. NÃO É NECESSÁRIO refazer a validação cruzada de 5 folds (4,1 horas).")
        print("  2. Basta acrescentar as 3 linhas à Tabela 4 do TCC.")
        print("  3. Justifique no texto que 8 épocas gerou overfitting e/ou 8e-5 causou")
        print("     instabilidade, provando que a configuração (5 ep, 5e-5) era o topo real.")
    elif diff <= 0.5:
        print("\n-> CONCLUSÃO: SUPEROU POR MARGEM MUITO PEQUENA (<= 0,5%).")
        print("  1. Conforme a regra do orientador ('ou superar por pouco'),")
        print("     basta acrescentar as novas linhas à Tabela 4.")
        print("  2. Não é mandatório refazer a validação cruzada de 4,1 horas; argumente")
        print("     no texto que o ganho residual não justifica o custo computacional.")
    else:
        print("\n-> CONCLUSÃO: UMA NOVA CONFIGURAÇÃO VENCEU COM FOLGA SIGNIFICATIVA (> 0,5%).")
        print("  1. Recomenda-se refazer a validação cruzada (cerca de 4,1 horas) com")
        print("     esta nova configuração para maximizar a qualidade final do modelo.")
    print("=" * 80 + "\n")


def main():
    args = parse_args()
    has_gpu = torch.cuda.is_available()

    if not has_gpu and not args.allow_cpu:
        raise RuntimeError(
            "A busca completa requer GPU CUDA. No Colab, selecione Ambiente de execução > "
            "Alterar tipo de ambiente de execução > GPU. Se estiver testando localmente, "
            "use --allow-cpu."
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

    device = torch.device("cuda" if has_gpu else "cpu")
    print("=" * 80)
    print("JURISCLASS AI - EXPANSÃO DE FRONTEIRA DO GRID SEARCH")
    print("=" * 80)
    if has_gpu:
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {properties.total_memory / 1024**3:.1f} GiB")
    else:
        print("Executando em CPU (--allow-cpu ativado)")

    print(f"Treino: {train_path}")
    print(f"Validação: {validation_path}")
    print(f"Pesos: {weights_path}")
    print(f"Benchmark a ser superado: {BENCHMARK_F1 * 100:.2f}% de F1-macro")

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

    # Amostragem idêntica à do grid_search.py original
    if len(df_train) > args.sample_size:
        df_train, _ = train_test_split(
            df_train,
            train_size=args.sample_size,
            stratify=df_train["document_type"],
            random_state=SEED,
        )

    print(f"\nAmostra de treino: {len(df_train)} páginas (estratificada, seed={SEED})")
    print(df_train["document_type"].value_counts().sort_index().to_string())
    print(f"Validação completa: {len(df_validation)} páginas\n")

    label2id = {label: index for index, label in enumerate(classes)}
    id2label = {index: label for index, label in enumerate(classes)}
    class_weights = torch.tensor(
        [class_weights_dict[label] for label in classes], dtype=torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizando treino e validação (max_length=512)...")
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

    print(f"Iniciando treinamento das {len(NEW_CONFIGURATIONS)} novas combinações de fronteira...\n")
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_start_time = time.time()

    for number, (epochs, effective_batch, learning_rate, warmup, desc) in enumerate(NEW_CONFIGURATIONS, 1):
        set_seed(SEED)
        micro_batch = min(args.micro_batch_size, effective_batch)
        if effective_batch % micro_batch:
            raise ValueError(
                f"O lote {effective_batch} não é divisível pelo microlote {micro_batch}."
            )
        accumulation = effective_batch // micro_batch

        print("-" * 80)
        print(f"[{number}/{len(NEW_CONFIGURATIONS)}] {desc.upper()}")
        print(
            f"Parametros: epocas={epochs}, lote={effective_batch}, "
            f"lr={learning_rate:.1e}, warmup={warmup}, "
            f"microlote={micro_batch}, acumulacao={accumulation}"
        )
        print("-" * 80)

        run_output_dir = Path("results_gs_expansion") / f"ep{epochs}_bs{effective_batch}_lr{learning_rate}_wu{warmup}"

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(classes),
            id2label=id2label,
            label2id=label2id,
        ).to(device)

        training_args = TrainingArguments(
            output_dir=str(run_output_dir),
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
            fp16=has_gpu,
            gradient_checkpointing=has_gpu,
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

        run_start = time.time()
        trainer.train()
        metrics = trainer.evaluate()
        run_duration = time.time() - run_start

        results.append({
            "epochs": epochs,
            "batch_size": effective_batch,
            "learning_rate": learning_rate,
            "warmup_steps": warmup,
            "val_loss": metrics["eval_loss"],
            "val_accuracy": metrics["eval_accuracy"],
            "val_f1_macro": metrics["eval_f1_macro"],
            "tempo_segundos": round(run_duration, 2),
            "tempo_minutos": round(run_duration / 60, 2),
        })

        df_partial = pd.DataFrame(results).sort_values("val_f1_macro", ascending=False)
        df_partial.to_csv(args.output, index=False)
        print(
            f"-> Concluído em {run_duration / 60:.2f} min | "
            f"F1-macro: {metrics['eval_f1_macro'] * 100:.2f}% | "
            f"Acurácia: {metrics['eval_accuracy'] * 100:.2f}% | "
            f"Perda: {metrics['eval_loss']:.4f}"
        )

        del trainer, model
        gc.collect()
        if has_gpu:
            torch.cuda.empty_cache()

    total_duration = time.time() - total_start_time
    df_expansion = pd.DataFrame(results).sort_values("val_f1_macro", ascending=False)
    df_expansion.to_csv(args.output, index=False)

    print("\n" + "=" * 80)
    print("EXPANSÃO CONCLUÍDA COM SUCESSO!")
    print(f"Tempo total decorrido: {total_duration / 60:.2f} minutos ({total_duration:.1f} segundos)")
    print(f"Resultados das 3 novas configurações salvos em: {args.output}")
    print("=" * 80)
    print(df_expansion.to_string(index=False))

    # Mescla com os resultados originais se o arquivo existir
    if args.original_results.is_file():
        print(f"\nMesclando com resultados originais de {args.original_results}...")
        df_orig = pd.read_csv(args.original_results)
        
        # Garante colunas de compatibilidade
        cols_base = ["epochs", "batch_size", "learning_rate", "warmup_steps", "val_loss", "val_accuracy", "val_f1_macro"]
        df_orig_base = df_orig[cols_base].copy()
        df_exp_base = df_expansion[cols_base].copy()

        df_combined = pd.concat([df_orig_base, df_exp_base], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=["epochs", "batch_size", "learning_rate", "warmup_steps"]
        ).sort_values("val_f1_macro", ascending=False)

        args.combined_output.parent.mkdir(parents=True, exist_ok=True)
        df_combined.to_csv(args.combined_output, index=False)
        print(f"Ranking consolidado salvo em: {args.combined_output}")
        print("\nTop 7 configurações consolidadas (originais + novas):")
        print(df_combined.head(7).to_string(index=False))

    # Relatório de decisão final para o TCC
    best_row = df_expansion.iloc[0]
    best_config_str = (
        f"ep={int(best_row['epochs'])}, lote={int(best_row['batch_size'])}, "
        f"lr={best_row['learning_rate']}, warmup={int(best_row['warmup_steps'])}"
    )
    display_decision_summary(best_row["val_f1_macro"], best_config_str, BENCHMARK_F1)


if __name__ == "__main__":
    main()
