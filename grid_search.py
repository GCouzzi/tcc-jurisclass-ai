import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EvalPrediction
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import json
import os
import itertools
import gc

DRY_RUN = False  # Mude para False quando for rodar de verdade
GRID_SEARCH_SAMPLE = 3000  # Tamanho da amostra estratificada para o Grid Search (None = usar tudo)

# Dataset customizado para o Hugging Face Trainer
class LegalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average='macro')
    return {"accuracy": acc, "f1_macro": f1}


def main():
    # 1. Carregar mapeamento de classes e pesos
    print("Carregando dados...")
    df_train = pd.read_csv("train_clean.csv")
    df_val = pd.read_csv("val_clean.csv")

    if DRY_RUN:
        df_train = df_train.head(100)
        df_val = df_val.head(50)
    elif GRID_SEARCH_SAMPLE and len(df_train) > GRID_SEARCH_SAMPLE:
        # Amostragem estratificada: reduz o dataset mantendo a proporção das classes
        df_train, _ = train_test_split(
            df_train, 
            train_size=GRID_SEARCH_SAMPLE, 
            stratify=df_train['document_type'], 
            random_state=42
        )
        print(f"Amostra estratificada: {len(df_train)} textos de treino (de {GRID_SEARCH_SAMPLE} solicitados)")
        print(f"Distribuição por classe:\n{df_train['document_type'].value_counts()}\n")

    with open("class_weights.json", "r") as f:
        class_weights_dict = json.load(f)

    # Criar label2id e id2label consistentes
    classes = sorted(list(class_weights_dict.keys()))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for i, c in enumerate(classes)}

    # Converter pesos para tensor (na ordem dos IDs)
    weights_list = [class_weights_dict[c] for c in classes]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(weights_list, dtype=torch.float).to(device)

    print(f"Usando device: {device}")

    # 2. Tokenização Antecipada (para não re-tokenizar em toda iteração do Grid Search)
    MODEL_NAME = "raquelsilveira/legalbertpt_fp"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Custom Trainer para Class Weights (precisa acessar class_weights_tensor)
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    print("Tokenizando dados (isso pode levar alguns minutos)...")
    train_encodings = tokenizer(df_train['body'].tolist(), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(df_val['body'].tolist(), truncation=True, padding=True, max_length=512)

    train_labels = [label2id[label] for label in df_train['document_type']]
    val_labels = [label2id[label] for label in df_val['document_type']]

    train_dataset = LegalDataset(train_encodings, train_labels)
    val_dataset = LegalDataset(val_encodings, val_labels)

    # 4. Busca em Grade (Grid Search)
    epochs_list = [3, 5]
    batch_size_list = [8, 16, 32]
    lr_list = [2e-5, 5e-5]
    warmup_list = [100, 500]

    if DRY_RUN:
        epochs_list = [1]
        batch_size_list = [8]
        lr_list = [5e-5]
        warmup_list = [100]

    combinations = list(itertools.product(epochs_list, batch_size_list, lr_list, warmup_list))
    print(f"Iniciando Grid Search com {len(combinations)} combinações...")

    results = []

    for i, (ep, bs, lr, wu) in enumerate(combinations):
        print(f"\n--- Iteração {i+1}/{len(combinations)} ---")
        print(f"Epochs: {ep} | Batch: {bs} | LR: {lr} | Warmup: {wu}")
        
        # Recarregar o modelo do zero para não acumular pesos
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=len(classes),
            id2label=id2label,
            label2id=label2id
        ).to(device)

        # Ajuste de Hardware para Placa de Vídeo com 4GB VRAM (GTX 1650 Ti)
        # Usamos um "micro_batch_size" de 2, e simulamos o tamanho do lote real acumulando gradientes.
        micro_batch_size = 2
        grad_accum_steps = max(1, bs // micro_batch_size)
        
        if bs < micro_batch_size:
            micro_batch_size = bs

        training_args = TrainingArguments(
            output_dir=f"./results_gs/ep{ep}_bs{bs}_lr{lr}_wu{wu}",
            num_train_epochs=ep,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size * 2,
            gradient_accumulation_steps=grad_accum_steps,
            learning_rate=lr,
            warmup_steps=wu,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            disable_tqdm=False,
            report_to="none",
            fp16=True,
            gradient_checkpointing=True,
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        
        # Avaliar no conjunto de validação
        eval_metrics = trainer.evaluate()
        
        results.append({
            "epochs": ep,
            "batch_size": bs,
            "learning_rate": lr,
            "warmup_steps": wu,
            "val_loss": eval_metrics.get("eval_loss", 0),
            "val_accuracy": eval_metrics.get("eval_accuracy", 0),
            "val_f1_macro": eval_metrics.get("eval_f1_macro", 0)
        })
        
        # Liberar memória da GPU
        del model
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 5. Salvar resultados
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="val_f1_macro", ascending=False)
    df_results.to_csv("grid_search_results.csv", index=False)

    print("\n=== GRID SEARCH FINALIZADO ===")
    print("Melhor configuração encontrada:")
    print(df_results.iloc[0])
    print("\nOs resultados completos foram salvos em 'grid_search_results.csv'.")


# OBRIGATÓRIO no Windows para evitar erro de multiprocessing/fork
if __name__ == '__main__':
    main()
