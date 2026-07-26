import pandas as pd
import numpy as np
import json
import torch
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EvalPrediction
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ==============================================================================
# CONFIGURAÇÕES DO BASELINE
# ==============================================================================
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
OUTPUT_DIR = "./results/bertimbau_base"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    print("=" * 60)
    print(f" BASELINE NEURAL: {MODEL_NAME}")
    print("=" * 60)

    # 1. Carregar Dados
    print("\n[1/6] Carregando dados...")
    df_train = pd.read_csv("dataset_clean/train_clean.csv")
    df_val   = pd.read_csv("dataset_clean/val_clean.csv")
    df_test  = pd.read_csv("dataset_clean/test_clean.csv")

    with open("results/class_weights.json", "r") as f:
        class_weights_dict = json.load(f)

    classes = sorted(list(class_weights_dict.keys()))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for i, c in enumerate(classes)}

    weights_list = [class_weights_dict[c] for c in classes]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(weights_list, dtype=torch.float).to(device)

    print(f"Treino: {len(df_train)} | Val: {len(df_val)} | Teste: {len(df_test)}")
    print(f"Device: {device}")

    # 2. Tokenização
    print(f"\n[2/6] Inicializando Tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)

    print("Tokenizando os textos (Isso pode demorar alguns minutos)...")
    train_encodings = tokenizer(df_train['body'].tolist(), truncation=True, padding=True, max_length=512)
    val_encodings   = tokenizer(df_val['body'].tolist(), truncation=True, padding=True, max_length=512)
    test_encodings  = tokenizer(df_test['body'].tolist(), truncation=True, padding=True, max_length=512)

    train_labels = [label2id[label] for label in df_train['document_type']]
    val_labels   = [label2id[label] for label in df_val['document_type']]
    test_labels  = [label2id[label] for label in df_test['document_type']]

    train_dataset = LegalDataset(train_encodings, train_labels)
    val_dataset   = LegalDataset(val_encodings, val_labels)
    test_dataset  = LegalDataset(test_encodings, test_labels)

    # 3. Preparar Modelo
    print(f"\n[3/6] Carregando Modelo ({MODEL_NAME})...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # Custom Trainer com Pesos
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # Hiperparâmetros Padrões (Default)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16, # Ajuste conforme GPU, 16 é um bom padrão pra colab T4
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=True if torch.cuda.is_available() else False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 4. Treinamento
    print("\n[4/6] Iniciando Treinamento (Fine-Tuning)...")
    t0 = time.time()
    trainer.train()
    print(f"Treinamento concluído em {(time.time()-t0)/60:.1f} minutos")

    # 5. Avaliação Final no Teste
    print("\n[5/6] Avaliando modelo no conjunto de Teste...")
    predictions = trainer.predict(test_dataset)
    y_test_pred = np.argmax(predictions.predictions, axis=1)
    y_test_true = predictions.label_ids

    acc  = accuracy_score(y_test_true, y_test_pred)
    f1_m = f1_score(y_test_true, y_test_pred, average='macro')
    f1_w = f1_score(y_test_true, y_test_pred, average='weighted')

    print("\n" + "=" * 50)
    print(" RESULTADOS FINAIS NO TESTE (BERTIMBAU) ")
    print("=" * 50)
    print(f"\nAcurácia:    {acc:.4f}")
    print(f"F1 Macro:    {f1_m:.4f}")
    print(f"F1 Weighted: {f1_w:.4f}\n")
    
    from sklearn.metrics import precision_score, recall_score
    labels_names = [id2label[i] for i in range(len(classes))]
    print(classification_report(y_test_true, y_test_pred, target_names=labels_names, digits=4))

    precision_per_class = precision_score(y_test_true, y_test_pred, labels=list(range(len(classes))), average=None)
    recall_per_class    = recall_score(y_test_true, y_test_pred, labels=list(range(len(classes))), average=None)
    f1_per_class        = f1_score(y_test_true, y_test_pred, labels=list(range(len(classes))), average=None)

    df_metrics = pd.DataFrame({
        'Classe': labels_names,
        'Precisão': precision_per_class,
        'Recall': recall_per_class,
        'F1-Score': f1_per_class,
        'Suporte': [sum(y_test_true == i) for i in range(len(classes))]
    })

    # 6. Salvar Resultados
    print("\n[6/6] Salvando matriz de confusão e relatórios...")
    cm = confusion_matrix(y_test_true, y_test_pred)
    labels_abrev = ['Acórdão', 'ARE', 'Despacho', 'RE', 'Sentença']

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Absoluto
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=labels_abrev, yticklabels=labels_abrev, ax=axes[0])
    axes[0].set_title(f'Matriz de Confusão — {MODEL_NAME} (Absoluto)')
    
    # Normalizado
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Purples', xticklabels=labels_abrev, yticklabels=labels_abrev, ax=axes[1])
    axes[1].set_title(f'Matriz de Confusão — {MODEL_NAME} (Normalizada)')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_bertimbau.png', dpi=150)
    plt.close()

    results_summary = {
        'modelo': 'BERTimbau Base',
        'epochs': 3,
        'batch_size': 16,
        'learning_rate': '5e-5',
        'accuracy_test': acc,
        'f1_macro_test': f1_m,
        'f1_weighted_test': f1_w
    }
    pd.DataFrame([results_summary]).to_csv(f'{OUTPUT_DIR}/bertimbau_results.csv', index=False)
    df_metrics.to_csv(f'{OUTPUT_DIR}/bertimbau_per_class.csv', index=False)
    
    print("Tudo concluído com sucesso!")

if __name__ == '__main__':
    main()
