import pandas as pd
import numpy as np
import torch
import json
import time
import os
import gc
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torch.utils.data import Dataset
from torch.nn.functional import softmax
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
DRY_RUN = False  # True = amostra pequena para validar o pipeline
MODEL_NAME = "raquelsilveira/legalbertpt_fp"
# Treino sempre no disco local rápido — sem I/O lento pro Drive durante o treino.
OUTPUT_DIR = "/content/results/results_cv" if os.path.isdir("/content") else "./results/results_cv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Backup pro Drive só no final de cada fold (evita erros de I/O sob carga de GPU).
DRIVE_BACKUP_DIR = None
if os.path.isdir("/content/drive/MyDrive"):
    DRIVE_BACKUP_DIR = "/content/drive/MyDrive/tcc_cross_validation/results_cv"
    os.makedirs(DRIVE_BACKUP_DIR, exist_ok=True)
    print(f"[INFO] Drive montado — backup pós-fold em: {DRIVE_BACKUP_DIR}")
else:
    print("[INFO] Drive não detectado — salvando apenas localmente.")

# Melhores hiperparâmetros do Grid Search (grid_search_results.csv, linha 1)
BEST_EPOCHS = 5
BEST_BATCH_SIZE = 16
BEST_LR = 5e-5
BEST_WARMUP = 500

# Cross-Validation
N_SPLITS = 5
RANDOM_STATE = 42
START_FROM_FOLD = 1  # Mude para retomar de um fold específico (ex: 3 = pula folds 1 e 2 já concluídos)

# ==============================================================================
# DATASET
# ==============================================================================
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

# ==============================================================================
# MÉTRICAS
# ==============================================================================
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average='macro')
    return {"accuracy": acc, "f1_macro": f1}

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def plot_confusion_matrices(y_true, y_pred, labels_abrev, title_suffix, save_path):
    """Gera e salva matrizes de confusão (absoluta e normalizada)."""
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Absoluto
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=labels_abrev, yticklabels=labels_abrev, ax=axes[0])
    axes[0].set_title(f'Matriz de Confusão — {title_suffix} (Absoluto)')
    axes[0].set_xlabel('Predito')
    axes[0].set_ylabel('Real')

    # Normalizado
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Purples',
                xticklabels=labels_abrev, yticklabels=labels_abrev, ax=axes[1])
    axes[1].set_title(f'Matriz de Confusão — {title_suffix} (Normalizada)')
    axes[1].set_xlabel('Predito')
    axes[1].set_ylabel('Real')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Matriz de confusão salva em: {save_path}")


def evaluate_and_report(y_true, y_pred, labels_names, labels_abrev, title, save_prefix):
    """Calcula métricas, imprime relatório e salva resultados."""
    acc = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average='macro')
    f1_w = f1_score(y_true, y_pred, average='weighted')
    prec_m = precision_score(y_true, y_pred, average='macro')
    rec_m = recall_score(y_true, y_pred, average='macro')

    print(f"\n{'=' * 60}")
    print(f" RESULTADOS NO TESTE — {title}")
    print(f"{'=' * 60}")
    print(f"\n  Acurácia:        {acc:.4f}")
    print(f"  Precisão Macro:  {prec_m:.4f}")
    print(f"  Recall Macro:    {rec_m:.4f}")
    print(f"  F1 Macro:        {f1_m:.4f}")
    print(f"  F1 Weighted:     {f1_w:.4f}\n")

    print(classification_report(y_true, y_pred, target_names=labels_names, digits=4))

    # Métricas por classe
    precision_per_class = precision_score(y_true, y_pred, labels=list(range(len(labels_names))), average=None)
    recall_per_class = recall_score(y_true, y_pred, labels=list(range(len(labels_names))), average=None)
    f1_per_class = f1_score(y_true, y_pred, labels=list(range(len(labels_names))), average=None)

    df_metrics = pd.DataFrame({
        'Classe': labels_names,
        'Precisão': precision_per_class,
        'Recall': recall_per_class,
        'F1-Score': f1_per_class,
        'Suporte': [sum(y_true == i) for i in range(len(labels_names))]
    })
    df_metrics.to_csv(f'{save_prefix}_per_class.csv', index=False)
    print(f"  Métricas por classe salvas em: {save_prefix}_per_class.csv")

    # Matriz de confusão
    plot_confusion_matrices(y_true, y_pred, labels_abrev, title,
                            f'{save_prefix}_confusion_matrix.png')

    # Resumo geral
    results_summary = {
        'modelo': title,
        'accuracy_test': acc,
        'precision_macro_test': prec_m,
        'recall_macro_test': rec_m,
        'f1_macro_test': f1_m,
        'f1_weighted_test': f1_w
    }
    pd.DataFrame([results_summary]).to_csv(f'{save_prefix}_results.csv', index=False)

    return results_summary


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print(f" CROSS-VALIDATION (5-FOLD) — {MODEL_NAME}")
    print("=" * 60)
    print(f"\n  Hiperparâmetros (Grid Search):")
    print(f"    Epochs:       {BEST_EPOCHS}")
    print(f"    Batch Size:   {BEST_BATCH_SIZE}")
    print(f"    Learning Rate:{BEST_LR}")
    print(f"    Warmup Steps: {BEST_WARMUP}")
    print(f"  DRY_RUN: {DRY_RUN}\n")

    # =========================================================================
    # 1. CARREGAR DADOS
    # =========================================================================
    print("[1/7] Carregando dados...")
    df_train_full = pd.read_csv("dataset_clean/train_clean.csv")
    df_test = pd.read_csv("dataset_clean/test_clean.csv")

    if DRY_RUN:
        df_train_full = df_train_full.head(200)
        df_test = df_test.head(100)

    with open("results/class_weights.json", "r") as f:
        class_weights_dict = json.load(f)

    classes = sorted(list(class_weights_dict.keys()))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for i, c in enumerate(classes)}
    num_labels = len(classes)

    labels_names = [id2label[i] for i in range(num_labels)]
    labels_abrev = ['Acórdão', 'ARE', 'Despacho', 'RE', 'Sentença']

    weights_list = [class_weights_dict[c] for c in classes]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(weights_list, dtype=torch.float).to(device)

    print(f"  Treino (completo): {len(df_train_full)} | Teste: {len(df_test)}")
    print(f"  Device: {device}")
    print(f"  Classes: {classes}")
    print(f"  Pesos:   {weights_list}")

    # =========================================================================
    # 2. TOKENIZAÇÃO DO CONJUNTO DE TESTE (feita uma única vez)
    # =========================================================================
    print(f"\n[2/7] Inicializando Tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("  Tokenizando conjunto de TESTE...")
    test_encodings = tokenizer(df_test['body'].tolist(), truncation=True, padding=True, max_length=512)
    test_labels = [label2id[label] for label in df_test['document_type']]
    test_dataset = LegalDataset(test_encodings, test_labels)

    # =========================================================================
    # 3. WEIGHTED TRAINER
    # =========================================================================
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # =========================================================================
    # 4. STRATIFIED K-FOLD CROSS-VALIDATION
    # =========================================================================
    print(f"\n[3/7] Iniciando Stratified {N_SPLITS}-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    X_train_full = df_train_full['body'].tolist()
    y_train_full = df_train_full['document_type'].tolist()
    y_train_full_ids = [label2id[label] for label in y_train_full]

    fold_results = []
    fold_model_paths = []

    # Recuperar resultados de folds anteriores (se retomando de sessão nova)
    cv_results_path = os.path.join(OUTPUT_DIR, "cv_fold_results.csv")
    if START_FROM_FOLD > 1:
        # Tenta disco local primeiro, senão busca do Drive
        if not os.path.exists(cv_results_path) and DRIVE_BACKUP_DIR:
            drive_csv = os.path.join(DRIVE_BACKUP_DIR, "cv_fold_results.csv")
            if os.path.exists(drive_csv):
                shutil.copy2(drive_csv, cv_results_path)
                print(f"  [RESUME] CSV recuperado do Drive.")
        if os.path.exists(cv_results_path):
            df_prev = pd.read_csv(cv_results_path)
            fold_results = df_prev.to_dict('records')
            print(f"  Recuperados {len(fold_results)} folds anteriores do CSV.")

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X_train_full, y_train_full)):
        fold_num = fold_idx + 1
        fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_num}")
        best_model_path = os.path.join(fold_output_dir, "best_model")

        # Para folds pulados: se modelo não existe localmente, restaura do Drive
        if fold_num < START_FROM_FOLD:
            if not os.path.isdir(best_model_path) and DRIVE_BACKUP_DIR:
                drive_model = os.path.join(DRIVE_BACKUP_DIR, f"fold_{fold_num}", "best_model")
                if os.path.isdir(drive_model):
                    os.makedirs(best_model_path, exist_ok=True)
                    for fname in os.listdir(drive_model):
                        shutil.copy2(os.path.join(drive_model, fname), os.path.join(best_model_path, fname))
                    print(f"  [RESUME] Modelo fold {fold_num} restaurado do Drive.")
        fold_model_paths.append(best_model_path)

        # Pular folds já concluídos
        if fold_num < START_FROM_FOLD:
            print(f"\n  ⏭ Fold {fold_num}/{N_SPLITS} — PULADO (já concluído)")
            continue

        print(f"\n{'─' * 60}")
        print(f"  FOLD {fold_num}/{N_SPLITS}")
        print(f"{'─' * 60}")
        print(f"  Treino: {len(train_indices)} amostras | Val: {len(val_indices)} amostras")

        os.makedirs(fold_output_dir, exist_ok=True)

        # Tokenizar fold de treino e validação
        print(f"  Tokenizando fold {fold_num}...")
        fold_train_texts = [X_train_full[i] for i in train_indices]
        fold_val_texts = [X_train_full[i] for i in val_indices]
        fold_train_labels = [y_train_full_ids[i] for i in train_indices]
        fold_val_labels = [y_train_full_ids[i] for i in val_indices]

        fold_train_encodings = tokenizer(fold_train_texts, truncation=True, padding=True, max_length=512)
        fold_val_encodings = tokenizer(fold_val_texts, truncation=True, padding=True, max_length=512)

        fold_train_dataset = LegalDataset(fold_train_encodings, fold_train_labels)
        fold_val_dataset = LegalDataset(fold_val_encodings, fold_val_labels)

        # Carregar modelo do zero
        print(f"  Carregando modelo {MODEL_NAME} (fold {fold_num})...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        ).to(device)

        # Ajuste de hardware (gradient accumulation para GPUs com pouca VRAM)
        micro_batch_size = 2
        grad_accum_steps = max(1, BEST_BATCH_SIZE // micro_batch_size)

        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=BEST_EPOCHS,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size * 2,
            gradient_accumulation_steps=grad_accum_steps,
            learning_rate=BEST_LR,
            warmup_steps=BEST_WARMUP,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",           # SEM checkpoints intermediários no disco
            logging_steps=100,
            load_best_model_at_end=False,  # Não depende de checkpoints em disco
            fp16=True if torch.cuda.is_available() else False,
            gradient_checkpointing=True,
            report_to="none",
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=fold_train_dataset,
            eval_dataset=fold_val_dataset,
            compute_metrics=compute_metrics
        )

        # Treinar
        print(f"  Treinando fold {fold_num}...")
        t0 = time.time()
        trainer.train()
        elapsed = (time.time() - t0) / 60
        print(f"  Fold {fold_num} treinado em {elapsed:.1f} minutos")

        # Avaliar no fold de validação
        eval_metrics = trainer.evaluate()
        val_acc = eval_metrics.get("eval_accuracy", 0)
        val_f1 = eval_metrics.get("eval_f1_macro", 0)
        val_loss = eval_metrics.get("eval_loss", 0)

        print(f"  Fold {fold_num} — Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1 Macro: {val_f1:.4f}")

        fold_results.append({
            'fold': fold_num,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1_macro': val_f1,
            'train_time_min': elapsed
        })

        # Salvar APENAS o modelo final do fold (sem checkpoints intermediários)
        os.makedirs(best_model_path, exist_ok=True)
        trainer.save_model(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        print(f"  Modelo final salvo em: {best_model_path}")

        # Limpar qualquer lixo de checkpoint que o Trainer tenha criado
        for item in os.listdir(fold_output_dir):
            item_path = os.path.join(fold_output_dir, item)
            if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                shutil.rmtree(item_path)

        # Salvar resultados parciais localmente
        pd.DataFrame(fold_results).to_csv(cv_results_path, index=False)
        print(f"  Resultados parciais salvos em: {cv_results_path}")

        # Copiar pro Drive agora que o fold terminou (GPU ociosa = sem conflito de I/O)
        if DRIVE_BACKUP_DIR:
            try:
                drive_fold_path = os.path.join(DRIVE_BACKUP_DIR, f"fold_{fold_num}", "best_model")
                os.makedirs(drive_fold_path, exist_ok=True)
                for fname in os.listdir(best_model_path):
                    src = os.path.join(best_model_path, fname)
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(drive_fold_path, fname))
                shutil.copy2(cv_results_path, os.path.join(DRIVE_BACKUP_DIR, "cv_fold_results.csv"))
                print(f"  [BACKUP] Fold {fold_num} copiado pro Drive com sucesso.")
            except Exception as e:
                print(f"  [BACKUP] Falha ao copiar pro Drive (fold continua salvo localmente): {e}")

        # Liberar memória
        del model, trainer, fold_train_dataset, fold_val_dataset
        del fold_train_encodings, fold_val_encodings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # 5. RESUMO DA CROSS-VALIDATION
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f" RESUMO DA {N_SPLITS}-FOLD CROSS-VALIDATION")
    print(f"{'=' * 60}")

    df_cv = pd.DataFrame(fold_results)
    df_cv.to_csv(os.path.join(OUTPUT_DIR, "cv_fold_results.csv"), index=False)

    mean_acc = df_cv['val_accuracy'].mean()
    std_acc = df_cv['val_accuracy'].std()
    mean_f1 = df_cv['val_f1_macro'].mean()
    std_f1 = df_cv['val_f1_macro'].std()
    mean_loss = df_cv['val_loss'].mean()
    std_loss = df_cv['val_loss'].std()

    print(f"\n  Val Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Val F1 Macro: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Val Loss:     {mean_loss:.4f} ± {std_loss:.4f}")

    for _, row in df_cv.iterrows():
        print(f"    Fold {int(row['fold'])}: Acc={row['val_accuracy']:.4f} | F1={row['val_f1_macro']:.4f} | Loss={row['val_loss']:.4f} | Tempo={row['train_time_min']:.1f}min")

    # Identificar o melhor fold
    best_fold_idx = df_cv['val_f1_macro'].idxmax()
    best_fold_num = int(df_cv.loc[best_fold_idx, 'fold'])
    best_fold_f1 = df_cv.loc[best_fold_idx, 'val_f1_macro']
    print(f"\n  ★ Melhor Fold: {best_fold_num} (F1 Macro = {best_fold_f1:.4f})")

    # =========================================================================
    # 6. AVALIAÇÃO NO TESTE — BEST FOLD
    # =========================================================================
    print(f"\n[4/7] Avaliando BEST FOLD (Fold {best_fold_num}) no conjunto de Teste...")

    best_model = AutoModelForSequenceClassification.from_pretrained(
        fold_model_paths[best_fold_idx]
    ).to(device)

    best_trainer = Trainer(
        model=best_model,
        compute_metrics=compute_metrics
    )

    best_predictions = best_trainer.predict(test_dataset)
    y_test_pred_best = np.argmax(best_predictions.predictions, axis=1)
    y_test_true = best_predictions.label_ids

    best_fold_results = evaluate_and_report(
        y_test_true, y_test_pred_best,
        labels_names, labels_abrev,
        f"Best Fold (Fold {best_fold_num})",
        os.path.join(OUTPUT_DIR, "best_fold")
    )

    del best_model, best_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # 7. AVALIAÇÃO NO TESTE — ENSEMBLE (SOFT VOTING)
    # =========================================================================
    print(f"\n[5/7] Avaliando ENSEMBLE (Soft Voting) com {N_SPLITS} modelos no conjunto de Teste...")

    all_probs = []

    for fold_i, model_path in enumerate(fold_model_paths):
        print(f"  Carregando modelo do fold {fold_i + 1}...")
        fold_model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

        fold_trainer = Trainer(
            model=fold_model,
            compute_metrics=compute_metrics
        )

        fold_predictions = fold_trainer.predict(test_dataset)
        # Converter logits para probabilidades (softmax)
        logits_tensor = torch.tensor(fold_predictions.predictions)
        probs = softmax(logits_tensor, dim=1).numpy()
        all_probs.append(probs)

        del fold_model, fold_trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Média das probabilidades (Soft Voting)
    ensemble_probs = np.mean(all_probs, axis=0)
    y_test_pred_ensemble = np.argmax(ensemble_probs, axis=1)

    ensemble_results = evaluate_and_report(
        y_test_true, y_test_pred_ensemble,
        labels_names, labels_abrev,
        "Ensemble (Soft Voting)",
        os.path.join(OUTPUT_DIR, "ensemble")
    )

    # =========================================================================
    # 8. COMPARAÇÃO FINAL
    # =========================================================================
    print(f"\n[6/7] Gerando comparação final...")

    comparison = pd.DataFrame([best_fold_results, ensemble_results])
    comparison.to_csv(os.path.join(OUTPUT_DIR, "comparison_best_fold_vs_ensemble.csv"), index=False)

    print(f"\n{'=' * 60}")
    print(f" COMPARAÇÃO FINAL: BEST FOLD vs ENSEMBLE")
    print(f"{'=' * 60}")
    print(f"\n  {'Métrica':<20} {'Best Fold':>12} {'Ensemble':>12}")
    print(f"  {'─' * 44}")
    print(f"  {'Acurácia':<20} {best_fold_results['accuracy_test']:>12.4f} {ensemble_results['accuracy_test']:>12.4f}")
    print(f"  {'Precisão Macro':<20} {best_fold_results['precision_macro_test']:>12.4f} {ensemble_results['precision_macro_test']:>12.4f}")
    print(f"  {'Recall Macro':<20} {best_fold_results['recall_macro_test']:>12.4f} {ensemble_results['recall_macro_test']:>12.4f}")
    print(f"  {'F1 Macro':<20} {best_fold_results['f1_macro_test']:>12.4f} {ensemble_results['f1_macro_test']:>12.4f}")
    print(f"  {'F1 Weighted':<20} {best_fold_results['f1_weighted_test']:>12.4f} {ensemble_results['f1_weighted_test']:>12.4f}")

    # =========================================================================
    # 9. SALVAR RESUMO GERAL
    # =========================================================================
    print(f"\n[7/7] Salvando resumo geral...")

    summary = {
        'modelo': MODEL_NAME,
        'epochs': BEST_EPOCHS,
        'batch_size': BEST_BATCH_SIZE,
        'learning_rate': BEST_LR,
        'warmup_steps': BEST_WARMUP,
        'n_splits': N_SPLITS,
        'cv_mean_f1_macro': mean_f1,
        'cv_std_f1_macro': std_f1,
        'cv_mean_accuracy': mean_acc,
        'cv_std_accuracy': std_acc,
        'best_fold': best_fold_num,
        'best_fold_val_f1': best_fold_f1,
        'best_fold_test_f1_macro': best_fold_results['f1_macro_test'],
        'best_fold_test_accuracy': best_fold_results['accuracy_test'],
        'ensemble_test_f1_macro': ensemble_results['f1_macro_test'],
        'ensemble_test_accuracy': ensemble_results['accuracy_test'],
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, "cv_summary.csv"), index=False)

    print(f"\n  Todos os resultados salvos em: {OUTPUT_DIR}/")
    print("  Arquivos gerados:")
    print("    - cv_fold_results.csv            (métricas por fold)")
    print("    - cv_summary.csv                 (resumo geral)")
    print("    - best_fold_results.csv          (resultado Best Fold no teste)")
    print("    - best_fold_per_class.csv        (métricas por classe - Best Fold)")
    print("    - best_fold_confusion_matrix.png (matriz de confusão - Best Fold)")
    print("    - ensemble_results.csv           (resultado Ensemble no teste)")
    print("    - ensemble_per_class.csv         (métricas por classe - Ensemble)")
    print("    - ensemble_confusion_matrix.png  (matriz de confusão - Ensemble)")
    print("    - comparison_best_fold_vs_ensemble.csv")

    print(f"\n{'=' * 60}")
    print(" CROSS-VALIDATION CONCLUÍDA COM SUCESSO!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
