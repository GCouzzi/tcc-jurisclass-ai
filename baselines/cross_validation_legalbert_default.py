"""Cross-validation independente do baseline LegalBERTpt para execução no Colab.

Execute este arquivo após gerar dataset_clean/ e results/class_weights.json.
Segue o protocolo de cross_validation.py, com os hiperparâmetros do baseline.
"""

import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


MODEL_NAME = "raquelsilveira/legalbertpt_fp"
RESULT_NAME = "legalbert_default_cv"
DRY_RUN = False
START_FROM_FOLD = 1  # Ajuste para retomar; preserve os modelos e relatórios anteriores.

EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WARMUP_STEPS = 0
MICRO_BATCH_SIZE = 2
N_SPLITS = 5
RANDOM_STATE = 42


class LegalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))(
            outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(prediction):
    predicted = np.argmax(prediction.predictions, axis=1)
    return {
        'accuracy': accuracy_score(prediction.label_ids, predicted),
        'f1_macro': f1_score(prediction.label_ids, predicted, average='macro'),
    }


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record_protocol(output, backup, config, splits, labels, classes, start_from_fold):
    """Identifica os dados e registra a divisão sem alterar os índices originais."""
    manifest = output / 'run_manifest.json'
    if not manifest.exists() and backup and (backup / manifest.name).is_file():
        shutil.copy2(backup / manifest.name, manifest)
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if previous != config:
            raise ValueError('Configuração/dados/versões diferentes da execução salva. Use outra pasta de resultados.')
        if start_from_fold == 1:
            raise FileExistsError(f'Já existe uma execução em {output}. Preserve-a antes de iniciar outra.')
    elif start_from_fold > 1:
        raise FileNotFoundError('Retomada exige run_manifest.json da execução anterior, local ou no Drive.')
    manifest.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')

    assignments = np.zeros(len(labels), dtype=int)
    distributions = []
    for fold, (train_indices, val_indices) in enumerate(splits, 1):
        assignments[val_indices] = fold
        for partition, indices in [('train', train_indices), ('validation', val_indices)]:
            counts = pd.Series([labels[index] for index in indices]).value_counts()
            for label in classes:
                count = int(counts.get(label, 0))
                distributions.append({
                    'fold': fold, 'partition': partition, 'class': label,
                    'count': count, 'proportion': count / len(indices),
                })
    pd.DataFrame({
        'row_index': np.arange(len(labels)), 'document_type': labels,
        'validation_fold': assignments,
    }).to_csv(output / 'fold_assignments.csv', index=False)
    pd.DataFrame(distributions).to_csv(output / 'fold_class_distribution.csv', index=False)


def evaluate_and_report(y_true, y_pred, classes, title, prefix):
    """Mesmas métricas e matrizes da avaliação original, na mesma ordem de classes."""
    result = {
        'modelo': title,
        'accuracy_test': accuracy_score(y_true, y_pred),
        'precision_macro_test': precision_score(y_true, y_pred, average='macro'),
        'recall_macro_test': recall_score(y_true, y_pred, average='macro'),
        'f1_macro_test': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted_test': f1_score(y_true, y_pred, average='weighted'),
    }
    label_ids = list(range(len(classes)))
    print(f'\n{title}')
    print(classification_report(y_true, y_pred, labels=label_ids, target_names=classes, digits=4))
    pd.DataFrame([result]).to_csv(f'{prefix}_results.csv', index=False)
    pd.DataFrame({
        'Classe': classes,
        'Precisão': precision_score(y_true, y_pred, labels=label_ids, average=None),
        'Recall': recall_score(y_true, y_pred, labels=label_ids, average=None),
        'F1-Score': f1_score(y_true, y_pred, labels=label_ids, average=None),
        'Suporte': [int(sum(y_true == label)) for label in label_ids],
    }).to_csv(f'{prefix}_per_class.csv', index=False)

    abbreviations = {
        'acordao_de_2_instancia': 'Acórdão',
        'agravo_em_recurso_extraordinario': 'ARE',
        'despacho_de_admissibilidade': 'Despacho',
        'peticao_do_RE': 'RE', 'sentenca': 'Sentença',
    }
    ticks = [abbreviations.get(label, label) for label in classes]
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    normalized = np.divide(cm, cm.sum(axis=1, keepdims=True),
                           out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, values, fmt, suffix in zip(axes, [cm, normalized], ['d', '.2%'], ['Absoluto', 'Normalizada']):
        sns.heatmap(values, annot=True, fmt=fmt, cmap='Purples', xticklabels=ticks, yticklabels=ticks, ax=ax)
        ax.set_title(f'Matriz de Confusão — {title} ({suffix})')
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
    plt.tight_layout()
    plt.savefig(f'{prefix}_confusion_matrix.png', dpi=150)
    plt.close(fig)
    return result


def predict_probabilities(model_path, dataset, output, device):
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device)
    args = TrainingArguments(
        output_dir=str(output), per_device_eval_batch_size=MICRO_BATCH_SIZE * 2,
        report_to='none',
    )
    trainer = Trainer(model=model, args=args, compute_metrics=compute_metrics)
    prediction = trainer.predict(dataset)
    probabilities = torch.softmax(torch.tensor(prediction.predictions), dim=1).numpy()
    del trainer, model
    release_memory()
    return probabilities


def main():
    model_name = MODEL_NAME
    result_name = RESULT_NAME
    start_from_fold = START_FROM_FOLD
    dry_run = DRY_RUN
    if not 1 <= start_from_fold <= N_SPLITS:
        raise ValueError(f'START_FROM_FOLD deve estar entre 1 e {N_SPLITS}.')
    if Path(result_name).name != result_name or result_name in {'', '.', '..', 'results_cv'}:
        raise ValueError('Use um nome de experimento separado de results_cv.')
    if dry_run:
        result_name += '_dry_run'
    local_root = Path('/content/results') if Path('/content').is_dir() else Path('results')
    output = local_root / result_name
    output.mkdir(parents=True, exist_ok=True)
    drive_root = Path('/content/drive/MyDrive')
    backup = drive_root / 'tcc_cross_validation' / result_name if drive_root.is_dir() else None
    if backup:
        backup.mkdir(parents=True, exist_ok=True)

    train_path = Path('dataset_clean/train_clean.csv')
    test_path = Path('dataset_clean/test_clean.csv')
    weights_path = Path('results/class_weights.json')
    for path in (train_path, test_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(f'Arquivo ausente: {path}. Gere os dados com pre_processamento.py na raiz do projeto.')
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    if dry_run:
        df_train = df_train.head(200)
        df_test = df_test.head(100)
    weights = json.loads(weights_path.read_text(encoding='utf-8'))
    classes = sorted(weights)
    label2id = {label: index for index, label in enumerate(classes)}
    id2label = {index: label for label, index in label2id.items()}
    for name, frame in [('treino', df_train), ('teste', df_test)]:
        if not {'body', 'document_type'}.issubset(frame.columns):
            raise ValueError(f'CSV de {name} deve conter body e document_type.')
        if frame.empty or frame[['body', 'document_type']].isna().any().any():
            raise ValueError(f'CSV de {name} vazio ou com textos/rótulos ausentes.')
        if set(frame['document_type']) - set(classes):
            raise ValueError(f'CSV de {name} contém classes sem peso.')
    if set(df_train['document_type']) != set(classes) or df_train['document_type'].value_counts().min() < N_SPLITS:
        raise ValueError('Treino precisa conter pelo menos cinco páginas de cada classe, inclusive no DRY_RUN.')

    texts = df_train['body'].tolist()
    labels = df_train['document_type'].tolist()
    label_ids = [label2id[label] for label in labels]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(texts, labels))
    config = {
        'model_name': model_name, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
        'micro_batch_size': MICRO_BATCH_SIZE, 'gradient_accumulation_steps': BATCH_SIZE // MICRO_BATCH_SIZE,
        'eval_batch_size': MICRO_BATCH_SIZE * 2, 'learning_rate': LEARNING_RATE,
        'warmup_steps': WARMUP_STEPS, 'weight_decay': 0.01, 'max_length': 512,
        'n_splits': N_SPLITS, 'random_state': RANDOM_STATE, 'dry_run': dry_run,
        'class_weights': weights, 'class_weight_source': 'global_training_set',
        'model_selection': 'last_epoch_then_best_fold_by_validation_f1_macro',
        'inputs_sha256': {str(path): file_sha256(path) for path in (train_path, test_path, weights_path)},
        'versions': {name: importlib.metadata.version(name) for name in ('scikit-learn', 'transformers', 'torch', 'pandas', 'numpy', 'accelerate')},
        'fp16': torch.cuda.is_available(), 'gradient_checkpointing': True,
    }
    record_protocol(output, backup, config, splits, labels, classes, start_from_fold)
    csv_path = output / 'cv_fold_results.csv'
    fold_results = []
    model_paths = [output / f'fold_{fold}' / 'best_model' for fold in range(1, N_SPLITS + 1)]
    if start_from_fold > 1:
        if not csv_path.exists() and backup and (backup / csv_path.name).is_file():
            shutil.copy2(backup / csv_path.name, csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError('Retomada exige cv_fold_results.csv dos folds anteriores.')
        previous = pd.read_csv(csv_path)
        previous = previous[previous['fold'] < start_from_fold]
        if sorted(previous['fold'].tolist()) != list(range(1, start_from_fold)):
            raise ValueError('CSV de retomada deve conter uma linha para cada fold anterior.')
        fold_results = previous.to_dict('records')
        for fold in range(1, start_from_fold):
            path = model_paths[fold - 1]
            if not (path / 'config.json').is_file() and backup:
                saved = backup / f'fold_{fold}' / 'best_model'
                if saved.is_dir():
                    shutil.copytree(saved, path, dirs_exist_ok=True)
            if not (path / 'config.json').is_file() or not any((path / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin', 'model.safetensors.index.json', 'pytorch_model.bin.index.json')):
                raise FileNotFoundError(f'Modelo anterior ausente/incompleto: {path}')

    print(f'\nCROSS-VALIDATION — {model_name}\nResultados: {output}')
    print(f'Treino: {len(df_train)} | Teste: {len(df_test)} | Pesos globais: {weights}')
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights = torch.tensor([weights[label] for label in classes], dtype=torch.float)

    for fold, (train_indices, val_indices) in enumerate(splits, 1):
        if fold < start_from_fold:
            print(f'Fold {fold}/{N_SPLITS}: recuperado.')
            continue
        print(f'\nFold {fold}/{N_SPLITS}: treino={len(train_indices)}, validação={len(val_indices)}')
        train_dataset = LegalDataset(
            tokenizer([texts[i] for i in train_indices], truncation=True, padding=True, max_length=512),
            [label_ids[i] for i in train_indices],
        )
        val_dataset = LegalDataset(
            tokenizer([texts[i] for i in val_indices], truncation=True, padding=True, max_length=512),
            [label_ids[i] for i in val_indices],
        )
        # Como no original, não reinicializamos as sementes antes de carregar o modelo.
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(classes), id2label=id2label, label2id=label2id,
        ).to(device)
        args = TrainingArguments(
            output_dir=str(output / f'fold_{fold}'), num_train_epochs=EPOCHS,
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            per_device_eval_batch_size=MICRO_BATCH_SIZE * 2,
            gradient_accumulation_steps=BATCH_SIZE // MICRO_BATCH_SIZE,
            learning_rate=LEARNING_RATE, warmup_steps=WARMUP_STEPS, weight_decay=0.01,
            eval_strategy='epoch', save_strategy='no', logging_steps=100,
            load_best_model_at_end=False, fp16=torch.cuda.is_available(),
            gradient_checkpointing=True, report_to='none',
        )
        trainer = WeightedTrainer(
            model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset,
            compute_metrics=compute_metrics, class_weights=class_weights,
        )
        started = time.time()
        trainer.train()
        elapsed = (time.time() - started) / 60
        metrics = trainer.evaluate()
        trainer.save_model(str(model_paths[fold - 1]))
        tokenizer.save_pretrained(str(model_paths[fold - 1]))
        fold_results.append({
            'fold': fold, 'val_loss': metrics['eval_loss'],
            'val_accuracy': metrics['eval_accuracy'], 'val_f1_macro': metrics['eval_f1_macro'],
            'train_time_min': elapsed,
        })
        pd.DataFrame(fold_results).to_csv(csv_path, index=False)
        if backup:
            try:
                shutil.copytree(model_paths[fold - 1], backup / f'fold_{fold}' / 'best_model', dirs_exist_ok=True)
                for name in ('run_manifest.json', 'fold_assignments.csv', 'fold_class_distribution.csv', csv_path.name):
                    shutil.copy2(output / name, backup / name)
            except OSError as error:
                print(f'[BACKUP] Falha; resultados permanecem em {output}: {error}')
        del trainer, model, train_dataset, val_dataset
        release_memory()

    df_cv = pd.DataFrame(fold_results).sort_values('fold').reset_index(drop=True)
    df_cv.to_csv(csv_path, index=False)
    best_row = df_cv.loc[df_cv['val_f1_macro'].idxmax()]
    best_fold = int(best_row['fold'])
    print(f"\nF1 macro CV: {df_cv.val_f1_macro.mean():.6f} ± {df_cv.val_f1_macro.std():.6f}")
    print(f'Melhor fold pela validação: {best_fold}')

    # O teste não participa dos folds nem da seleção do melhor fold.
    test_dataset = LegalDataset(
        tokenizer(df_test['body'].tolist(), truncation=True, padding=True, max_length=512),
        [label2id[label] for label in df_test['document_type']],
    )
    y_true = np.array(test_dataset.labels)
    best_probs = predict_probabilities(model_paths[best_fold - 1], test_dataset, output / 'evaluation', device)
    best_result = evaluate_and_report(
        y_true, best_probs.argmax(axis=1), classes, f'Best Fold (Fold {best_fold})', output / 'best_fold',
    )
    all_probs = []
    for fold, model_path in enumerate(model_paths, 1):
        print(f'Ensemble: inferência do fold {fold}/{N_SPLITS}')
        all_probs.append(predict_probabilities(model_path, test_dataset, output / 'evaluation', device))
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_result = evaluate_and_report(
        y_true, ensemble_probs.argmax(axis=1), classes, 'Ensemble (Soft Voting)', output / 'ensemble',
    )
    pd.DataFrame([best_result, ensemble_result]).to_csv(output / 'comparison_best_fold_vs_ensemble.csv', index=False)
    summary = {
        'modelo': model_name, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE, 'warmup_steps': WARMUP_STEPS, 'n_splits': N_SPLITS,
        'cv_mean_f1_macro': df_cv.val_f1_macro.mean(), 'cv_std_f1_macro': df_cv.val_f1_macro.std(),
        'cv_mean_accuracy': df_cv.val_accuracy.mean(), 'cv_std_accuracy': df_cv.val_accuracy.std(),
        'best_fold': best_fold, 'best_fold_val_f1': best_row['val_f1_macro'],
        'best_fold_test_f1_macro': best_result['f1_macro_test'],
        'best_fold_test_accuracy': best_result['accuracy_test'],
        'ensemble_test_f1_macro': ensemble_result['f1_macro_test'],
        'ensemble_test_accuracy': ensemble_result['accuracy_test'],
    }
    pd.DataFrame([summary]).to_csv(output / 'cv_summary.csv', index=False)
    if backup:
        try:
            for path in output.iterdir():
                if path.is_file():
                    shutil.copy2(path, backup / path.name)
        except OSError as error:
            print(f'[BACKUP] Falha no backup final; resultados permanecem em {output}: {error}')
    print(f'\nCross-validation concluída. Relatórios em: {output}')


if __name__ == "__main__":
    main()
