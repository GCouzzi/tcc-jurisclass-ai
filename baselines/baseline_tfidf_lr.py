import pandas as pd
import numpy as np
import json
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

def main():
    print("=" * 60)
    print(" BASELINE: TF-IDF + REGRESSÃO LOGÍSTICA")
    print("=" * 60)

    # 1. Carregar dados
    print("\n[1/6] Carregando dados...")
    df_train = pd.read_csv("train_clean.csv")
    df_val   = pd.read_csv("val_clean.csv")
    df_test  = pd.read_csv("test_clean.csv")

    with open("class_weights.json", "r") as f:
        class_weights_dict = json.load(f)

    # Label mapping
    classes = sorted(list(class_weights_dict.keys()))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for i, c in enumerate(classes)}

    print(f"  Treino: {len(df_train)} | Validação: {len(df_val)} | Teste: {len(df_test)}")
    print(f"  Classes ({len(classes)}): {classes}")

    # 2. Vetorização TF-IDF
    print("\n[2/6] Vetorizando textos com TF-IDF...")
    t0 = time.time()

    tfidf = TfidfVectorizer(
        max_features=50000,
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_train = tfidf.fit_transform(df_train['body'].fillna(''))
    X_val   = tfidf.transform(df_val['body'].fillna(''))
    X_test  = tfidf.transform(df_test['body'].fillna(''))

    y_train = df_train['document_type'].values
    y_val   = df_val['document_type'].values
    y_test  = df_test['document_type'].values

    print(f"  Vetorização concluída em {time.time()-t0:.1f}s")
    print(f"  Shape TF-IDF: {X_train.shape}")
    print(f"  Vocabulário: {len(tfidf.vocabulary_)} termos")

    # 3. Treinar Regressão Logística
    print("\n[3/6] Treinando Regressão Logística com class_weight='balanced'...")
    t0 = time.time()

    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        random_state=42,
        n_jobs=-1
    )

    lr_model.fit(X_train, y_train)
    print(f"  Treinamento concluído em {time.time()-t0:.1f}s")

    # 4. Avaliação (Validação e Teste)
    print("\n[4/6] Avaliando modelo...")
    y_val_pred = lr_model.predict(X_val)
    y_test_pred = lr_model.predict(X_test)

    print("\n" + "=" * 50)
    print(" RESULTADOS FINAIS NO CONJUNTO DE TESTE ")
    print("=" * 50)

    acc  = accuracy_score(y_test, y_test_pred)
    f1_m = f1_score(y_test, y_test_pred, average='macro')
    f1_w = f1_score(y_test, y_test_pred, average='weighted')

    print(f"\nAcurácia:    {acc:.4f}")
    print(f"F1 Macro:    {f1_m:.4f}")
    print(f"F1 Weighted: {f1_w:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, y_test_pred, digits=4))

    # 5. Métricas detalhadas por classe
    labels_abrev = ['Acórdão', 'ARE', 'Despacho', 'RE', 'Sentença']
    precision_per_class = precision_score(y_test, y_test_pred, labels=classes, average=None)
    recall_per_class    = recall_score(y_test, y_test_pred, labels=classes, average=None)
    f1_per_class        = f1_score(y_test, y_test_pred, labels=classes, average=None)

    df_metrics = pd.DataFrame({
        'Classe': labels_abrev,
        'Precisão': precision_per_class,
        'Recall': recall_per_class,
        'F1-Score': f1_per_class,
        'Suporte': [sum(y_test == c) for c in classes]
    })

    # 6. Salvar Resultados e Gráficos
    print("\n[5/6] Salvando matriz de confusão e métricas...")
    
    out_dir = "results/tf-idf_lr"
    os.makedirs(out_dir, exist_ok=True)
    
    cm = confusion_matrix(y_test, y_test_pred, labels=classes)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Matriz absoluta
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=labels_abrev, yticklabels=labels_abrev, ax=axes[0])
    axes[0].set_title('Matriz de Confusão — TF-IDF + Reg. Logística (Absoluto)', fontsize=13)
    axes[0].set_ylabel('Classe Real')
    axes[0].set_xlabel('Classe Predita')

    # Matriz normalizada (%)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Oranges',
                xticklabels=labels_abrev, yticklabels=labels_abrev, ax=axes[1])
    axes[1].set_title('Matriz de Confusão — TF-IDF + Reg. Logística (Normalizada)', fontsize=13)
    axes[1].set_ylabel('Classe Real')
    axes[1].set_xlabel('Classe Predita')

    plt.tight_layout()
    plt.savefig(f'{out_dir}/confusion_matrix_tfidf_lr.png', dpi=150, bbox_inches='tight')
    plt.close() # Fecha a figura pra nao travar o script

    results_summary = {
        'modelo': 'TF-IDF + Regressão Logística',
        'tfidf_max_features': 50000,
        'tfidf_ngram_range': '(1, 2)',
        'lr_C': 1.0,
        'lr_solver': 'lbfgs',
        'class_weight': 'balanced',
        'accuracy_test': acc,
        'f1_macro_test': f1_m,
        'f1_weighted_test': f1_w
    }

    df_summary = pd.DataFrame([results_summary])
    df_summary.to_csv(f'{out_dir}/baseline_tfidf_lr_results.csv', index=False)
    df_metrics.to_csv(f'{out_dir}/baseline_tfidf_lr_per_class.csv', index=False)

    print("\n[6/6] Processo finalizado com sucesso!")
    print(f"  - {out_dir}/baseline_tfidf_lr_results.csv (resumo geral)")
    print(f"  - {out_dir}/baseline_tfidf_lr_per_class.csv (por classe)")
    print(f"  - {out_dir}/confusion_matrix_tfidf_lr.png (matriz de confusão salva)")

if __name__ == '__main__':
    main()
