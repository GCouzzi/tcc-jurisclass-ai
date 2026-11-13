import os
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo em uso: {device}")
    
    # 1. Caminhos
    repo = "gcouzzi/jurisclass-legalbertpt"
    test_csv = "dataset_clean/test_clean.csv" if os.path.exists("dataset_clean/test_clean.csv") else "implementacao-tcc-testes/dataset_clean/test_clean.csv"
    out_dir = "results/results_cv" if os.path.exists("results") else "implementacao-tcc-testes/results/results_cv"
    weights_path = "results/class_weights.json" if os.path.exists("results/class_weights.json") else "implementacao-tcc-testes/results/class_weights.json"
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Dados de Teste
    df_test = pd.read_csv(test_csv)
    print(f"Total de amostras de teste: {len(df_test)}")
    
    with open(weights_path, 'r', encoding='utf-8') as f:
        class_weights = json.load(f)
    classes = sorted(list(class_weights.keys()))
    label2id = {c: i for i, c in enumerate(classes)}
    
    # Coluna do label pode ser 'label' ou 'document_type'
    col_label = 'label' if 'label' in df_test.columns else 'document_type'
    y_test_true = np.array([label2id[c] for c in df_test[col_label]])
    np.save(os.path.join(out_dir, "y_test_true.npy"), y_test_true)
    
    # 3. Tokenização
    tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="fold_5/best_model")
    print("Tokenizando textos de teste...")
    tokens = [tokenizer(t, truncation=True, max_length=512) for t in df_test['body'].tolist()]
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
    loader = DataLoader(tokens, batch_size=64 if torch.cuda.is_available() else 16, shuffle=False, collate_fn=collator)
    
    # 4. Inferência nos 5 folds
    all_fold_probs = []
    best_fold_preds = None
    
    t0 = time.time()
    for fold in range(1, 6):
        subfolder = f"fold_{fold}/best_model"
        print(f"\n[Fold {fold}/5] Baixando/carregando {repo} ({subfolder})...")
        model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder=subfolder)
        if torch.cuda.is_available():
            model = model.half().to(device)
        else:
            model = model.to(device)
        model.eval()
        
        fold_probs = []
        with torch.inference_mode():
            for batch in loader:
                b = {k: v.to(device) for k, v in batch.items()}
                logits = model(**b).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                fold_probs.append(probs)
                
        fold_probs = np.vstack(fold_probs)
        all_fold_probs.append(fold_probs)
        fold_preds = np.argmax(fold_probs, axis=1)
        
        acc = accuracy_score(y_test_true, fold_preds)
        f1 = f1_score(y_test_true, fold_preds, average='macro')
        print(f"  -> Fold {fold} concluído em {time.time()-t0:.1f}s | Acc: {acc*100:.2f}% | F1-macro: {f1*100:.2f}%")
        
        if fold == 5:
            best_fold_preds = fold_preds
            np.save(os.path.join(out_dir, "best_fold_test_pred.npy"), best_fold_preds)
            print("  -> Salvo: best_fold_test_pred.npy")
            
    # 5. Ensemble (Soft Voting)
    ensemble_probs = np.mean(all_fold_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    ens_acc = accuracy_score(y_test_true, ensemble_preds)
    ens_f1 = f1_score(y_test_true, ensemble_preds, average='macro')
    
    np.save(os.path.join(out_dir, "ensemble_test_pred.npy"), ensemble_preds)
    np.save(os.path.join(out_dir, "y_test_probs.npy"), ensemble_probs)
    
    print("\n" + "=" * 50)
    print(f"RESULTADO FINAL JURISCLASS AI (ENSEMBLE):")
    print(f"Acurácia: {ens_acc*100:.2f}%")
    print(f"F1-macro: {ens_f1*100:.2f}%")
    print("=" * 50)
    print(f"[OK] Arquivos salvos em: {out_dir}/")
    print("  - best_fold_test_pred.npy")
    print("  - ensemble_test_pred.npy")
    print("  - y_test_true.npy")
    print("  - y_test_probs.npy")

if __name__ == "__main__":
    main()
