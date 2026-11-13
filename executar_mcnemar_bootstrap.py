import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score

# 1. Mapeamento e Classes
WEIGHTS_PATH = 'implementacao-tcc-testes/results/class_weights.json'
with open(WEIGHTS_PATH, 'r', encoding='utf-8') as f:
    class_weights = json.load(f)
classes = sorted(list(class_weights.keys()))
label2id = {c: i for i, c in enumerate(classes)}

# Carregar Gabarito do Teste (10.105 amostras)
y_true = np.load('implementacao-tcc-testes/results/bertimbau_base/y_test_true.npy')
N = len(y_true)
print(f"Total de amostras no conjunto de teste: {N}")

# 2. Carregar Predicoes dos Modelos
def load_pred(path, is_string=False):
    arr = np.load(path, allow_pickle=True)
    if is_string:
        arr = np.array([label2id[x] for x in arr])
    return arr.astype(int)

models_pred = {
    'TF-IDF + Regressão Logística': load_pred('implementacao-tcc-testes/results/tf-idf_lr/y_test_pred.npy', is_string=True),
    'TF-IDF + SVM': load_pred('implementacao-tcc-testes/results/tf-idf_svm/y_test_pred.npy', is_string=True),
    'BERTimbau Base (Treino Único)': load_pred('implementacao-tcc-testes/results/bertimbau_base/y_test_pred.npy'),
    'LegalBert-pt Ref. (Treino Único)': load_pred('implementacao-tcc-testes/results/legalbert_fp_base/y_test_pred.npy'),
    'BERTimbau CV (Best Fold)': load_pred('implementacao-tcc-testes/results/bertimbau_cv/best_fold_test_pred.npy'),
    'BERTimbau CV (Ensemble)': load_pred('implementacao-tcc-testes/results/bertimbau_cv/ensemble_test_pred.npy'),
    'LegalBert-pt CV (Best Fold)': load_pred('implementacao-tcc-testes/results/legalbert_default_cv/best_fold_test_pred.npy'),
    'LegalBert-pt CV (Ensemble)': load_pred('implementacao-tcc-testes/results/legalbert_default_cv/ensemble_test_pred.npy'),
}

# Verificar se temos o JurisClass AI salvo
jc_best_path = 'implementacao-tcc-testes/results/results_cv/best_fold_test_pred.npy'
jc_ens_path = 'implementacao-tcc-testes/results/results_cv/ensemble_test_pred.npy'

has_jurisclass = os.path.exists(jc_ens_path) and os.path.exists(jc_best_path)

if has_jurisclass:
    models_pred['JurisClass AI (Best Fold)'] = load_pred(jc_best_path)
    models_pred['JurisClass AI (Ensemble final)'] = load_pred(jc_ens_path)
    print("Predições do JurisClass AI encontradas!")
else:
    print("\n[AVISO] Predições do JurisClass AI ainda não estão presentes em results/results_cv/.")
    print("O script executará a análise completa com os modelos disponíveis.\n")

# 3. Teste de McNemar com Correção de Edwards
def mcnemar_test(pred_a, pred_b, y_t):
    corr_a = (pred_a == y_t)
    corr_b = (pred_b == y_t)
    b = int(np.sum(corr_a & ~corr_b))
    c = int(np.sum(~corr_a & corr_b))
    
    total = b + c
    if total == 0:
        return 0.0, 1.0, b, c
    chi2 = ((abs(b - c) - 1.0) ** 2) / total
    p_val = stats.chi2.sf(chi2, df=1)
    return chi2, p_val, b, c

# 4. Ajuste de P-valores por Holm-Bonferroni
def holm_bonferroni(p_vals):
    p_vals = np.array(p_vals)
    n = len(p_vals)
    order = np.argsort(p_vals)
    adjusted = np.zeros(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        p_adj = p_vals[idx] * (n - rank)
        running_max = max(running_max, p_adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted

# 5. Bootstrap de 1.000 Sorteios com Reposição
print("=" * 60)
print(" EXECUTANDO BOOTSTRAP (1.000 SORTEIOS) ")
print("=" * 60)
np.random.seed(42)
B = 1000

# Pré-alocar matriz de métricas
model_names = list(models_pred.keys())
f1_bootstrap = {m: np.zeros(B) for m in model_names}
acc_bootstrap = {m: np.zeros(B) for m in model_names}

t0 = time.time()
for b_idx in range(B):
    sample_indices = np.random.randint(0, N, size=N)
    y_sample = y_true[sample_indices]
    
    for m in model_names:
        pred_sample = models_pred[m][sample_indices]
        acc_bootstrap[m][b_idx] = accuracy_score(y_sample, pred_sample)
        f1_bootstrap[m][b_idx] = f1_score(y_sample, pred_sample, average='macro', zero_division=0)
    
    if (b_idx + 1) % 200 == 0:
        print(f"  Progresso: {b_idx + 1}/{B} iterações em {time.time()-t0:.1f}s")

# Calcular Intervalos de Confiança (2.5% e 97.5%)
bootstrap_results = []
for m in model_names:
    f1_orig = f1_score(y_true, models_pred[m], average='macro')
    acc_orig = accuracy_score(y_true, models_pred[m])
    
    f1_ci_low = np.percentile(f1_bootstrap[m], 2.5) * 100
    f1_ci_high = np.percentile(f1_bootstrap[m], 97.5) * 100
    
    acc_ci_low = np.percentile(acc_bootstrap[m], 2.5) * 100
    acc_ci_high = np.percentile(acc_bootstrap[m], 97.5) * 100
    
    bootstrap_results.append({
        'Modelo': m,
        'Acurácia': acc_orig * 100,
        'Acurácia IC 95%': f"[{acc_ci_low:.2f}%; {acc_ci_high:.2f}%]",
        'F1-macro': f1_orig * 100,
        'F1-macro IC 95%': f"[{f1_ci_low:.2f}%; {f1_ci_high:.2f}%]",
        'f1_mean': f1_orig * 100,
        'f1_ci_low': f1_ci_low,
        'f1_ci_high': f1_ci_high
    })

df_bootstrap = pd.DataFrame(bootstrap_results)
print("\n=== RESULTADOS DO BOOTSTRAP (INTERVALOS DE CONFIANÇA 95%) ===")
for r in bootstrap_results:
    print(f"{r['Modelo']:<32} | F1: {r['F1-macro']:5.2f}% {r['F1-macro IC 95%']} | Acc: {r['Acurácia']:5.2f}% {r['Acurácia IC 95%']}")

# 6. Testes de McNemar
print("\n" + "=" * 60)
print(" TESTES DE MCNEMAR COM AJUSTE DE HOLM ")
print("=" * 60)

# Lista de comparações
mcnemar_pairs = []

# Se o JurisClass AI estiver presente, faz os testes de McNemar do JurisClass AI (Ensemble final) contra todos os modelos da Tabela 6:
if has_jurisclass:
    target_models = [
        'TF-IDF + Regressão Logística',
        'TF-IDF + SVM',
        'BERTimbau Base (Treino Único)',
        'LegalBert-pt Ref. (Treino Único)',
        'BERTimbau CV (Ensemble)',
        'LegalBert-pt CV (Ensemble)',
        'JurisClass AI (Best Fold)'
    ]
    for m in target_models:
        mcnemar_pairs.append(('JurisClass AI (Ensemble final)', m))
    
    # Teste de referência da literatura (LegalBert-pt vs BERTimbau Base em treino único)
    mcnemar_pairs.append(('LegalBert-pt Ref. (Treino Único)', 'BERTimbau Base (Treino Único)'))
else:
    # Comparações entre os modelos disponíveis
    ref_ens = 'LegalBert-pt CV (Ensemble)'
    comparisons = [
        'TF-IDF + Regressão Logística',
        'TF-IDF + SVM',
        'BERTimbau Base (Treino Único)',
        'LegalBert-pt Ref. (Treino Único)',
        'BERTimbau CV (Ensemble)',
        'LegalBert-pt CV (Best Fold)'
    ]
    for m in comparisons:
        mcnemar_pairs.append((ref_ens, m))
    mcnemar_pairs.append(('LegalBert-pt Ref. (Treino Único)', 'BERTimbau Base (Treino Único)'))

raw_p_values = []
mcnemar_records = []

for mod_a, mod_b in mcnemar_pairs:
    chi2, p_val, b, c = mcnemar_test(models_pred[mod_a], models_pred[mod_b], y_true)
    raw_p_values.append(p_val)
    mcnemar_records.append({
        'Modelo A': mod_a,
        'Modelo B': mod_b,
        'b (A certo, B errado)': b,
        'c (A errado, B certo)': c,
        'chi2': chi2,
        'p_raw': p_val
    })

adj_p_values = holm_bonferroni(raw_p_values)

for i, rec in enumerate(mcnemar_records):
    rec['p_adj'] = adj_p_values[i]
    rec['Significativo (p < 0.05)'] = "Sim" if adj_p_values[i] < 0.05 else "Não"

df_mcnemar = pd.DataFrame(mcnemar_records)
print(df_mcnemar[['Modelo A', 'Modelo B', 'b (A certo, B errado)', 'c (A errado, B certo)', 'chi2', 'p_raw', 'p_adj', 'Significativo (p < 0.05)']].to_string(index=False))

# Salvar relatórios em CSV
out_dir = 'implementacao-tcc-testes/results/statistical_tests'
os.makedirs(out_dir, exist_ok=True)
df_bootstrap.to_csv(os.path.join(out_dir, 'bootstrap_results.csv'), index=False)
df_mcnemar.to_csv(os.path.join(out_dir, 'mcnemar_results.csv'), index=False)

# 7. Gerar Gráfico de Intervalos de Confiança (Ponto e Barra) apenas para os 8 modelos da Tabela 6
tabela6_order = [
    'TF-IDF + Regressão Logística',
    'TF-IDF + SVM',
    'BERTimbau Base (Treino Único)',
    'LegalBert-pt Ref. (Treino Único)',
    'BERTimbau CV (Ensemble)',
    'LegalBert-pt CV (Ensemble)',
    'JurisClass AI (Best Fold)',
    'JurisClass AI (Ensemble final)',
]

display_labels = {
    'TF-IDF + Regressão Logística': 'TF-IDF + Regressão Logística',
    'TF-IDF + SVM': 'TF-IDF + SVM (LinearSVC)',
    'BERTimbau Base (Treino Único)': 'BERTimbau Base',
    'LegalBert-pt Ref. (Treino Único)': 'LegalBert-pt de referência',
    'BERTimbau CV (Ensemble)': 'BERTimbau Base (Ensemble)',
    'LegalBert-pt CV (Ensemble)': 'LegalBert-pt de referência (Ensemble)',
    'JurisClass AI (Best Fold)': 'JurisClass AI (Best Fold)',
    'JurisClass AI (Ensemble final)': 'JurisClass AI (Ensemble final)',
}

df_plot = df_bootstrap[df_bootstrap['Modelo'].isin(tabela6_order)].copy()
df_plot['sort_key'] = df_plot['Modelo'].map(lambda x: tabela6_order.index(x))
df_plot = df_plot.sort_values('sort_key').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 4.5))
y_positions = np.arange(len(df_plot))

for idx, row in df_plot.iterrows():
    f1_val = row['f1_mean']
    ci_low = row['f1_ci_low']
    ci_high = row['f1_ci_high']
    err_low = f1_val - ci_low
    err_high = ci_high - f1_val
    
    is_final = row['Modelo'] == 'JurisClass AI (Ensemble final)'
    color = '#1f4e78' if not is_final else '#b32400'
    
    ax.errorbar(
        f1_val, idx,
        xerr=[[err_low], [err_high]],
        fmt='o', color=color, ecolor=color,
        elinewidth=2.2 if is_final else 1.8,
        capsize=5, capthick=2 if is_final else 1.5,
        markersize=7 if is_final else 6
    )

ax.set_yticks(y_positions)
ax.set_yticklabels([display_labels[m] for m in df_plot['Modelo']], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('F1-macro (%) no Conjunto de Teste Independente', fontsize=11, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
fig_pdf = 'Latex/figuras/intervalos_confianca_modelos.pdf'
fig_png = 'Latex/figuras/intervalos_confianca_modelos.png'
plt.savefig(fig_pdf, dpi=300)
plt.savefig(fig_png, dpi=300)
plt.close(fig)

print(f"\n[OK] Gráfico de intervalos salvo em:")
print(f"  - {fig_pdf}")
print(f"  - {fig_png}")
print(f"[OK] CSVs salvos em: {out_dir}/")
