import pandas as pd

df_train = pd.read_csv("dataset_victor/train_small.csv")
df_val   = pd.read_csv("dataset_victor/validation_small.csv")
df_test  = pd.read_csv("dataset_victor/test_small.csv")

# Removendo a classe 'outros' conforme metodologia
df_train = df_train[df_train["document_type"] != "outros"]
df_val   = df_val[df_val["document_type"] != "outros"]
df_test  = df_test[df_test["document_type"] != "outros"]

splits = {"Treino": df_train, "Validação": df_val, "Teste": df_test}

# ─── DISTRIBUIÇÃO POR THEME ──────────────────────────────────────────────────
print("=" * 60)
print("DISTRIBUIÇÃO POR THEME (sem 'outros')")
print("=" * 60)

for nome, df in splits.items():
    print(f"\n--- {nome} ({len(df)} registros) ---")
    counts = df["themes"].value_counts()
    pcts   = df["themes"].value_counts(normalize=True).mul(100).round(1)
    print(pd.DataFrame({"n": counts, "%": pcts}).to_string())

# ─── DISTRIBUIÇÃO POR DOCUMENT_TYPE ─────────────────────────────────────────
print("\n" + "=" * 60)
print("DISTRIBUIÇÃO POR DOCUMENT_TYPE (sem 'outros')")
print("=" * 60)

for nome, df in splits.items():
    print(f"\n--- {nome} ({len(df)} registros) ---")
    counts = df["document_type"].value_counts()
    pcts   = df["document_type"].value_counts(normalize=True).mul(100).round(1)
    print(pd.DataFrame({"n": counts, "%": pcts}).to_string())