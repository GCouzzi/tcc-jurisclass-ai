import pandas as pd

# ─── CARREGAR DATASETS ──────────────────────────────────────────────────────
# Dataset original (SVic)
orig_train = pd.read_csv("dataset_victor/train_small.csv")
orig_val   = pd.read_csv("dataset_victor/validation_small.csv")
orig_test  = pd.read_csv("dataset_victor/test_small.csv")

# Dataset limpo (pós pré-processamento)
clean_train = pd.read_csv("train_clean.csv")
clean_val   = pd.read_csv("val_clean.csv")
clean_test  = pd.read_csv("test_clean.csv")


def print_dist(nome, df):
    print(f"\n--- {nome} ({len(df)} registros) ---")
    counts = df["document_type"].value_counts()
    pcts   = df["document_type"].value_counts(normalize=True).mul(100).round(1)
    print(pd.DataFrame({"n": counts, "%": pcts}).to_string())


# ─── DATASET ORIGINAL (TODAS AS CLASSES) ────────────────────────────────────
print("=" * 60)
print("DATASET ORIGINAL — TODAS AS CLASSES")
print("=" * 60)

for nome, df in [("Treino", orig_train), ("Validação", orig_val), ("Teste", orig_test)]:
    print_dist(nome, df)

# ─── DATASET ORIGINAL (SEM 'OUTROS') ───────────────────────────────────────
print("\n" + "=" * 60)
print("DATASET ORIGINAL — SEM CLASSE 'OUTROS'")
print("=" * 60)

for nome, df in [("Treino", orig_train), ("Validação", orig_val), ("Teste", orig_test)]:
    print_dist(nome, df[df["document_type"] != "outros"])

# ─── DATASET LIMPO (PÓS PRÉ-PROCESSAMENTO) ─────────────────────────────────
print("\n" + "=" * 60)
print("DATASET LIMPO (PÓS PRÉ-PROCESSAMENTO)")
print("=" * 60)

for nome, df in [("Treino", clean_train), ("Validação", clean_val), ("Teste", clean_test)]:
    print_dist(nome, df)