import pandas as pd
import re

# ─── 1. CARREGAR OS DADOS ────────────────────────────────────────────────────
# Ajuste os caminhos conforme onde estão seus arquivos
df_train = pd.read_csv("dataset_victor/train_small.csv")
df_val   = pd.read_csv("dataset_victor/validation_small.csv")
df_test  = pd.read_csv("dataset_victor/test_small.csv")

print("=== SHAPE ORIGINAL ===")
print(f"Treino:     {df_train.shape}")
print(f"Validação:  {df_val.shape}")
print(f"Teste:      {df_test.shape}")

print("\n=== COLUNAS ===")
print(df_train.columns.tolist())

print("\n=== DISTRIBUIÇÃO DE CLASSES (treino) ===")
print(df_train["document_type"].value_counts())

print("\n=== EXEMPLO DE REGISTRO ===")
print(df_train.head(2).to_string())


# ─── 2. REMOVER CLASSE 'Others' ──────────────────────────────────────────────
df_train = df_train[df_train["document_type"] != "outros"]
df_val   = df_val[df_val["document_type"]     != "outros"]
df_test  = df_test[df_test["document_type"]   != "outros"]

print("\n=== SHAPE APÓS REMOÇÃO DE 'Others' ===")
print(f"Treino:     {df_train.shape}")
print(f"Validação:  {df_val.shape}")
print(f"Teste:      {df_test.shape}")

print("\n=== DISTRIBUIÇÃO APÓS REMOÇÃO DE OUTROS (treino) ===")
print(df_train["document_type"].value_counts())

print("\n=== DISTRIBUIÇÃO APÓS REMOÇÃO DE OUTROS (validação) ===")
print(df_val["document_type"].value_counts())

print("\n=== DISTRIBUIÇÃO APÓS REMOÇÃO DE OUTROS (teste) ===")
print(df_test["document_type"].value_counts())


# ─── 3. LIMPEZA TEXTUAL ──────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove sequências de caracteres repetidos (ex: ______, ------)
    text = re.sub(r'([_\-=\*#])\1{2,}', ' ', text)
    # Substitui quebras de linha por espaço
    text = re.sub(r'\n+', ' ', text)
    # Remove caracteres não informativos (mantém letras, números, pontuação)
    text = re.sub(r'[^\w\s\.,;:!?\-\(\)\"\'\/]', ' ', text)
    # Remove espaços redundantes
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

for df in [df_train, df_val, df_test]:
    df["body"] = df["body"].apply(clean_text)

print("\n=== EXEMPLO APÓS LIMPEZA ===")
print(df_train["body"].iloc[0][:300])


# ─── 4. SALVAR OS DADOS LIMPOS ───────────────────────────────────────────────
df_train.to_csv("train_clean.csv", index=False)
df_val.to_csv("val_clean.csv",     index=False)
df_test.to_csv("test_clean.csv",   index=False)

print("\n✅ Arquivos salvos: train_clean.csv, val_clean.csv, test_clean.csv")