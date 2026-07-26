import pandas as pd
import re
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

df_train = pd.read_csv("dataset_victor/train_small.csv")
df_val   = pd.read_csv("dataset_victor/validation_small.csv")
df_test  = pd.read_csv("dataset_victor/test_small.csv")

df_train = df_train[df_train["document_type"] != "outros"].copy()
df_val   = df_val[df_val["document_type"] != "outros"].copy()
df_test  = df_test[df_test["document_type"] != "outros"].copy()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'([_\-=\*#])\1{2,}', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^\w\s\.,;:!?\-\(\)\"\'\/]', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

for df in [df_train, df_val, df_test]:
    df["body"] = df["body"].apply(clean_text)

MIN_CHARS = 50
for name, df in [("treino", df_train), ("validação", df_val), ("teste", df_test)]:
    antes = len(df)
    mask = df["body"].str.len() >= MIN_CHARS
    if name == "treino":
        df_train = df[mask].copy()
    elif name == "validação":
        df_val = df[mask].copy()
    else:
        df_test = df[mask].copy()
    removidos = antes - mask.sum()
    print(f"  {name}: {removidos} registros removidos (< {MIN_CHARS} chars) | {antes} → {mask.sum()}")

classes = np.array(sorted(df_train["document_type"].unique()))
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=df_train["document_type"]
)
class_weight_dict = dict(zip(classes, weights))

print("\n=== CLASS WEIGHTS ===")
for cls, w in sorted(class_weight_dict.items(), key=lambda x: -x[1]):
    print(f"  {cls:45s} {w:.4f}")

df_train.to_csv("dataset_clean/train_clean.csv", index=False)
df_val.to_csv("dataset_clean/val_clean.csv",     index=False)
df_test.to_csv("dataset_clean/test_clean.csv",   index=False)

import json
with open("results/class_weights.json", "w") as f:
    json.dump(class_weight_dict, f, indent=2)

print("\nArquivos salvos.")