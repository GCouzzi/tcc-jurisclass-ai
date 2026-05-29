import fitz

caminho = "dataset_victor/small/test/AI_850120_789456_93_11052013.pdf"

doc = fitz.open(caminho)

print(f"Total de páginas: {len(doc)}")

# ─── 2. EXTRAIR TEXTO PÁGINA A PÁGINA ────────────────────────────────────────
for i, pagina in enumerate(doc):
    texto = pagina.get_text()
    print(texto[:300])  # mostra só os primeiros 300 caracteres

    if i == 2:  # para após 3 páginas para não poluir o output
        break

doc.close()