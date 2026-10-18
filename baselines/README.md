# Baselines com validação cruzada

Os baselines com validação fixa continuam em `baseline_bertimbau.py` e
`baseline_legalbert_default.py`. As novas execuções usam:

| Script | Modelo | Resultados locais |
| --- | --- | --- |
| `cross_validation_bertimbau.py` | `neuralmind/bert-base-portuguese-cased` | `results/bertimbau_cv/` |
| `cross_validation_legalbert_default.py` | `raquelsilveira/legalbertpt_fp` | `results/legalbert_default_cv/` |

Execute a partir da raiz do projeto, no ambiente de treinamento com as mesmas
bibliotecas usadas em `cross_validation.py`:

```bash
python baselines/cross_validation_bertimbau.py
python baselines/cross_validation_legalbert_default.py
```

Os CSVs limpos são gerados localmente antes do treinamento. Os scripts de
cross-validation somente leem os arquivos prontos; não executam pré-processamento.

## Protocolo

Cada script contém o pipeline completo e pode ser executado sozinho no Colab,
sem importar outros arquivos Python do projeto. Ambos seguem o procedimento de
`../cross_validation.py`. Cada modelo usa 3 épocas, lote efetivo 16 (microlote 2
e acumulação de 8 passos em uma GPU), lote de avaliação 4, learning rate `5e-5`,
weight decay `0.01` e warmup de 0 passos. Usa gradient checkpointing e FP16
quando CUDA está disponível, como a cross-validation original.

Os cinco folds são produzidos somente a partir de `train_clean.csv`, com
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Para reconstruir as
divisões originais, mantenha os mesmos dados, ordem das linhas e versão do
scikit-learn. Regenerar o CSV com os mesmos arquivos brutos e o mesmo
pré-processamento preserva essa ordem. `val_clean.csv` não participa dos folds.

Preservamos os pesos globais de `results/class_weights.json`, a entropia cruzada
ponderada, a tokenização cased com limite de 512 tokens e a avaliação a cada
época. O modelo salvo em `fold_N/best_model` é o da última época: não há seleção
da melhor época, diferentemente dos baselines com validação fixa. A semente 42
é a da divisão dos folds; o controle de aleatoriedade do treino segue o script
original, sem adicionar uma chamada a `set_seed` antes de carregar cada modelo.

O teste fica separado e só é avaliado após os cinco treinamentos. São gerados
os mesmos relatórios do script original: métricas de validação por fold,
média e desvio-padrão amostral, avaliação do melhor fold escolhido pelo F1 macro
de validação e ensemble por média das probabilidades softmax. Para ambos são
salvos resumo, métricas por classe e matrizes de confusão absoluta e normalizada.
O teste de rejeição de Others de `teste_others.py` é um experimento separado e
não é executado por estes scripts.

Cada execução também registra a configuração, assinaturas SHA-256 dos arquivos
de entrada, versões das bibliotecas, a associação de cada linha ao fold de
validação e a distribuição de classes em treino e validação. Isso permite
conferir que os dois baselines usaram as mesmas divisões; não comprova a
identidade com uma execução antiga sem esses registros.

## Colab e retomada

No Colab, os resultados ficam em `/content/results/<nome_do_resultado>/`.
Se o Drive estiver montado, há backup após cada fold em
`/content/drive/MyDrive/tcc_cross_validation/<nome_do_resultado>/`.
Envie somente o script do baseline desejado e os arquivos de dados necessários.
Não é necessário enviar os CSVs brutos nem `pre_processamento.py` ao Colab.
A estrutura mínima é:

```text
/content/
├── cross_validation_bertimbau.py         # ou cross_validation_legalbert_default.py
├── dataset_clean/
│   ├── train_clean.csv
│   └── test_clean.csv
└── results/
    └── class_weights.json
```

Se colocar o script diretamente em `/content`, execute, por exemplo,
`python cross_validation_bertimbau.py`, com `dataset_clean/` e `results/` também
em `/content`. Repita a instalação usada pela cross-validation original:

```python
%pip install -q transformers accelerate pandas scikit-learn
```

O procedimento original não fixou versões. O PyTorch, Matplotlib, Seaborn e as
demais dependências já estavam disponíveis no runtime do Colab. Para evitar
misturar versões depois de uma tentativa de instalação com pins, comece em um
runtime novo antes de executar essa célula.

Para retomar, ajuste `START_FROM_FOLD` no script correspondente (por exemplo,
`3` reaproveita os folds 1 e 2). Os modelos, o CSV parcial e os registros da
execução anterior precisam estar disponíveis localmente ou no backup do Drive.
A retomada verifica a identidade dos arquivos de entrada e a configuração.
Não misture resultados de datasets, modelos ou configurações diferentes.
Para começar outro experimento, use uma pasta de saída nova ou preserve e mova
os resultados anteriores antes de executar. `DRY_RUN=True` usa amostras pequenas
e uma pasta com sufixo `_dry_run`; suas métricas não servem para a comparação final.
