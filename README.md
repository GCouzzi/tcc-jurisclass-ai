<div align="center">
  <h1>⚖️ JurisClass AI</h1>
  <p><strong>Classificador Inteligente de Documentos Jurídicos Brasileiros utilizando LegalBERT e Ensemble Learning</strong></p>
</div>

<br>

## 📖 Sobre o Projeto
O **JurisClass AI** é um sistema completo (End-to-End) focado na classificação automática de documentos jurídicos oriundos do poder judiciário brasileiro (como o STF). Ele utiliza modelos baseados em arquitetura Transformers, especificamente o **LegalBERT-pt** (pré-treinado em textos legais do Brasil), aplicando técnicas de _Fine-Tuning_ e _Ensemble Learning (Soft Voting)_ sobre uma estratégia de 5-Fold Cross-Validation.

Além de todo o pipeline de experimentação, limpeza de dados e treinamento, o repositório conta com uma solução de software distribuída em contêineres: uma **API Backend em FastAPI** (com suporte a OCR via Tesseract) e um **Painel Frontend (Web App)** moderno para interação do usuário final.

---

## 🏗️ Arquitetura do Sistema

O projeto é dividido em três grandes blocos:

1. **Pipeline de Machine Learning (Treinamento e Baselines)**
   - Algoritmos para extração, balanceamento e limpeza do Dataset.
   - Scripts de _Grid Search_ para otimização de hiperparâmetros (Hugging Face Trainer).
   - Validação Cruzada Estratificada de 5 Folds (`cross_validation.py`).
   - Avaliação de Baselines: TF-IDF + Regressão Logística, TF-IDF + SVM, Bertimbau e LegalBERT Vanilla.

2. **Backend (FastAPI)**
   - API de Inferência REST rápida e escalável.
   - Integração com OCR Tesseract para extrair texto de PDFs escaneados.
   - Suporte dinâmico para rodar o modelo no modo _Single Fold_ (Best Fold) ou em modo **Ensemble** (agrupando os 5 modelos gerados no Cross-Validation por meio de Soft Voting).

3. **Frontend (Painel Web)**
   - Aplicação SPA vanilla (HTML, CSS, JS) responsiva e intuitiva.
   - Upload interativo de arquivos, barra de carregamento e exibição formatada das previsões e da confiança do modelo.

---

## 🤖 Modelos e Inferência

O coração deste projeto é o `raquelsilveira/legalbertpt_fp`, submetido a um processo de fine-tuning.

O sistema trabalha com **5 classes documentais**:
* Acórdão (Acórdão de 2ª Instância)
* Agravo (ARE)
* Despacho (Despacho de Admissibilidade)
* Recurso Extraordinário (Petição do RE)
* Sentença

O modelo final em produção emprega um sistema de **Soft Voting Ensemble**. Os tensores e configurações de cada um dos 5 folds são carregados na memória. Para cada requisição, os cinco modelos avaliam o documento, as probabilidades (softmax) de cada modelo são somadas e divididas, gerando uma predição robusta, minimizando a variância e mitigando o risco de overfitting para uma distribuição específica.

---

## 🚀 Como Executar Localmente (Docker)

A maneira mais fácil e segura de rodar a aplicação completa é utilizando o Docker Compose, que subirá automaticamente o Frontend e o Backend.

### Pré-requisitos
- [Docker](https://www.docker.com/) e [Docker Compose](https://docs.docker.com/compose/) instalados.
- Ter os pesos dos modelos (`model.safetensors`) gerados e posicionados corretamente na pasta `results/results_cv/fold_X/best_model`. *(Nota: devido ao tamanho, os tensores são versionados via Git LFS).*

### Passos para Inicialização

1. Clone o repositório (Certifique-se de usar o Git LFS se desejar baixar os tensores hospedados no repositório):
   ```bash
   git clone https://github.com/GCouzzi/implementacao-tcc-testes.git
   cd implementacao-tcc-testes
   git lfs pull
   ```

2. Construa as imagens e levante os serviços:
   ```bash
   docker compose up -d --build
   ```

3. Acesse a aplicação:
   - **Painel Web (Frontend):** [http://localhost:5500](http://localhost:5500)
   - **Documentação da API (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

*(Se desejar alterar do modo Ensemble para um único modelo leve, edite a variável `USE_ENSEMBLE=true` para `false` no arquivo `docker-compose.yml` e reinicie os containers).*

---

## 📂 Estrutura de Diretórios

```text
├── backend/                   # API em FastAPI, integração OCR e scripts de inferência
├── frontend/                  # Frontend (HTML, CSS, JS puros)
├── baselines/                 # Scripts de treinamento dos modelos de baseline (TF-IDF, SVM, etc)
├── dataset_clean/             # (Ignorado no Git) Dados limpos gerados pelo pré-processamento
├── dataset_victor/            # (Ignorado no Git) Dataset bruto original
├── results/                   # Resultados, matrizes de confusão e arquivos CSV das métricas
│   └── results_cv/            # Onde os modelos (Folds 1 a 5) devem ficar salvos
├── cross_validation.py        # Script de Fine-tuning (Stratified 5-Fold)
├── grid_search.py             # Busca manual de hiperparâmetros com Hugging Face Trainer
├── pre_processamento.py       # Limpeza e balanceamento do dataset
├── docker-compose.yml         # Orquestração do Backend e Frontend
└── .gitattributes             # Regras de atributos do Git
```

---

## 📈 Preparação do Dataset e Treinamento

Os diretórios `dataset_victor/` e `dataset_clean/` são ignorados pelo Git. Portanto, o dataset precisa ser obtido separadamente e colocado manualmente no projeto.

Todos os comandos desta seção devem ser executados na raiz do repositório, isto é, no mesmo diretório de `pre_processamento.py`.

### 1. Organize os arquivos brutos

Depois de obter o dataset, crie a pasta `dataset_victor/` na raiz e coloque nela estes três CSVs:

```text
dataset_victor/
├── train_small.csv
├── validation_small.csv
└── test_small.csv
```

Os scripts atuais não leem os arquivos `*_medium.csv` nem os PDFs existentes em `dataset_victor/small/`. Para o pipeline descrito neste repositório, somente os três CSVs acima são necessários. Cada CSV deve possuir, pelo menos, as colunas `body` e `document_type`.

### 2. Crie e configure o ambiente Python

É recomendado usar Python 3.11 em um ambiente virtual:

```bash
python -m venv .venv
```

No Windows PowerShell, ative o ambiente com:

```powershell
.\.venv\Scripts\Activate.ps1
```

No Linux ou macOS, use:

```bash
source .venv/bin/activate
```

Instale as dependências do backend e as bibliotecas adicionais usadas no treinamento:

```bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt pandas scikit-learn matplotlib seaborn accelerate
```

### 3. Execute o pré-processamento

Como `dataset_clean/` também é ignorado pelo Git, crie o diretório antes da primeira execução:

```bash
mkdir dataset_clean
python pre_processamento.py
```

O script remove registros da classe `outros`, limpa o conteúdo da coluna `body`, descarta textos com menos de 50 caracteres e gera:

```text
dataset_clean/
├── train_clean.csv
├── val_clean.csv
└── test_clean.csv

results/
└── class_weights.json
```

O arquivo `class_weights.json` é calculado somente a partir das classes de `train_clean.csv`.

### 4. Execute o Grid Search

Para testar as combinações definidas em `grid_search.py`, execute:

```bash
python grid_search.py
```

O script treina com `train_clean.csv`, avalia cada combinação com `val_clean.csv` e salva o ranking em `results/grid_search_results.csv`. O conjunto de teste não participa da busca de hiperparâmetros.

No Google Colab, ative uma GPU, copie o projeto e execute:

```python
!pip install -q transformers accelerate pandas scikit-learn
!python grid_search.py --overwrite
```

O script localiza os arquivos tanto nas pastas do projeto quanto diretamente em `/content`. Também é possível informar `--train`, `--validation` e `--class-weights`.

> O resultado do Grid Search não é carregado automaticamente por `cross_validation.py`. Se uma configuração diferente for escolhida, atualize manualmente as constantes `BEST_EPOCHS`, `BEST_BATCH_SIZE`, `BEST_LR` e `BEST_WARMUP` no início do script.

### 5. Treine e avalie o modelo final

```bash
python cross_validation.py
```

Esse script usa `train_clean.csv` para criar os cinco folds estratificados. Em cada rodada, quatro folds são usados para treinamento e um para validação. O arquivo `val_clean.csv` não é usado nessa etapa. O `test_clean.csv` permanece separado e é usado somente na avaliação final do melhor fold e do ensemble.

Os modelos e relatórios são salvos em `results/results_cv/`. Para retomar uma execução interrompida, ajuste a constante `START_FROM_FOLD` em `cross_validation.py`.

### Uso de train, validation e test por script

| Script | Treino | Validação | Teste |
|---|---|---|---|
| `pre_processamento.py` | Limpa `train_small.csv`, gera `train_clean.csv` e calcula os pesos das classes. | Limpa `validation_small.csv` e gera `val_clean.csv`. | Limpa `test_small.csv` e gera `test_clean.csv`. |
| `grid_search.py` | Treina cada combinação; por padrão usa uma amostra estratificada de até 3.000 textos. | Calcula as métricas usadas para ordenar as combinações. | Não utiliza. |
| `cross_validation.py` | Origina os cinco folds de treino e validação. | Não utiliza `val_clean.csv`; a validação vem dos folds do conjunto de treino. | Avalia o melhor fold e o ensemble somente após o treinamento. |
| `baseline_bertimbau.py` | Ajusta o modelo Bertimbau. | Avalia a cada época e seleciona o melhor checkpoint pelo F1 macro. | Produz as métricas e a matriz de confusão finais. |
| `baseline_legalbert_default.py` | Ajusta o LegalBERT com a configuração padrão. | Avalia a cada época e seleciona o melhor checkpoint pelo F1 macro. | Produz as métricas e a matriz de confusão finais. |
| `baseline_tfidf_lr.py` | Ajusta o TF-IDF e a regressão logística. | Gera uma predição, mas ela não é usada nas métricas salvas atualmente. | Produz todas as métricas e a matriz de confusão finais. |
| `baseline_tfidf_svm.py` | Ajusta o TF-IDF e o SVM calibrado. | Gera uma predição, mas ela não é usada nas métricas salvas atualmente. | Produz todas as métricas e a matriz de confusão finais. |

### 6. Execute a aplicação com Docker

Depois que `results/class_weights.json` estiver disponível, execute. Os modelos serão obtidos do Hugging Face Hub na inicialização:

```bash
docker compose up -d --build
```

O backend usa os modelos treinados para inferência e não lê diretamente os CSVs do dataset. O frontend fica disponível em [http://localhost:5500](http://localhost:5500), e a documentação da API em [http://localhost:8000/docs](http://localhost:8000/docs).

*Nota: o treinamento de Transformers é significativamente mais rápido com uma GPU compatível com CUDA.*

---

## 🛠️ Tecnologias Utilizadas

- **Modelos & IA:** PyTorch, HuggingFace Transformers, Scikit-learn
- **Backend:** FastAPI, Uvicorn, Tesseract OCR (Pytesseract), PyMuPDF (fitz)
- **Frontend:** Vanilla JS, HTML5, CSS3 Moderno
- **Infraestrutura:** Docker, Docker Compose, Hugging Face Hub
- **Análise de Dados:** Pandas, Numpy, Seaborn, Matplotlib

---
*Projeto acadêmico.*

## Teste com documentos Others

Execute `python teste_others.py`. O script compara o melhor fold (fold 5) com o ensemble dos cinco folds, escolhe um limiar próprio para cada configuração usando somente a validação e grava um único arquivo: `results/rejection/resultado_others.json`. Por padrão, usa 2.129 páginas Others e 2.129 conhecidas em cada partição.

Os pesos são baixados do repositório público https://huggingface.co/gcouzzi/jurisclass-legalbertpt na revisão fixa registrada no código. Para usar os pesos locais, acrescente `--local-models`.

### Ativar a revisão por limiar no painel

Depois de executar `teste_others.py`, use o limiar selecionado para a mesma configuração do painel. Com o ensemble padrão (`USE_ENSEMBLE=true`), crie um arquivo `.env` ao lado de `docker-compose.yml`:

```env
CONFIDENCE_THRESHOLD=0.98765
```

Use ponto como separador decimal e substitua o exemplo pelo valor de `configurations.ensemble_5_folds.validation.selected_threshold.threshold` no JSON. Reinicie os serviços com `docker compose up -d --build`. Para o fold 5 individual, configure também `USE_ENSEMBLE=false` e use o limiar de `configurations.best_fold_5`.

Se `CONFIDENCE_THRESHOLD` estiver ausente ou vazio, o mecanismo de revisão fica desativado: o painel retorna normalmente a classe de maior pontuação. Quando a variável estiver definida, pontuações abaixo do limiar retornam `Não classificado` e `Revisão manual necessária`; pontuações iguais ou superiores continuam classificadas.