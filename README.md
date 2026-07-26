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
   - Integração com OCR Tesseract para extrair texto de PDFs escaneados ou imagens.
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
├── painel/                    # Frontend (HTML, CSS, JS puros)
├── baselines/                 # Scripts de treinamento dos modelos de baseline (TF-IDF, SVM, etc)
├── dataset_clean/             # (Ignorado no Git) Dados limpos gerados pelo pré-processamento
├── dataset_victor/            # (Ignorado no Git) Dataset bruto original
├── results/                   # Resultados, matrizes de confusão e arquivos CSV das métricas
│   └── results_cv/            # Onde os modelos (Folds 1 a 5) devem ficar salvos
├── cross_validation.py        # Script de Fine-tuning (Stratified 5-Fold)
├── grid_search.py             # Otimização de Hiperparâmetros (Optuna/Trainer)
├── pre_processamento.py       # Limpeza e balanceamento do dataset
├── docker-compose.yml         # Orquestração do Backend e Frontend
└── .gitattributes             # Configuração LFS para os arquivos de tensor
```

---

## 📈 Treinamento e Reprodutibilidade

Se você deseja reproduzir os resultados e treinar os modelos do zero:

1. Instale os requerimentos num ambiente Python virtual: `pip install -r backend/requirements.txt`
2. Gere o dataset processado executando: `python pre_processamento.py`
3. Descubra os melhores parâmetros rodando: `python grid_search.py`
4. Treine o modelo final utilizando: `python cross_validation.py`

*Nota: O treinamento requer de preferência uma GPU (CUDA). O script suporta salvamento contínuo; se o treinamento for interrompido, ele pode ser reiniciado ajustando a constante `START_FROM_FOLD` no script.*

---

## 🛠️ Tecnologias Utilizadas

- **Modelos & IA:** PyTorch, HuggingFace Transformers, Scikit-learn
- **Backend:** FastAPI, Uvicorn, Tesseract OCR (Pytesseract), PyMuPDF (fitz)
- **Frontend:** Vanilla JS, HTML5, CSS3 Moderno
- **Infraestrutura:** Docker, Docker Compose, Git LFS
- **Análise de Dados:** Pandas, Numpy, Seaborn, Matplotlib

---
*Projeto acadêmico.*
