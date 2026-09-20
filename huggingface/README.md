---
language:
- pt
library_name: transformers
pipeline_tag: text-classification
base_model: raquelsilveira/legalbertpt_fp
tags:
- legal
- bert
- document-classification
- ensemble
---

# JurisClass AI — LegalBert-pt ajustado ao SVic

Este repositório contém os cinco modelos de validação cruzada do TCC de Gabriel de Salles Mapeli Couzzi. Cada subpasta `fold_N/best_model/` contém um classificador independente no formato Transformers/Safetensors. O repositório não contém documentos do corpus.

## Tarefa e rótulos

Classificação por página entre Acórdão (`acordao_de_2_instancia`), ARE (`agravo_em_recurso_extraordinario`), Despacho (`despacho_de_admissibilidade`), RE (`peticao_do_RE`) e Sentença (`sentenca`). A categoria residual `outros`/Others foi excluída do treinamento.

O modelo base é [raquelsilveira/legalbertpt_fp](https://huggingface.co/raquelsilveira/legalbertpt_fp). Não foi realizado pré-treinamento do zero. O ajuste usou cinco épocas, taxa de aprendizado 5e-5, lote efetivo 16 e 500 passos de aquecimento. A seleção dos hiperparâmetros foi realizada por busca em grade; os folds foram estratificados por página, sem agrupamento por processo. O fold 5 foi selecionado pelo maior F1-macro de validação interna. Os diretórios `best_model` contêm o estado da última época de cada fold, conforme o código de treinamento.

## Uso de um fold

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "gcouzzi/jurisclass-legalbertpt"
revision = "SUBSTITUA_PELO_HASH_DO_COMMIT"
subfolder = "fold_5/best_model"
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder=subfolder, revision=revision)
model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder=subfolder, revision=revision)
model.eval()
inputs = tokenizer(texto_limpo, return_tensors="pt", truncation=True, max_length=512)
with torch.inference_mode():
    probabilities = model(**inputs).logits.softmax(dim=-1)[0]
```

Use a limpeza de `backend/preprocessing.py` do projeto. Ela remove sequências decorativas repetidas, normaliza quebras de linha, substitui caracteres não permitidos e reduz espaços repetidos. Preserve acentuação e capitalização; rejeite entradas com menos de 50 caracteres após limpeza. O tokenizador trunca a entrada em 512 tokens, incluindo tokens especiais.

Para o ensemble, carregue os cinco folds, confira os mesmos rótulos e calcule a média aritmética das cinco distribuições softmax. Um pipeline de classificação que carregue somente um fold não reproduz o ensemble.

## Resultados existentes

Avaliação em 10.105 páginas do teste pré-processado, restrita às cinco classes conhecidas, sem abstenção:

| Configuração | Acurácia | F1-macro |
|---|---:|---:|
| Fold 5 | 0,918555 | 0,905572 |
| Ensemble dos cinco folds | 0,926670 | 0,912941 |

Os resultados não medem identificação de Others. Um piloto exploratório com 32 páginas Others da validação observou 29 com pontuação acima de 0,90 no fold 5; o piloto não estima a taxa final e não valida um limiar. A amostra principal de rejeição permanece pendente.

## Limitações e uso previsto

Destinado à pesquisa e ao apoio à triagem documental com revisão humana. Pontuações softmax não são probabilidades calibradas de acerto. O modelo pode atribuir confiança elevada a tipos de documento que não aprendeu. O exemplo de limiar 0,60 não constitui configuração validada; o limiar precisa ser determinado para a configuração efetivamente utilizada.

Treinamento e avaliação são por página. A classificação do início de um PDF não equivale à compreensão do documento completo. Os folds internos podem conter páginas relacionadas ao mesmo processo, e não houve avaliação de produtividade, usabilidade ou impacto jurídico real. A melhoria global do ensemble não se estendeu à classe Despacho.

## Proveniência e licença

Os pesos derivados mantêm a necessidade de observar os termos do modelo base e dos dados usados no ajuste. O modelo base declara licença OpenRAIL em sua ficha no Hub; nenhuma licença adicional foi presumida para estes pesos derivados. Consulte os responsáveis antes de redistribuir os pesos sob novos termos.
