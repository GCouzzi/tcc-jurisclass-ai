# Modelos no Hugging Face

Os cinco folds estão públicos em https://huggingface.co/gcouzzi/jurisclass-legalbertpt.

A revisão usada pelo painel e pelo teste é fixada em `backend/model_source.py`. O código baixa `fold_1/best_model` até `fold_5/best_model` para o cache local e executa a inferência na máquina atual. Nenhum token é necessário para o repositório público.

Execute `python teste_others.py` para comparar o fold 5 e o ensemble. Use `python teste_others.py --local-models` para ler os pesos locais.

O Hub armazena e versiona os pesos; a inferência continua sendo executada pela máquina que roda Python. Para hospedá-la no Hugging Face, implante a API em um Space ou Inference Endpoint. O ensemble requer o mesmo fluxo do painel: cinco folds, média das probabilidades e limiar específico do ensemble.

O painel usa o Hub por padrão. As variáveis relevantes são `HF_MODEL_REPO`, `HF_MODEL_REVISION`, `USE_ENSEMBLE`, `FALLBACK_FOLD` e `CONFIDENCE_THRESHOLD`. Deixe `HF_MODEL_REPO` vazio para usar `MODEL_DIR`.