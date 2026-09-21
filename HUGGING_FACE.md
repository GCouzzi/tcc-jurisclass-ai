# Modelos no Hugging Face

Os cinco folds estão públicos em <https://huggingface.co/gcouzzi/jurisclass-legalbertpt>.

A revisão usada pelo painel e por `teste_others.py` é fixada em `backend/model_source.py`. O código baixa `fold_1/best_model` até `fold_5/best_model` para o cache do Hugging Face e executa a inferência na máquina atual. Nenhum token é necessário para o repositório público.

O repositório Git deste projeto não contém os pesos: `results/results_cv` possui somente métricas, relatórios e matrizes de confusão. Os diretórios `results/results_cv/fold_N/best_model`, quando gerados pelo treinamento, são ignorados pelo `.gitignore`.

Apesar do nome dos diretórios no Hub, nas execuções publicadas `best_model` contém o estado ao final das cinco épocas de cada fold. O treinamento de cross-validation usa `save_strategy="no"` e `load_best_model_at_end=False`; portanto, não há seleção do melhor checkpoint entre épocas. O fold 5 foi chamado de “melhor fold” porque obteve o maior F1-macro de validação entre os cinco modelos finais.

## Uso no painel

O painel usa o Hub por padrão. Se `HF_MODEL_REPO` estiver ausente ou vazio, o Docker Compose usa `gcouzzi/jurisclass-legalbertpt`. `HF_MODEL_REVISION` fixa a versão baixada, evitando que uma alteração futura no Hub mude silenciosamente o modelo executado.

As variáveis relevantes são `HF_MODEL_REPO`, `HF_MODEL_REVISION`, `HF_TOKEN`, `USE_ENSEMBLE`, `FALLBACK_FOLD` e `CONFIDENCE_THRESHOLD`. O limiar é opcional: sem `CONFIDENCE_THRESHOLD`, a classe de maior pontuação é retornada normalmente; com a variável definida, entradas abaixo dela são encaminhadas à revisão manual.

O Hub apenas armazena e versiona os pesos. A inferência continua sendo executada pela máquina ou pelo contêiner que roda a API. O ensemble carrega os cinco folds e calcula a média aritmética das cinco distribuições softmax.

## Teste de rejeição de Others

Execute `python teste_others.py` para comparar o fold 5 e o ensemble. O script baixa os mesmos pesos fixados no código, escolhe um limiar usando a validação e só então mede esse limiar no teste. O relatório é gravado em `results/rejection/resultado_others.json`.

A opção `python teste_others.py --local-models` existe para quem já possui os cinco pesos em `results/results_cv/fold_N/best_model`, por exemplo após executar `cross_validation.py`. Esses pesos locais não acompanham o repositório Git.
