# Projeto: Previsão de Tempo até Desligamento Voluntário

Este projeto implementa um pipeline de MLOps para prever, via regressão, quantos meses faltam para um colaborador pedir desligamento voluntário da empresa. Utiliza MLflow para rastreamento de experimentos, exportação do modelo em ONNX, Makefile para automação e variáveis de ambiente em `.env`.

## Estrutura do Projeto

```
classification_model_mlops/
├── Makefile
├── requirements.txt
├── .env.example
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── predict.py
└── data/
    └── colaboradores.csv (você deve fornecer)
```

## Passos para Executar

### 1. Instale as dependências

```bash
make install
```

### 2. Configure as variáveis de ambiente

Copie o arquivo de exemplo e edite conforme necessário:

```bash
cp .env.example .env
```

Edite `.env` para ajustar o endereço do MLflow, se necessário.

### 3. Prepare os dados

Coloque seu arquivo de colaboradores em `data/colaboradores.csv`.

### 4. Pré-processamento dos dados

```bash
make preprocess
```

Gera `data/processed.csv` com as features e o alvo `meses_ate_desligamento`.

### 5. Treinamento do modelo

```bash
make train
```

- Treina um modelo de regressão Random Forest.
- Faz tracking dos experimentos com MLflow.
- Exporta o modelo para `model/model.onnx`.

### 6. Predição

```bash
make predict
```

Executa um exemplo de predição usando o modelo ONNX.

### 7. MLflow UI (opcional)

Para visualizar experimentos:

```bash
make mlflow-ui
```

Acesse: http://localhost:5000

## Observações
- O pipeline considera apenas colaboradores que pediram desligamento voluntário (status_code == 4).
- O alvo é o número de meses até o desligamento voluntário, calculado a partir da coluna `dt_yearmonth`.
- Adapte o caminho do dataset em `src/data_preprocessing.py` se necessário.

## Requisitos
- Python 3.8+
- Linux

## Contato
Dúvidas ou sugestões? Abra uma issue ou entre em contato.
