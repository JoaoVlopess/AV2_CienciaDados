# Projeto 2 — Preparação dos dados do ENEM 2023 (Linguagens)

Script que lê a base nacional **completa** do ENEM 2023, seleciona as 12 features de perfil,
valida e padroniza os códigos, cria 3 features derivadas e salva dois arquivos Parquet
prontos para a etapa de modelagem, mais relatórios de auditoria.
A especificação seguida é `docs/documentacao_tratamento_enem_2023_lc.md`.

> Esta etapa **não** treina modelos, não divide treino/teste, não imputa nem padroniza valores.

## 1. Pré-requisitos

- Python 3.10 ou superior
- O pacote oficial do Inep extraído na raiz do projeto, de modo que exista:
  `microdados_enem_2023/DADOS/MICRODADOS_ENEM_2023.csv`
  (download: <https://download.inep.gov.br/microdados/microdados_enem_2023.zip>)
- Cerca de 2 GB de RAM livre (com o bloco padrão de 500 mil linhas)

Instale as dependências (de preferência num ambiente virtual):

```powershell
python -m venv .venv
.venv\Scripts\activate          # no Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Como rodar

Sempre a partir da **raiz do projeto** (a pasta que contém `src/`):

```powershell
python -m pytest -q                       # 1) testes com dados fictícios (segundos)
python -m src.dados.executar_tratamento   # 2) tratamento da base completa (alguns minutos)
```

Opções:

| Opção | Para que serve |
|---|---|
| `--entrada CAMINHO` | CSV em outro local (padrão: `microdados_enem_2023/DADOS/MICRODADOS_ENEM_2023.csv`) |
| `--saida PASTA` | Pasta de saída (padrão: `dados/processed`) |
| `--bloco N` | Linhas por bloco; use `--bloco 200000` se faltar memória. O resultado é o mesmo |
| `--sem-hash` | Pula o SHA-256 do CSV (mais rápido, menos rastreável) |
| `--total-esperado N` | Total de linhas para conferência (padrão: 3.933.955) |

Código de saída: `0` = OK; `1` = alguma invariante falhou (Parquets **não** são gravados,
só os relatórios explicando o motivo); `2` = arquivo de origem ausente/inválido.

## 3. O que é gerado

```text
dados/processed/
├── enem_2023_lc_diagnostico.parquet   # TODAS as linhas, 17 colunas
├── enem_2023_lc_elegiveis.parquet     # só quem tem NU_NOTA_LC (inclui nota 0), 17 colunas
└── relatorios/
    ├── resumo_processamento.json      # totais, hash, regras, alertas e invariantes
    ├── qualidade_colunas.csv          # 1 linha por coluna: tipos, ausentes, inválidos, distintos
    ├── frequencias_categorias.csv     # contagem/percentual por categoria (diagnóstico e elegíveis)
    └── exemplos_anomalias.csv         # códigos fora do dicionário e quantas vezes aparecem
```

As 17 colunas: `NU_INSCRICAO` (só controle), as 12 originais (`Q006`, `Q001`, `Q002`, `TP_ESCOLA`,
`TP_ST_CONCLUSAO`, `TP_FAIXA_ETARIA`, `Q005`, `Q024`, `Q025`, `IN_TREINEIRO`, `SG_UF_PROVA`,
`TP_LINGUA`), as 3 derivadas (`RENDA_PER_CAPITA_APROX`, `ESCOLARIDADE_MAX_RESP`,
`PERFIL_ACESSO_DIGITAL`) e o alvo `NU_NOTA_LC`.

Os CSVs de relatório usam `;` como separador, vírgula decimal e UTF-8 com BOM (abrem direto no Excel).

## 4. Usando a saída na modelagem

```python
import pandas as pd
from src.dados import configuracao as cfg

df = pd.read_parquet("dados/processed/enem_2023_lc_elegiveis.parquet")
X = df[cfg.FEATURES_ORIGINAIS + cfg.FEATURES_DERIVADAS]   # NU_INSCRICAO nunca entra em X
y = df[cfg.COLUNA_ALVO]
```

Imputação, codificação e padronização devem ser ajustadas **só no treino**, dentro do protocolo de validação.

### 4.1 Regressão linear no Google Colab — `notebooks/regressao_linear_lc.ipynb`

1. Abra o notebook no Colab (`Arquivo → Fazer upload de notebook`).
2. No painel de arquivos (ícone de pasta, à esquerda), envie `dados/processed/enem_2023_lc_elegiveis.parquet` (≈ 31 MB).
   Não usa Google Drive.
3. `Ambiente de execução → Executar tudo`. Se o arquivo não tiver sido enviado, a Seção 2 abre um botão de upload.
4. No fim, o notebook baixa `saidas_regressao_linear_lc.zip`; descompacte-o na raiz do projeto
   (o Colab apaga os arquivos ao desconectar).

O notebook treina baseline (`DummyRegressor`), `LinearRegression` e `Ridge` (com busca de `alpha`) na validação cruzada
de 3 dobras do desenvolvimento e gera (dentro do zip):

| Saída | Conteúdo |
|---|---|
| `dados/splits/particoes_lc.parquet` | Divisão 80/20 (`random_state=42`) e dobra de cada participante — **reutilizada** se já existir |
| `resultados/experimentos.csv` | Uma linha por experimento: hipótese, features, preparação, parâmetros, métricas por dobra, média, dp, tempos, conclusão |
| `resultados/*_lc*.csv` | Comparação, busca do Ridge, erros por faixa/escola/UF, coeficientes, curva de aprendizado |
| `resultados/figuras/lc_*.png` | Gráficos do desenvolvimento |

O **teste não é avaliado** (`AVALIAR_TESTE = False`, Seção 15) até a equipe fixar a solução final pela validação cruzada.
Para os modelos lineares, `IN_TREINEIRO` (= `TP_ST_CONCLUSAO == 3`) e `Q025` (contido em `PERFIL_ACESSO_DIGITAL`) saem de `X`,
e `TP_ESCOLA` entra só como indicador de escola privada; a Seção 6 do notebook mostra as redundâncias.

### 4.2 Conjuntos de features para comparar modelos — `src/conjuntos/`

Planejamento e justificativas em `docs/conjuntos_features_lc.md`. Gera um parquet por conjunto (C01–C15 e 8 ablações),
só com o **desenvolvimento**, mais um `manifesto.json` com colunas, hipóteses e tipos:

```powershell
python -m src.conjuntos.gerar_conjuntos               # amostra de 300.000 (100 mil por dobra) em dados/conjuntos/
python -m src.conjuntos.gerar_conjuntos --amostra 0   # desenvolvimento completo
```

Na primeira execução, o script lê 28 colunas extras do CSV original (`dados/processed/enem_2023_lc_ampliada.parquet`)
e cria `dados/splits/particoes_lc.parquet` com a mesma regra do notebook. Se esse arquivo já existir, ele é reutilizado.

```python
from src.conjuntos.preprocessamento import carregar_conjunto, montar_pipeline
X, y, dobras = carregar_conjunto("C04")
pipe = montar_pipeline("C04", "arvore", DecisionTreeRegressor(random_state=42))  # ou "linear", "svm", "boosting"
```

### 4.3 Base final do conjunto escolhido (C07)

A equipe escolheu o **C07** (23 features; 2º lugar na triagem, com menos da metade das colunas do 1º).
Para a modelagem final, ele é exportado com **todos** os participantes elegíveis e as colunas `CONJUNTO`/`DOBRA`:

```powershell
python -m src.conjuntos.exportar_conjunto C07   # -> dados/processed/enem_2023_lc_c07.parquet (2.822.643 linhas, 38 MB)
```

Features, ausências e preparação necessária para o treino: `dados/processed/README_C07.md`.

### 4.4 Triagem dos conjuntos com 3 modelos — `src/modelagem/triagem_conjuntos.py`

Comparação que levou à escolha do C07. Compara os conjuntos gerados em 4.2 com regressão linear (`LinearRegression`), árvore de decisão
(`DecisionTreeRegressor(max_depth=12, min_samples_leaf=100)`) e boosting (`HistGradientBoostingRegressor`),
além do baseline pela média, nas 3 dobras fixas do desenvolvimento:

```powershell
python -m src.modelagem.triagem_conjuntos                                   # todos os conjuntos x 3 modelos
python -m src.modelagem.triagem_conjuntos --conjuntos C02 C04 --modelos boosting
```

| Saída | Conteúdo |
|---|---|
| `resultados/experimentos_triagem.csv` | uma linha por experimento, com as mesmas colunas de `experimentos.csv` (+ `id_conjunto`, `familia`, `n_features_saida`) |
| `resultados/triagem_conjuntos_lc.csv` | uma linha por conjunto: RMSE/MAE/R² de cada modelo e diferença para C02 (atual) e C11 (completo) |
| `resultados/figuras/lc_triagem_conjuntos.png` | RMSE ± dp por conjunto e modelo |
| `resultados/figuras/lc_triagem_ablacao.png` | quanto o RMSE piora ao retirar cada bloco de C11 |

Experimentos já gravados são pulados (`--refazer` força). O arquivo é separado de `experimentos.csv` para não ser
sobrescrito pelo zip do Colab; as colunas são as mesmas, então dá para concatenar os dois.

## 5. Estrutura do código

| Arquivo | Responsabilidade |
|---|---|
| `src/dados/configuracao.py` | Caminhos, as 14 colunas de entrada, domínios, faixas de renda, rótulos |
| `src/dados/leitura.py` | Conferência do arquivo (encoding, separador, colunas, edição, hash) e leitura em blocos |
| `src/dados/tratamento.py` | Padronização e validação de domínios, sem imputação |
| `src/dados/engenharia_features.py` | As três features derivadas (funções puras, não leem o alvo) |
| `src/dados/validacao.py` | Contagens acumuladas, duplicatas, relatórios e invariantes |
| `src/dados/executar_tratamento.py` | Orquestração e gravação das saídas |
| `src/conjuntos/` | Base ampliada, features novas, definição dos conjuntos e preparação por família de modelo |
| `src/modelagem/triagem_conjuntos.py` | Triagem: conjuntos × (linear, árvore, boosting) na validação cruzada |
| `tests/test_tratamento.py` | Testes da seção 13 da especificação |
| `tests/test_conjuntos.py` | Features novas e todos os conjuntos × famílias com dados fictícios |
| `tests/test_triagem.py` | Triagem de ponta a ponta (gravação, retomada, resumo e figuras) com dados fictícios |
| `notebooks/regressao_linear_lc.ipynb` | Modelagem: baseline, regressão linear e Ridge (Colab) |
