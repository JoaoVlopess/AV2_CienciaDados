# Base do conjunto C07 — ENEM 2023, previsão da nota de Linguagens (`NU_NOTA_LC`)

Arquivo: **`enem_2023_lc_c07.parquet`** (38 MB)
**2.822.643 linhas × 27 colunas**: todos os participantes do ENEM 2023 com nota de Linguagens, já divididos
em desenvolvimento e teste, com as **23 features do conjunto C07**.

```powershell
python -m src.conjuntos.exportar_conjunto C07     # gera/atualiza o arquivo (~15 s; a 1ª vez lê o CSV, ~1 min)
```

```python
import pandas as pd
df = pd.read_parquet("dados/processed/enem_2023_lc_c07.parquet")
dev = df[df["CONJUNTO"] == "desenvolvimento"].reset_index(drop=True)   # use SÓ isto até fixar a solução final
X = dev.drop(columns=["NU_INSCRICAO", "CONJUNTO", "DOBRA", "NU_NOTA_LC"])  # 23 features
y = dev["NU_NOTA_LC"]
```

---

## 1. Por que o C07

Na triagem de 24 conjuntos (`resultados/triagem_conjuntos_lc.csv`, amostra de 300 mil participantes do desenvolvimento,
validação cruzada em 3 dobras), o C07 ficou em **2º lugar nos três modelos**, atrás apenas do conjunto completo (C11, 54 colunas):

| RMSE na validação (pontos) | Conjunto atual (C02, 15 col.) | **C07 (23 col.)** | Completo (C11, 54 col.) |
|---|---:|---:|---:|
| Regressão linear | 66,32 | **65,68** | 64,87 |
| Árvore de decisão | 66,70 | **66,31** | 66,28 |
| Boosting (HistGB) | 65,61 | **64,87** | 64,10 |

Baseline pela média: 75,57. O C07 usa menos da metade das colunas do C11 e perde ~0,8 ponto (empata na árvore):
foi escolhido por ser mais leve. Hipótese do conjunto: **a ocupação dos pais e a trajetória escolar (rede federal ×
estadual, atraso escolar) acrescentam sinal ao perfil socioeconômico atual.**

## 2. Linhas e partições

| Parte | Linhas | `DOBRA` | Uso |
|---|---:|---|---|
| Desenvolvimento (80%) | 2.258.114 | 1, 2, 3 (752.705 / 752.705 / 752.704) | comparar modelos, buscar hiperparâmetros, curva de aprendizado |
| Teste (20%) | 564.529 | 0 | **só a avaliação final**, depois de fixar a solução |

- Mesma divisão de `dados/splits/particoes_lc.parquet` e do notebook (`train_test_split(test_size=0.2, random_state=42)` +
  `KFold(3, shuffle=True, random_state=42)` no desenvolvimento).
- As dobras da validação cruzada saem da coluna `DOBRA`:
  ```python
  import numpy as np
  dobras = [(np.flatnonzero(dev["DOBRA"] != k), np.flatnonzero(dev["DOBRA"] == k)) for k in (1, 2, 3)]
  # cross_validate(pipeline, X, y, cv=dobras, ...)
  ```
- Mesmos participantes da base tratada (`enem_2023_lc_elegiveis.parquet`), na mesma ordem: inclui treineiros, todas as idades
  e as **2.169 notas zero** (1.736 no desenvolvimento). Nenhuma linha foi removida por falta de feature.

## 3. As 27 colunas

### Controle e alvo

| Coluna | Tipo | Conteúdo |
|---|---|---|
| `NU_INSCRICAO` | texto | identificador (máscara do Inep). **Nunca entra em X** |
| `CONJUNTO` | texto | `desenvolvimento` / `teste` |
| `DOBRA` | inteiro | 1–3 no desenvolvimento, 0 no teste |
| `NU_NOTA_LC` | decimal | **alvo**. Desenvolvimento: média 518,1, dp 75,5, mediana 523,1, 1% = 323,3, 99% = 673,5, máx. 820,8 |

### 23 features

Os códigos ficam como **texto** (`"1"`, `"A"`), mesmo os numéricos do dicionário, para não serem lidos como medidas.
Contagens, binárias e contínuas ficam como **decimal** (`float64`). Percentuais no desenvolvimento.

**Originais do Inep (14)**

| Coluna | O que é | Valores | Tipo para modelar |
|---|---|---|---|
| `Q006` | renda familiar | `A` nenhuma … `Q` > R$ 26.400 (17 faixas) | ordinal |
| `Q001` / `Q002` | escolaridade do pai / da mãe | `A` nunca estudou … `G` pós; **`H` não sei** (9,5% / 2,9%) | ordinal + especial |
| `TP_ESCOLA` | tipo de escola do EM | `1` não respondeu (61,0%), `2` pública (31,0%), `3` privada (8,0%) | categórica |
| `TP_ST_CONCLUSAO` | situação do EM | `1` concluiu (41,2%), `2` conclui em 2023 (39,0%), `3` depois de 2023 (19,4%), `4` não cursa (0,4%) | categórica |
| `TP_FAIXA_ETARIA` | faixa etária | `1` < 17 … `10` = 25 anos, `11`–`19` faixas de 5 anos, `20` > 70 | categórica (ordenada) |
| `Q005` | moradores (com o participante) | 1–20, mediana 4 | numérica |
| `Q024` | computadores | `A` nenhum … `E` 4 ou mais | ordinal |
| `Q025` | internet | `A` não, `B` sim | binária (letra) |
| `IN_TREINEIRO` | treineiro | 0 / 1 (19,4%) | binária |
| `SG_UF_PROVA` | UF da prova (não é residência) | 27 siglas | categórica |
| `TP_LINGUA` | língua estrangeira escolhida | 0 inglês (57,5%), 1 espanhol (42,5%) | binária |
| `Q003` / `Q004` | ocupação do pai / da mãe | `A` grupo 1 (lavrador…) … `E` grupo 5 (médico, engenheiro…); **`F` não sei** (12,1% / 8,7%) | ordinal + especial |

**Derivadas pela equipe (9)**, todas calculadas só com a própria linha, sem olhar a nota:

| Coluna | Como é calculada | Valores | Tipo |
|---|---|---|---|
| `RENDA_PER_CAPITA_APROX` | ponto médio da faixa de `Q006` ÷ `Q005` | R$; mediana 550, 99% = 5.940, máx. 23.100. **Vazio em 1,6%** (`Q006 = Q`, faixa sem teto) | numérica, assimétrica |
| `ESCOLARIDADE_MAX_RESP` | maior entre `Q001` e `Q002`, se ambos em A–G | `A`–`G`; **`INDETERMINADA`** se algum é "não sei" (10,4%) | ordinal + especial |
| `PERFIL_ACESSO_DIGITAL` | `Q024` × `Q025` | `COMPUTADOR_E_INTERNET` 51,1%, `SOMENTE_INTERNET` 40,5%, `SEM_COMPUTADOR_SEM_INTERNET` 7,9%, `SOMENTE_COMPUTADOR` 0,5% | categórica |
| `OCUPACAO_MAX_RESP` | maior entre `Q003` e `Q004`, se ambos em A–E | `A`–`E`; **`INDETERMINADA`** se algum é "não sei" (17,4%) | ordinal + especial |
| `CAPITAL_CULTURAL_RESP` | nível de `ESCOLARIDADE_MAX_RESP` (0–6) + nível de `OCUPACAO_MAX_RESP` (0–4) | 0–10, mediana 6. **Vazio em 22,1%** (algum indeterminado) | numérica |
| `N_RESP_NAO_SEI` | nº de "não sei" em `Q001`–`Q004` | 0 (77,9%) … 4 (0,6%) | numérica (contagem) |
| `TIPO_ESCOLA_DETALHADO` | dependência administrativa da escola quando informada; senão `TP_ESCOLA` | `NAO_INFORMADA` 61,0%, `ESTADUAL` 17,1%, `PUBLICA_SEM_DETALHE` 11,3%, `PRIVADA` 8,7%, `FEDERAL` 1,6%, `MUNICIPAL` 0,2% | categórica |
| `ATRASO_ESCOLAR` | 1 se conclui o EM em 2023 (`TP_ST_CONCLUSAO = 2`) com 19 anos ou mais | 0 / 1 (4,2%) | binária |
| `ANOS_DESDE_CONCLUSAO` | de `TP_ANO_CONCLUIU`: 1 = 2022 … 16 = 2007; 17 = "antes de 2007" (limite inferior) | 1–17, mediana 3. **Vazio em 62,6%** | numérica |

## 4. Ausências: onde estão e o que significam

Não há ausentes nas colunas de texto. Os vazios (`NaN`) estão em 3 colunas numéricas e **não são aleatórios**:

| Coluna | Vazios (total) | Motivo |
|---|---:|---|
| `RENDA_PER_CAPITA_APROX` | 44.132 (1,6%) | `Q006 = Q` (> R$ 26.400) não tem ponto médio; não inventamos um valor |
| `CAPITAL_CULTURAL_RESP` | 623.881 (22,1%) | escolaridade ou ocupação de algum responsável é "não sei" |
| `ANOS_DESDE_CONCLUSAO` | 1.766.944 (62,6%) | **100%** de quem ainda não concluiu (`TP_ST_CONCLUSAO` 2, 3, 4) e 9,3% de quem concluiu mas não informou o ano |

Os três vazios carregam informação (renda muito alta, responsável desconhecido, ainda na escola), por isso o tratamento
recomendado é **imputar + indicador de ausência**, não descartar linhas.

## 5. Limpezas e preparação necessárias antes de treinar

A base está **limpa** (códigos validados contra o dicionário, sem duplicatas, sem imputação), mas **não está no formato
numérico final**. Tudo abaixo deve ser ajustado **só no treino de cada dobra** (dentro de um `Pipeline`), nunca no
desenvolvimento inteiro nem no teste.

> **Atalho:** `src.conjuntos.preprocessamento.montar_pipeline("C07", familia, regressor)` já faz todos os itens abaixo.
> `familia` = `"linear"`, `"svm"`, `"arvore"` ou `"boosting"`. Ex.:
> `cross_validate(montar_pipeline("C07", "boosting", HistGradientBoostingRegressor(random_state=42)), X, y, cv=dobras)`.

### 5.1 Codificação

| Colunas | Modelos lineares / SVM / KNN | Árvores / boosting |
|---|---|---|
| Ordinais: `Q006`, `Q001`, `Q002`, `Q003`, `Q004`, `Q024`, `Q025`, `ESCOLARIDADE_MAX_RESP`, `OCUPACAO_MAX_RESP` | posição na ordem (A = 0, B = 1, …) | posição na ordem |
| Especiais "não sei" (`H` em Q001/Q002, `F` em Q003/Q004, `INDETERMINADA`) | **não têm posição na ordem**: viram vazio → mediana do treino + coluna indicadora | código depois do último nível (a árvore separa sozinha) |
| Categóricas: `TP_ESCOLA`, `TP_ST_CONCLUSAO`, `TP_FAIXA_ETARIA`, `SG_UF_PROVA`, `PERFIL_ACESSO_DIGITAL`, `TIPO_ESCOLA_DETALHADO` | one-hot, sem a 1ª categoria (referência) | códigos inteiros; no `HistGradientBoostingRegressor`, marcar como categóricas |
| Binárias: `IN_TREINEIRO`, `TP_LINGUA`, `ATRASO_ESCOLAR` | como estão (0/1) | como estão |

### 5.2 Valores ausentes

- **Lineares / SVM / KNN:** `SimpleImputer(strategy="median", add_indicator=True)` nas 3 colunas com vazio (e nos especiais da 5.1).
- **Árvores / boosting:** podem manter `NaN` (`DecisionTreeRegressor`, `RandomForestRegressor` e `HistGradientBoostingRegressor`
  aceitam nativamente no scikit-learn ≥ 1.4). `BaggingRegressor`, `KNeighborsRegressor` e `SVR` não aceitam: imputar.

### 5.3 Escala e assimetria

- `RENDA_PER_CAPITA_APROX` é muito assimétrica (assimetria 3,8): aplicar `log1p` antes de padronizar nos modelos lineares/SVM/KNN.
- Padronizar (`StandardScaler`) as numéricas e ordinais para linear, SVM e KNN. Árvores não precisam.

### 5.4 Redundâncias (só para regressão linear sem regularização)

Estas colunas são **função exata de outras do C07** e deixam a matriz sem posto completo; retire-as de X nos modelos lineares
(as árvores podem mantê-las):

| Retirar | Porque é determinada por |
|---|---|
| `IN_TREINEIRO` | `TP_ST_CONCLUSAO == 3` |
| `Q025` | `PERFIL_ACESSO_DIGITAL` |
| `TP_ESCOLA` | `TIPO_ESCOLA_DETALHADO` (que o refina) |
| `CAPITAL_CULTURAL_RESP` | soma dos níveis de `ESCOLARIDADE_MAX_RESP` e `OCUPACAO_MAX_RESP` |
| `N_RESP_NAO_SEI` | soma dos indicadores de "não sei" de `Q001`–`Q004` |

### 5.5 Particularidades dos dados que merecem registro no artigo

- **Escola só é informada por concluintes de 2023.** `TIPO_ESCOLA_DETALHADO = NAO_INFORMADA` coincide com `TP_ST_CONCLUSAO ≠ 2`
  em todas as linhas exceto **3**. Nos modelos lineares isso gera colinearidade quase exata (o coeficiente da combinação
  depende de 3 participantes): use regularização (Ridge) ou interprete os coeficientes de tipo de escola com cautela.
  As previsões não são afetadas.
- **Declaração × cadastro da escola.** 16.881 participantes declararam escola pública (`TP_ESCOLA = 2`), mas a escola
  é privada pelo cadastro (`TP_DEPENDENCIA_ADM_ESC = 4`); 124 fazem o caminho inverso. `TIPO_ESCOLA_DETALHADO` segue o
  cadastro, que é a informação da escola, não da pessoa.
- **`ANOS_DESDE_CONCLUSAO` "17"** significa "antes de 2007" (17 anos ou mais): é um limite inferior.
- **Notas zero (2.169)** são notas obtidas, não ausência. Se testar treino sem zeros, retire-os **só do treino de cada dobra**
  e avalie com eles (regra do enunciado).
- **Faixas raras do alvo:** só 1% das notas fica abaixo de 323 e 1% acima de 674. Os erros tendem a se concentrar nos
  extremos; é onde o balanceamento (ponderação/reamostragem só no treino) deve ser investigado.
- **`SG_UF_PROVA`** é onde a prova foi feita, não necessariamente a residência. **`TP_LINGUA`** é a opção, não a proficiência.

### 5.6 O que NÃO fazer

- Não ajustar codificação, imputação, escala, PCA ou seleção no desenvolvimento inteiro antes da validação cruzada.
- Não usar linhas com `CONJUNTO == "teste"` para escolher features, modelos, hiperparâmetros ou tamanho do treino.
- Não incluir `NU_INSCRICAO`, `CONJUNTO` ou `DOBRA` em X.
- Não converter `NaN` do alvo em zero (não há: todo participante aqui tem nota).

## 6. Origem

- `dados/processed/enem_2023_lc_elegiveis.parquet` (tratamento em `src/dados/`, ver `README_BASE_TRATADA.md`) +
  colunas extras do `MICRODADOS_ENEM_2023.csv` validadas em `src/conjuntos/base_ampliada.py`
  (auditoria em `relatorios/resumo_base_ampliada.json`: 0 códigos inválidos).
- Features derivadas: `src/conjuntos/engenharia.py`. Definição do conjunto: `CONJUNTOS["C07"]` em `src/conjuntos/configuracao.py`.
- Justificativa das features e da escolha: `docs/conjuntos_features_lc.md`.
