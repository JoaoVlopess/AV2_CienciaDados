# Conjuntos de features candidatos — ENEM 2023, alvo `NU_NOTA_LC`

Planejamento dos conjuntos de features que o pipeline de treino vai comparar.
**Nada aqui foi treinado ainda.** Os números da Seção 2 são exploratórios (média da nota por categoria),
servem para priorizar, e não substituem a validação cruzada.

---

## 1. O que o enunciado pede e onde cada conjunto entra

| Exigência do enunciado | Onde está coberta neste plano |
|---|---|
| Criar ≥ 3 features com hipóteses claras | Já existem 3 (`RENDA_PER_CAPITA_APROX`, `ESCOLARIDADE_MAX_RESP`, `PERFIL_ACESSO_DIGITAL`); a Seção 4 propõe mais 15 (D04–D18) |
| Comparar features originais × novas × resultado da seleção | Conjuntos **C01** (originais), **C02/C04…** (com novas), **C12** (seleção) |
| Testar um método de seleção | **C12** |
| PCA em ≥ 2 regressores, justificar nº de componentes e bloco | **C13** (PCA global) e **C14** (PCA só do bloco de bens) |
| Registrar colunas usadas, criadas e removidas, com justificativa | Seções 3 e 4 |
| Aprofundamento: ≥ 2 hipóteses além do mínimo | Ablação por blocos (Seção 6), localização fina (C08), interações (C10), clusters (C15), dados externos (C16), solução enxuta (C03) |
| Sem vazamento, transformações ajustadas só no treino da dobra | Seção 7 (o que pode ser pré-calculado e o que precisa ficar dentro do `Pipeline`) |

---

## 2. Evidência exploratória (só no desenvolvimento)

**Como foi medido:** 2.258.114 participantes de desenvolvimento (a mesma divisão do notebook,
`train_test_split(test_size=0.20, random_state=42)` sobre a ordem de `enem_2023_lc_elegiveis.parquet`).
O teste não foi lido. Para cada coluna, calculamos o **η²**: a fração da variância de `NU_NOTA_LC` explicada
só pela média de cada categoria. É o R² de um "modelo" que usa apenas aquela coluna. Variáveis contínuas
foram agrupadas em 20 quantis.

**Como ler:** η² é univariado e **não soma**. Bens e renda são correlacionados (Spearman 0,5 a 0,64 entre
`Q006` e `Q008`, `Q010`, `Q014`, `Q024`), então o ganho conjunto será bem menor que a soma.

### 2.1 Colunas brutas (ordenadas por η²)

| Coluna | η² | Ausente | Categorias | Hoje no conjunto? |
|---|---:|---:|---:|---|
| `RENDA_PER_CAPITA_APROX` | 0,154 | 1,6% | contínua | sim |
| `Q006` renda familiar | 0,146 | 0% | 17 | sim |
| `Q024` computadores | 0,125 | 0% | 5 | sim |
| `PERFIL_ACESSO_DIGITAL` | 0,110 | 0% | 4 | sim |
| **`Q003` ocupação do pai** | **0,104** | 0% | 6 | **não** |
| `ESCOLARIDADE_MAX_RESP` | 0,097 | 0% | 8 | sim |
| **`Q004` ocupação da mãe** | **0,094** | 0% | 6 | **não** |
| **`CO_MUNICIPIO_PROVA`** | **0,093** | 0% | 1.750 | **não** |
| `Q001` / `Q002` escolaridade | 0,093 / 0,088 | 0% | 8 | sim |
| **`Q008` banheiros** | **0,072** | 0% | 5 | **não** |
| **`Q010` carros** | **0,067** | 0% | 5 | **não** |
| `TP_LINGUA` | 0,067 | 0% | 2 | sim |
| **`Q018` aspirador** | **0,064** | 0% | 2 | **não** |
| `Q019`, `Q016`, `Q013`, `Q014` (TV, micro-ondas, freezer, lava-roupa) | 0,053 a 0,058 | 0% | 5 | não |
| `SG_UF_PROVA` | 0,053 | 0% | 27 | sim |
| **`TP_COR_RACA`** | **0,052** | 0% | 6 | **não** |
| `Q022` celulares | 0,046 | 0% | 5 | não |
| `TP_ESCOLA` | 0,043 | 0% | 3 | sim |
| `TP_DEPENDENCIA_ADM_ESC` | 0,039 | **73,3%** | 4 | não |
| `CO_MUNICIPIO_ESC` | 0,038 | **73,3%** | 5.473 | não |
| `Q025` internet | 0,027 | 0% | 2 | sim |
| `Q009` quartos, `Q021` TV assinatura, `Q023` telefone fixo, `Q007` empregada | 0,015 a 0,026 | 0% | 2 a 5 | não |
| `Q005` moradores | 0,011 | 0% | 20 | sim |
| `Q012`, `Q020`, `Q017`, `Q015` | 0,010 a 0,011 | 0% | 2 a 5 | não |
| `TP_ANO_CONCLUIU` | 0,010 | 0% (62,6% = "não informado") | 18 | não |
| `Q011` motocicleta | 0,005 | 0% | 5 | não |
| `TP_ST_CONCLUSAO` / `TP_FAIXA_ETARIA` | 0,005 / 0,005 | 0% | 4 / 20 | sim |
| `TP_NACIONALIDADE`, `TP_ESTADO_CIVIL`, `TP_ENSINO` | 0,002 a 0,004 | `TP_ENSINO`: 61,9% | — | não |
| `TP_LOCALIZACAO_ESC`, `TP_SIT_FUNC_ESC` | 0,002 / 0,000 | 73,3% | — | não |
| `TP_SEXO` | 0,000 (F 517,1 × M 519,7) | 0% | 2 | não |
| `IN_TREINEIRO` | **0,000** (518,2 × 517,9) | 0% | 2 | sim |

### 2.2 Features derivadas candidatas

| Feature (proposta na Seção 4) | η² | Comparar com |
|---|---:|---|
| `Q006 × TP_LINGUA` (cruzamento) | **0,169** | `Q006` sozinha: 0,146 |
| `Q006 × ESCOLARIDADE_MAX_RESP` | 0,162 | 0,146 / 0,097 |
| `ESCOLARIDADE_MAX × OCUPACAO_MAX` | 0,128 | 0,097 / 0,106 |
| `INDICE_BENS` | 0,116 | melhor bem isolado (`Q008`): 0,072 |
| `INDICE_BENS_POR_MORADOR` | 0,116 | igual ao índice bruto: dividir por morador não ajudou |
| `OCUPACAO_MAX_RESP` | 0,106 | `Q003`: 0,104 |
| `TP_ESCOLA × SG_UF_PROVA` | 0,094 | 0,043 / 0,053 separadas |
| `TIPO_ESCOLA_DETALHADO` | 0,052 | `TP_ESCOLA`: 0,043 |
| `REGIAO` | 0,046 | `SG_UF_PROVA`: 0,053 com 27 categorias × 5 |
| `MORADORES_POR_QUARTO` | 0,042 | `Q005`: 0,011 |
| `DEFASAGEM_IDADE_SERIE` (4 categorias; implementada como `ATRASO_ESCOLAR`) | 0,014 | `TP_FAIXA_ETARIA` e `TP_ST_CONCLUSAO`: 0,005 cada |
| `N_RESP_NAO_SEI` | 0,010 | — |
| `MUN_ESCOLA_IGUAL_PROVA` | 0,002 | **descartada** |

### 2.3 O que esses números sugerem

1. **A ocupação dos responsáveis (`Q003`, `Q004`) é o maior sinal que ficou de fora.** Está no nível da escolaridade.
2. **O município da prova parece valer quase o dobro da UF** (0,093 × 0,053). Com 1.750 categorias, exige codificação
   supervisionada fora da dobra ou agregação (Seção 4).
3. **Bens domésticos** (`Q008`, `Q010`, `Q018`) e um **índice de bens** competem com a renda. A hipótese é que corrigem
   o ruído da renda declarada em faixas.
4. **`TP_LINGUA` interage com renda**: inglês tem média 534,9 e espanhol 495,5. O cruzamento é o maior η² medido.
5. **`IN_TREINEIRO` não separa nada sozinho** (518,2 × 517,9), mas a combinação idade + situação (**atraso escolar**:
   478,6, n = 94.687) separa.
6. **Contexto da escola só existe para concluintes de 2023.** `CO_MUNICIPIO_ESC` e `TP_DEPENDENCIA_ADM_ESC` vêm
   preenchidos para 68,5% de quem tem `TP_ST_CONCLUSAO = 2` e para **0%** dos demais. A ausência é estrutural:
   ela codifica a situação escolar e não é aleatória. Dentro da rede pública, **federal (550,6) × estadual (498,6)**
   é uma diferença grande, escondida em `TP_ESCOLA`.
7. **Sexo, estado civil, nacionalidade, `TP_ENSINO`, localização e situação da escola** têm sinal desprezível.
   Vão só no conjunto completo, e a seleção deve removê-los. Isso também é um resultado a reportar.

---

## 3. Inventário das 76 colunas

### 3.1 Proibidas em `X` (vazamento ou sem informação)

| Colunas | Motivo |
|---|---|
| `NU_NOTA_CN`, `NU_NOTA_CH`, `NU_NOTA_MT`, `NU_NOTA_REDACAO`, `NU_NOTA_COMP1`–`5` | Notas: proibidas pelo enunciado |
| `TX_RESPOSTAS_*`, `TX_GABARITO_*` | Produzidas pela prova/correção |
| `TP_PRESENCA_*`, `TP_STATUS_REDACAO` | Proibidas como features (só diagnóstico) |
| `NU_INSCRICAO` | Só controle de linhas e partições |
| `NU_ANO` | Constante (2023) |
| `CO_PROVA_*` | Cor do caderno não tem significado de perfil. As versões adaptadas (Braile, ledor, Libras: médias 412 a 489, < 0,2% dos participantes) indicam deficiência, mas são definidas na aplicação da prova. η² = 0,001. **Descartar e justificar.** |
| `NO_MUNICIPIO_*`, `CO_UF_*` | Redundantes com `CO_MUNICIPIO_*` / `SG_UF_*` |

### 3.2 Permitidas, por bloco

| Bloco | Colunas brutas | Derivadas |
|---|---|---|
| **RENDA** | `Q006`, `Q005` | `RENDA_PER_CAPITA_APROX` |
| **FAMÍLIA** (escolaridade + ocupação) | `Q001`, `Q002`, `Q003`, `Q004` | `ESCOLARIDADE_MAX_RESP`, `OCUPACAO_MAX_RESP`, `CAPITAL_CULTURAL_RESP`, `N_RESP_NAO_SEI` |
| **BENS** | `Q007`–`Q023` | `INDICE_BENS`, `INDICE_BENS_POR_MORADOR`, `MORADORES_POR_QUARTO`, `PCA_BENS_*` |
| **DIGITAL** | `Q024`, `Q025`, `Q022` | `PERFIL_ACESSO_DIGITAL` |
| **ESCOLAR** | `TP_ESCOLA`, `TP_ST_CONCLUSAO`, `TP_ANO_CONCLUIU`, `IN_TREINEIRO`, `TP_ENSINO`, `TP_DEPENDENCIA_ADM_ESC`, `TP_LOCALIZACAO_ESC`, `TP_SIT_FUNC_ESC` | `TIPO_ESCOLA_DETALHADO`, `ATRASO_ESCOLAR`, `ANOS_DESDE_CONCLUSAO` |
| **DEMOGRAFIA** | `TP_FAIXA_ETARIA`, `TP_SEXO`, `TP_COR_RACA`, `TP_ESTADO_CIVIL`, `TP_NACIONALIDADE` | — |
| **LOCALIZAÇÃO** | `SG_UF_PROVA`, `CO_MUNICIPIO_PROVA`, `CO_MUNICIPIO_ESC`, `SG_UF_ESC` | `REGIAO`, `MUN_PROVA_TE`, `MUN_PROVA_PORTE` |
| **LÍNGUA** | `TP_LINGUA` | interações com renda |

> **Redundâncias conhecidas (já tratadas no notebook linear):** `IN_TREINEIRO` ≡ `TP_ST_CONCLUSAO == 3`;
> `Q025` está contida em `PERFIL_ACESSO_DIGITAL`. Para modelos lineares, sempre retirar uma das duas.
> As novas derivadas criam redundâncias do mesmo tipo (ex.: `OCUPACAO_MAX_RESP` × `Q003`/`Q004`), então conferir o posto da matriz.

---

## 4. Catálogo de features derivadas

As linhas **D01–D03** já existem em `src/dados/engenharia_features.py`. As demais são propostas.
**Tipo de cálculo** define onde a feature pode ser criada (Seção 7):
*linha* = depende só da própria linha e pode ser pré-calculada no parquet;
*aprendida* = usa estatísticas da base e precisa ficar dentro do `Pipeline`, ajustada no treino de cada dobra.

| ID | Feature | Como calcular | Hipótese | Cálculo |
|---|---|---|---|---|
| D01 | `RENDA_PER_CAPITA_APROX` | ponto médio de `Q006` ÷ `Q005`; `Q` = vazio | renda dividida por mais pessoas vale menos | linha |
| D02 | `ESCOLARIDADE_MAX_RESP` | máx(`Q001`, `Q002`) em A–G; `H` → `INDETERMINADA` | um responsável com ensino superior já muda o ambiente | linha |
| D03 | `PERFIL_ACESSO_DIGITAL` | `Q024` × `Q025` em 4 perfis | "só internet" ≈ acesso só pelo celular | linha |
| D04 | `OCUPACAO_MAX_RESP` | máx(`Q003`, `Q004`) nos grupos 1–5; `F` ("não sei") em algum → `INDETERMINADA` (mesma regra de D02) | ocupação resume renda, estabilidade e capital cultural | linha |
| D05 | `CAPITAL_CULTURAL_RESP` | nível de D02 (0–6) + nível de D04 (0–4), com indicador se algum for indeterminado; para árvores, o cruzamento D02 × D04 | escolaridade e ocupação juntas descrevem melhor o ambiente familiar | linha |
| D06 | `N_RESP_NAO_SEI` | nº de "não sei" em `Q001`–`Q004` (0–4) | desconhecer a escolaridade/ocupação dos pais sinaliza ausência de um responsável (média cai de 521,6 para 474,8) | linha |
| D07 | `INDICE_BENS` | soma ordinal (A = 0, B = 1, …) de `Q007`, `Q008`, `Q009`, `Q010`, `Q012`–`Q017`, `Q019`, `Q022`, `Q024` + binários `Q018`, `Q020`, `Q021`, `Q023`, `Q025`. `Q011` (moto) fica fora: Spearman ≈ 0 com renda | patrimônio acumulado é menos ruidoso que a renda declarada em faixa | linha |
| D08 | `INDICE_BENS_POR_MORADOR` | D07 ÷ `Q005` | mesma lógica de D01. Na EDA **não** superou D07: testar e registrar | linha |
| D09 | `MORADORES_POR_QUARTO` | `Q005` ÷ nº de quartos (`Q009`: B = 1 … E = 4+). `Q009 = A` (0,74%) → vazio + indicador; "4 ou mais" é limite inferior (explicitar) | adensamento domiciliar ≈ falta de espaço para estudar | linha |
| D10 | `TIPO_ESCOLA_DETALHADO` | `TP_DEPENDENCIA_ADM_ESC` quando houver (federal/estadual/municipal/privada); senão `TP_ESCOLA` (pública sem detalhe/privada); senão `NAO_INFORMADA` | separar a rede federal (550,6) da estadual (498,6) | linha |
| D11 | `ATRASO_ESCOLAR` | 1 se `TP_ST_CONCLUSAO = 2` e faixa ≥ 19 anos; senão 0. (Na EDA medimos 4 categorias; as outras 3 repetem `TP_ST_CONCLUSAO`, então só o indicador de atraso entra.) | atraso escolar é um marcador forte (478,6 × 516,7) que idade ou situação isoladas não captam | linha |
| D12 | `ANOS_DESDE_CONCLUSAO` | a partir de `TP_ANO_CONCLUIU` (1 = 2022 … 17 = antes de 2007); 0 → vazio + indicador | tempo longe da escola × experiência | linha |
| D13 | `REGIAO` | região de `SG_UF_PROVA` (N, NE, SE, S, CO) | simplificar a UF (5 × 27 categorias) | linha |
| D14 | `MUN_PROVA_TE` | *target encoding* de `CO_MUNICIPIO_PROVA` com suavização para a média, **fora da dobra** (ex.: `sklearn.preprocessing.TargetEncoder`, que faz *cross-fitting* interno) | o município capta desigualdades locais que a UF esconde | **aprendida** |
| D15 | `MUN_PROVA_PORTE` | nº de inscritos do município no treino da dobra (log) | proxy não supervisionado de porte/urbanização | **aprendida** |
| D16 | `INTER_RENDA_X_ESPANHOL`, `INTER_RENDA_X_ESCOLARIDADE`, `INTER_DIGITAL_X_ESCOLARIDADE` | produtos de níveis: nível de `Q006` (0–16) × `TP_LINGUA`; nível de `Q006` × nível de D02 (indeterminada = 0); (`PERFIL = COMPUTADOR_E_INTERNET`) × nível de D02. Produtos, e não one-hot do cruzamento, para não repetir os efeitos principais. `TP_ESCOLA × REGIAO` ficou de fora pelo mesmo motivo | efeitos que se combinam; úteis para modelos lineares, árvores já aprendem | linha |
| D17 | `PCA_BENS_1..k` | PCA só do bloco BENS padronizado | índice de bens aprendido, alternativa a D07 | **aprendida** |
| D18 | `DIST_CLUSTER_1..k` | distâncias aos centróides de `MiniBatchKMeans` ajustado no treino da dobra | perfis socioeconômicos típicos | **aprendida** |
| — | `MUN_ESCOLA_IGUAL_PROVA` | escola e prova no mesmo município | mobilidade | **descartada** (η² = 0,002) |

**Dados externos (opcional, C16).** O enunciado exige indicadores de **2023** na mesma área geográfica, com fonte,
chave de junção, cobertura e perdas registradas. Candidatos a conferir antes de usar:
IDEB 2023 do ensino médio por município (Inep; usa o SAEB, não o ENEM, então não vaza o alvo);
indicadores do Censo Escolar 2023 por município. Chave: `CO_MUNICIPIO_PROVA`. Lembrar que município de prova
**não** é residência.

---

## 5. Conjuntos de features

Convenções: `ORIG12` = as 12 originais; `DER3` = D01–D03; `+` = acrescenta ao conjunto anterior indicado.
Para modelos lineares, aplicar sempre a regra de redundância da Seção 3.2.

| ID | Nome | Composição | Nº aprox. de colunas brutas | Hipótese / papel |
|---|---|---|---:|---|
| **C00** | BASELINE | nenhuma (`DummyRegressor`) | 0 | referência obrigatória |
| **C01** | ORIGINAIS | `ORIG12` | 12 | representação inicial (exigida na comparação) |
| **C02** | ATUAL | `ORIG12` + `DER3` | 15 | conjunto usado hoje |
| **C03** | ENXUTO | `Q006`, `RENDA_PER_CAPITA_APROX`, `ESCOLARIDADE_MAX_RESP`, `Q024`, `TP_ESCOLA`, `TP_LINGUA`, `SG_UF_PROVA` | 7 | "dá para manter o desempenho com uma solução mais simples?". Viabiliza KNN e SVR na base inteira |
| **C04** | FAMÍLIA AMPLIADA | C02 + `Q003`, `Q004`, D04, D05, D06 | 20 | a ocupação dos pais é o maior sinal que ficou de fora |
| **C05** | BENS BRUTOS | C04 + `Q007`–`Q023` (17 colunas) | 37 | patrimônio complementa a renda declarada |
| **C06** | BENS RESUMIDOS | C04 + D07, D08, D09, `Q008`, `Q010`, `Q018` | 26 | 17 colunas de bens comprimidas em ~6 mantêm o ganho de C05? |
| **C07** | TRAJETÓRIA ESCOLAR | C04 + D10, D11, D12 | 23 | rede federal × estadual e atraso escolar acrescentam sinal |
| **C08** | LOCALIZAÇÃO FINA | C04 + D14, D15 (mantém `SG_UF_PROVA`) · **C08b:** troca UF por D13 `REGIAO` | 22 | o município acrescenta informação além da UF? A região basta? |
| **C09** | DEMOGRÁFICO | C04 + `TP_COR_RACA`, `TP_SEXO`, `TP_ESTADO_CIVIL`, `TP_NACIONALIDADE` | 24 | quanto a demografia acrescenta depois do socioeconômico (ver cuidados na Seção 8) |
| **C10** | INTERAÇÕES | C04 + D16 | ~24 | interações ajudam modelos lineares, mas não árvores? |
| **C11** | COMPLETO | união de C04–C10 + `TP_ENSINO`, `TP_LOCALIZACAO_ESC` (54 colunas). Brutas substituídas por derivadas ficam fora: `TP_ANO_CONCLUIU` (→ D12) e `TP_DEPENDENCIA_ADM_ESC` (→ D10). Também ficam fora `CO_MUNICIPIO_ESC`/`SG_UF_ESC` (73% vazios, redundantes com a prova) e `TP_SIT_FUNC_ESC` (η² = 0) | 54 | teto de informação. Entrada da seleção (C12) e do PCA (C13) |
| **C12** | SELECIONADO | seleção dentro da dobra sobre C11: `SelectFromModel(LassoCV)` ou importância por permutação do `HistGradientBoosting`; alternativa `SelectKBest(mutual_info_regression, k ∈ {10, 15, 20, 30})` | k variável | **método de seleção exigido** |
| **C13** | PCA GLOBAL | C11 codificado + padronizado → `PCA`/`IncrementalPCA` com ≥ 80% da variância (componentes **substituem** as features) | k componentes | **PCA exigido** (rodar em ≥ 2 regressores, ex.: Ridge e KNN) |
| **C14** | PCA DO BLOCO BENS | C04 + D17 (PCA só de `Q007`–`Q023`, ≥ 80% da variância do bloco) | ~20 + k | bloco homogêneo e correlacionado: o PCA tem justificativa natural |
| **C15** | CLUSTERS | C04 + D18 (k ∈ {4, 6, 8}) | 20 + k | aprofundamento: perfis semelhantes |
| **C16** | EXTERNO | C08 + indicadores municipais 2023 | 22 + m | aprofundamento: dados externos |

**Ordem de prioridade esperada** (pela EDA, não confirmada): C04 > C02, e C05/C06 > C04, com C08 e C10 trazendo ganhos
menores e C09 ganho pequeno. C11 deve ser o melhor para *boosting*, e C03 deve perder pouco em relação a C11. Se a CV disser outra coisa, isso também é resultado a discutir.

---

## 6. Ablação por blocos (hipótese de aprofundamento)

Partindo de **C11** (ou do vencedor da Fase 1), retirar **um bloco por vez** e medir ΔRMSE nas mesmas 3 dobras:

| ID | Retira | Pergunta |
|---|---|---|
| A-RENDA | `Q006`, `Q005`, D01, D08 | quanto da renda os bens e a ocupação já cobrem? |
| A-FAMILIA | `Q001`–`Q004`, D02, D04–D06 | o peso do contexto familiar |
| A-BENS | `Q007`–`Q023`, D07, D09, D17 | os bens são redundantes com a renda? |
| A-DIGITAL | `Q024`, `Q025`, `Q022`, D03 | acesso digital × renda |
| A-ESCOLAR | `TP_ESCOLA`, `TP_ST_CONCLUSAO`, `TP_ANO_CONCLUIU`, campos da escola, D10–D12 | — |
| A-DEMO | `TP_FAIXA_ETARIA`, `TP_SEXO`, `TP_COR_RACA`, `TP_ESTADO_CIVIL`, `TP_NACIONALIDADE` | — |
| A-LOCAL | `SG_UF_PROVA`, `CO_MUNICIPIO_*`, D13–D15 | a localização acrescenta informação? |
| A-LINGUA | `TP_LINGUA`, interações com língua | — |

Rodar com um modelo rápido e não linear (`HistGradientBoostingRegressor`) e com Ridge: o contraste entre os dois mostra o que é efeito de interação.

---

## 7. Variantes de codificação (eixo independente dos conjuntos)

Testar sobre o conjunto vencedor, mantendo o mesmo regressor e as mesmas dobras:

| Variante | Opções | Observação |
|---|---|---|
| Letras ordinais (`Q006`, `Q001`–`Q004`, bens) | ordinal A = 0… × one-hot | ordinal é natural para árvores; one-hot deixa o linear capturar não linearidade |
| "Não sei" / `INDETERMINADA` | categoria própria × vazio + indicador | o enunciado pede tratar o desconhecimento separadamente |
| Faixa aberta `Q006 = Q` (1,6% sem renda per capita) | mediana do treino + indicador × limite inferior (R$ 26.400) ÷ moradores + indicador × NaN nativo (HGB) | explicitar a aproximação |
| `CO_MUNICIPIO_PROVA` | target encoding fora da dobra × frequência × descartar × categórica nativa (HGB/LightGBM) | one-hot de 1.750 colunas inviabiliza a maioria dos modelos |
| Campos da escola (73% vazios) | categoria `NAO_INFORMADA` (D10) × descartar | a ausência é estrutural (Seção 2.3, item 6) |
| Faixa etária | 20 códigos × agrupada (< 17, 17–19, 20–25, 26+) | reduz colunas para KNN/SVR |

### O que pode ser pré-calculado e o que fica no `Pipeline`

- **Pode ir para o parquet (cálculo por linha):** D01–D13 e D16. Não usam o alvo nem estatísticas de outras linhas.
- **Precisa ficar dentro do `Pipeline` (ajustado em cada dobra):** imputação, codificação, padronização, D14 (target encoding,
  com *cross-fitting* no treino e estatística do treino aplicada à validação, mais tratamento de município não visto),
  D15, D17, D18, quantis, seleção (C12) e PCA (C13).

---

## 8. Cuidados

- **Base atual não tem as colunas novas.** `src/dados/configuracao.py` lê só 14 colunas (lista positiva). Para C04 em
  diante, é preciso ampliar `COLUNAS_ENTRADA` e os domínios válidos (`Q003`, `Q004`, `Q007`–`Q023`, `TP_COR_RACA`,
  `TP_ANO_CONCLUIU`, `TP_DEPENDENCIA_ADM_ESC`, `CO_MUNICIPIO_PROVA` etc.) e gerar um parquet ampliado, mantendo
  `NU_INSCRICAO` para reutilizar `dados/splits/particoes_lc.parquet`.
- **Custo computacional.** Com ~1,5 milhão de linhas por treino de dobra, SVR com kernel RBF/polinomial (custo
  quadrático) é inviável: usar `LinearSVR` ou `Nystroem` + `LinearSVR` e justificar. O KNN fica caro em alta dimensão,
  o que é mais um motivo para C03/C13. O enunciado proíbe trocar a base por amostra para economizar processamento.
  Subamostra interna do *bagging*/*pasting* é permitida.
- **Atributos sensíveis** (`TP_COR_RACA`, `TP_SEXO`). O enunciado lista cor/raça e sexo como candidatos. Se entrarem,
  interpretar como **associação, não causalidade**, e comparar C04 × C09 para mostrar o quanto acrescentam.
  Mesmo fora de `X`, servem para a análise de erros por grupo.
- **Interpretação de `TP_LINGUA`:** é a opção do participante, não a proficiência.
- **Colinearidade estrutural nos modelos lineares.** Além das redundâncias da Seção 3.2, a base tem duas exatas:
  `TP_COR_RACA = 6` está no dicionário mas não ocorre (coluna só de zeros), e `TIPO_ESCOLA_DETALHADO = NAO_INFORMADA`
  ocorre **exatamente** quando `TP_ST_CONCLUSAO ≠ 2` (só concluintes de 2023 informam a escola). A etapa
  `RemoverColinearidade` (linear/SVM) descarta, no treino de cada dobra, a coluna que é combinação exata das anteriores.
  Conferido nas 3 dobras de todos os conjuntos: posto completo.
- **η² alto de categorias numerosas** é levemente inflado (≈ nº de categorias ÷ n; para o município, ≈ 0,001).
  A medida honesta vem da CV com target encoding fora da dobra.

---

## 9. Plano de execução sugerido para o pipeline

1. **Fase 1, triagem de conjuntos:** C01–C11 e C08b × {`Ridge`, `HistGradientBoostingRegressor`} nas 3 dobras fixas.
   Registrar tudo em `resultados/experimentos.csv` (hipótese, features, preparação, métricas por dobra, média, dp, tempo).
2. **Fase 2, seleção e redução:** C12 e C13/C14 sobre o melhor conjunto da Fase 1. Para o PCA, rodar em ≥ 2 regressores.
3. **Fase 3, todos os modelos obrigatórios** (linear com e sem regularização, SVM, árvore, KNN, bagging, pasting,
   RF, boosting, stacking) no conjunto vencedor e, se viável, também em C03 (enxuto) para comparar custo e erro.
4. **Fase 4, aprofundamento:** ablação (Seção 6) + 1 a 2 entre C08 (localização), C10 (interações), C15 (clusters),
   C16 (externos) e variantes de codificação (Seção 7).
5. Os 3 melhores regressores distintos seguem para a busca de hiperparâmetros com ≥ 10 configurações cada.

---

## 10. Implementação (`src/conjuntos/`)

| Arquivo | O que faz |
|---|---|
| `configuracao.py` | colunas extras, domínios, tipo de cada coluna, redundâncias lineares, definição de C01–C16 e das 8 ablações `A_*` |
| `engenharia.py` | features por linha (D04–D13, D16), funções puras que não leem o alvo |
| `base_ampliada.py` | lê as 28 colunas extras do CSV, valida os domínios, junta com `enem_2023_lc_elegiveis.parquet` → `dados/processed/enem_2023_lc_ampliada.parquet` (58 colunas); cria/reutiliza `dados/splits/particoes_lc.parquet` (mesma regra do notebook) |
| `gerar_conjuntos.py` | amostra do desenvolvimento (1/3 por dobra) → um parquet por conjunto + `manifesto.json` |
| `preprocessamento.py` | `carregar_conjunto`, `montar_preprocessador`, `montar_pipeline`: codificação por família e etapas aprendidas (target encoding, porte, PCA, k-means, seleção) dentro do `Pipeline` |

**Gerar** (a partir da raiz; ~1 min na primeira vez):

```powershell
python -m src.conjuntos.gerar_conjuntos                  # 300.000 linhas do desenvolvimento (100 mil por dobra)
python -m src.conjuntos.gerar_conjuntos --amostra 0      # desenvolvimento completo, para confirmar o vencedor
python -m src.conjuntos.gerar_conjuntos --conjuntos C02 C04 C08
```

**Usar** (a mesma chamada serve para qualquer conjunto e qualquer família):

```python
from sklearn.model_selection import cross_validate
from src.conjuntos.preprocessamento import carregar_conjunto, montar_pipeline

X, y, dobras = carregar_conjunto("C04")                     # dobras = 3 dobras fixas (treino, validação)
pipe = montar_pipeline("C04", "linear", LinearRegression())  # família: "linear", "svm", "arvore" ou "boosting"
res = cross_validate(pipe, X, y, cv=dobras, scoring="neg_root_mean_squared_error")
```

| Família | Codificação | Para |
|---|---|---|
| `linear`, `svm` | níveis ordinais padronizados, "não sei" → mediana + indicador, one-hot (sem a 1ª categoria), log1p em renda/densidades, sem redundâncias e sem colinearidade exata | `LinearRegression`, `Ridge`, `LinearSVR`/`SVR` |
| `arvore` | códigos ordinais (especiais no fim da ordem), NaN mantido | `DecisionTreeRegressor`, RF, bagging |
| `boosting` | igual a `arvore`, com categóricas marcadas como `category` | `HistGradientBoostingRegressor` (`categorical_features="from_dtype"`) |

C12 (seleção) e C13 (PCA) usam a codificação escalonada em qualquer família, porque `SelectKBest` e PCA não aceitam NaN.
O `k` da seleção pode entrar na busca como `features__selecao__k`.

**Triagem (Fase 1).** `python -m src.modelagem.triagem_conjuntos` roda todos os conjuntos com regressão linear,
árvore de decisão e `HistGradientBoostingRegressor` e grava `resultados/experimentos_triagem.csv`,
`resultados/triagem_conjuntos_lc.csv` e as figuras `lc_triagem_conjuntos.png` / `lc_triagem_ablacao.png`
(detalhes no README, seção 4.4).

**Escolha da equipe: C07.** 2º lugar nos três modelos (boosting 64,87 × 64,10 do C11), com 23 colunas em vez de 54.
Base final com todos os elegíveis: `python -m src.conjuntos.exportar_conjunto C07` → `dados/processed/enem_2023_lc_c07.parquet`,
documentada em `dados/processed/README_C07.md`.

**Tamanho da amostra.** 300 mil linhas (13% do desenvolvimento) bastam para ordenar conjuntos e cabem na RAM do Colab
(cada parquet tem de 3 a 7 MB). Como o enunciado não aceita amostra na comparação principal, o conjunto vencedor
deve ser rodado de novo com `--amostra 0`.
