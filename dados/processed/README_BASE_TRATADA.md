# Base tratada — ENEM 2023, previsão da nota de Linguagens (`NU_NOTA_LC`)

Arquivo principal: **`enem_2023_lc_elegiveis.parquet`**
**2.822.643 linhas × 17 colunas**, uma linha por participante do ENEM 2023 que tem nota de Linguagens.

```python
import pandas as pd
df = pd.read_parquet("enem_2023_lc_elegiveis.parquet")
print(df.shape)  # (2822643, 17)

X = df.drop(columns=["NU_INSCRICAO", "NU_NOTA_LC"])  # 15 features
y = df["NU_NOTA_LC"]                                  # alvo
```

---

## 1. De onde vem

- **Fonte:** microdados oficiais do Inep, `MICRODADOS_ENEM_2023.csv` (3.933.955 participantes, 76 colunas).
- **Conferência:** SHA-256 do CSV original registrado em `relatorios/resumo_processamento.json` (começa com `c8b1fae0a97b6ac8`).
- **Cobertura:** Brasil inteiro, sem amostragem e sem filtro de estado, idade, tipo de escola ou treineiro.

## 2. Quem está na base (e por quê)

| Entrou | Ficou de fora |
|---|---|
| Todos que têm `NU_NOTA_LC` registrada (**2.822.643**) | Quem não tem nota de LC (**1.111.312**, em geral faltosos) |
| Inclui as **2.169 notas zero** | — |
| Inclui os **548.249 treineiros** (19,4%) | — |

- **Por que só quem tem nota:** sem nota não existe o valor a ser previsto, e não dá para treinar nem avaliar.
- **Por que manter nota zero:** zero é uma nota obtida, não ausência de nota.
- **Por que manter treineiros:** eles também fazem a prova, e excluí-los mudaria a população estudada.
- **Ninguém foi removido por falta de informação nas features.**

## 3. As 17 colunas

### Controle

| Coluna | O que é | Por que está aqui |
|---|---|---|
| `NU_INSCRICAO` | Número de inscrição (texto) | Só para identificar a linha e conferir duplicatas. **Não use como feature.** |
| `NU_NOTA_LC` | Nota de Linguagens, Códigos e suas Tecnologias (0 a 820,8; média 518,1) | É o **alvo** (`y`) que os modelos vão prever. |

### 12 features originais (vindas do Inep, sem alteração de significado)

| Coluna | O que é | Valores | Distribuição nos elegíveis | Por que está aqui |
|---|---|---|---|---|
| `Q006` | Renda mensal da família | `A` nenhuma, `B` até R$ 1.320, … `P` até R$ 26.400, `Q` acima de R$ 26.400 | B = 29,3%, C = 15,6%, Q = 1,6% | Condição econômica |
| `Q001` | Escolaridade do pai/responsável | `A` nunca estudou … `G` pós-graduação, **`H` não sei** | E = 30,1%, H = 9,5% | Contexto educacional da família |
| `Q002` | Escolaridade da mãe/responsável | igual a `Q001` | — | Contexto educacional da família |
| `TP_ESCOLA` | Tipo de escola do ensino médio | `1` não respondeu, `2` pública, `3` privada | **1 = 61,0%**, 2 = 31,0%, 3 = 8,0% | Diferença entre rede pública e privada |
| `TP_ST_CONCLUSAO` | Situação do ensino médio | `1` já concluiu, `2` conclui em 2023, `3` conclui depois de 2023, `4` não cursa | 1 = 41,2%, 2 = 39,0%, 3 = 19,4% | Momento escolar do participante |
| `TP_FAIXA_ETARIA` | Faixa etária | `1` menos de 17 · `2`–`10` = 17 a 25 anos · `11`–`19` = faixas de 5 anos · `20` mais de 70 | — | Idade e experiência |
| `Q005` | Pessoas que moram na casa (contando o participante) | 1 a 20 | mediana 4 | Usada no cálculo da renda por pessoa |
| `Q024` | Computadores em casa | `A` nenhum, `B` 1, `C` 2, `D` 3, `E` 4 ou mais | A = 48,4%, B = 35,9% | Acesso a recursos de estudo |
| `Q025` | Internet em casa | `A` não, `B` sim | B = 91,6% | Acesso a recursos de estudo |
| `IN_TREINEIRO` | Fez a prova só para treinar | `0` não, `1` sim | 1 = 19,4% | O perfil e a motivação de quem treina são diferentes |
| `SG_UF_PROVA` | Estado onde fez a prova | 27 siglas | todas as 27 presentes | Diferenças regionais. **Não** é necessariamente o estado onde mora |
| `TP_LINGUA` | Língua estrangeira escolhida | `0` inglês, `1` espanhol | 0 = 57,5% | Parte da prova de LC depende dela. **Não** mede fluência |

### 3 features criadas pela equipe

| Coluna | Como é calculada | Valores | Por que foi criada |
|---|---|---|---|
| `RENDA_PER_CAPITA_APROX` | Ponto médio da faixa de `Q006` ÷ `Q005`. Ex.: `B` (0 a 1.320 → 660) com 2 moradores = **330** | Reais por pessoa; mediana R$ 550. **Vazio em 44.132 linhas** (todas com `Q006 = Q`) | R$ 3.000 para 2 pessoas é diferente de R$ 3.000 para 8 pessoas. A faixa `Q` não tem teto, então não existe ponto médio; foi deixada vazia em vez de inventar um valor |
| `ESCOLARIDADE_MAX_RESP` | A maior escolaridade entre `Q001` e `Q002`, **só quando os dois são conhecidos** (A–G) | `A`–`G` ou `INDETERMINADA` (10,4%) | Um único responsável com faculdade já muda o ambiente da casa. Se algum dos dois é `H` ("não sei"), não dá para afirmar qual é o máximo |
| `PERFIL_ACESSO_DIGITAL` | Combinação de `Q024` (tem computador?) e `Q025` (tem internet?) | `COMPUTADOR_E_INTERNET` 51,1% · `SOMENTE_INTERNET` 40,5% · `SEM_COMPUTADOR_SEM_INTERNET` 7,9% · `SOMENTE_COMPUTADOR` 0,5% | Junta as duas informações num perfil só. "Somente internet" geralmente significa acesso só pelo celular |

## 4. Qualidade dos dados

- **Ausentes:** nenhum nas 12 features originais. O único vazio é `RENDA_PER_CAPITA_APROX` (44.132 linhas, sempre `Q006 = Q`).
- **Códigos inválidos:** nenhum, todos os valores estão no dicionário do Inep.
- **Inscrições duplicadas:** nenhuma.
- **Valores preenchidos artificialmente:** nenhum. Não houve imputação, padronização nem codificação.

## 5. O que foi propositalmente deixado de fora

| Removido | Motivo |
|---|---|
| Notas de Matemática, Ciências da Natureza, Ciências Humanas e Redação | Vazamento: são resultados da mesma prova e não estariam disponíveis antes dela |
| Respostas, gabaritos e código da prova (`TX_*`, `CO_PROVA_*`) | Vazamento: a nota é calculada a partir deles |
| Presença e situação da redação (`TP_PRESENCA_*`, `TP_STATUS_REDACAO`) | Informações do dia da prova, não do perfil do participante |
| As outras ~50 colunas (sexo, cor/raça, município, demais perguntas do questionário) | Fora das 12 features escolhidas pela equipe |

## 6. Cuidados antes de treinar

A base está **limpa**, mas **ainda não está no formato numérico** que os modelos exigem. Faça estas etapas **no Colab, depois de separar treino e teste**, ajustando tudo só no treino (de preferência num `Pipeline` do scikit-learn):

1. **Letras → números:**
   - colunas ordinais (`Q006`, `Q001`, `Q002`, `Q024`, `ESCOLARIDADE_MAX_RESP`): A = 0, B = 1, …;
   - `H` e `INDETERMINADA` não têm posição na ordem, então trate-os como categoria separada ou use um indicador próprio.
2. **One-hot** para as colunas sem ordem: `SG_UF_PROVA`, `PERFIL_ACESSO_DIGITAL`, `TP_ESCOLA`, `TP_ST_CONCLUSAO`.
3. **Renda vazia:** preencha com a mediana do treino e, se quiser, crie uma coluna indicando que estava vazia. Outra saída é usar um modelo que aceite valor ausente (`HistGradientBoostingRegressor`).
4. **Escala:** padronize para regressão linear, SVM e KNN. Modelos de árvore não precisam.

**Atenção à coluna `TP_ESCOLA`:** 61% dos participantes estão como "não respondeu". Isso é do próprio Inep e foi mantido como categoria, mas pesa na interpretação dessa variável.

## 7. Outros arquivos

| Arquivo | Para que serve |
|---|---|
| `enem_2023_lc_diagnostico.parquet` | Mesmas 17 colunas com **todos** os 3.933.955 inscritos, inclusive sem nota. Serve só para análise descritiva |
| `relatorios/resumo_processamento.json` | Totais, hash do CSV original, regras aplicadas e conferências |
| `relatorios/qualidade_colunas.csv` | Ausentes, inválidos e valores distintos por coluna |
| `relatorios/frequencias_categorias.csv` | Contagem e percentual de cada categoria, com rótulo |
| `relatorios/exemplos_anomalias.csv` | Valores fora do dicionário (vazio nesta base) |

Gerado por `python -m src.dados.executar_tratamento`, versão das regras 1.0.0.
