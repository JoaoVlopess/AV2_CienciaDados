"""Configuração dos conjuntos de features (docs/conjuntos_features_lc.md).

Colunas extras lidas do CSV, domínios válidos, tipo de cada coluna para a codificação,
regras de redundância para modelos lineares e a definição dos conjuntos C01–C16 e das ablações.
"""

from ..dados import configuracao as cfg_dados

# --- Caminhos e parâmetros ----------------------------------------------------------
RAIZ_PROJETO = cfg_dados.RAIZ_PROJETO
ARQUIVO_ELEGIVEIS = cfg_dados.PASTA_SAIDA_PADRAO / cfg_dados.ARQUIVO_ELEGIVEIS
ARQUIVO_BASE_AMPLIADA = cfg_dados.PASTA_SAIDA_PADRAO / "enem_2023_lc_ampliada.parquet"
ARQUIVO_RESUMO_AMPLIADA = cfg_dados.PASTA_SAIDA_PADRAO / cfg_dados.PASTA_RELATORIOS / "resumo_base_ampliada.json"
ARQUIVO_PARTICOES = RAIZ_PROJETO / "dados" / "splits" / "particoes_lc.parquet"
PASTA_CONJUNTOS = RAIZ_PROJETO / "dados" / "conjuntos"

COLUNA_ID = cfg_dados.COLUNA_ID
COLUNA_ALVO = cfg_dados.COLUNA_ALVO
COLUNA_DOBRA = "DOBRA"
SEMENTE = 42
AMOSTRA_PADRAO = 300_000  # linhas do desenvolvimento usadas na triagem dos conjuntos
VERSAO_CONJUNTOS = "1.0.0"
AUSENTE = "AUSENTE"  # categoria explícita para ausência estrutural (ex.: campos da escola)

# --- Colunas extras lidas do CSV (além das 12 de src/dados) ----------------------------
BENS_BRUTOS = [f"Q{i:03d}" for i in range(7, 24)]  # Q007–Q023
COLUNAS_EXTRAS = [
    "TP_SEXO", "TP_ESTADO_CIVIL", "TP_COR_RACA", "TP_NACIONALIDADE",
    "TP_ANO_CONCLUIU", "TP_ENSINO", "TP_DEPENDENCIA_ADM_ESC", "TP_LOCALIZACAO_ESC",
    "CO_MUNICIPIO_PROVA", "Q003", "Q004", *BENS_BRUTOS,
]
# Fora de propósito: CO_MUNICIPIO_ESC / SG_UF_ESC (73% vazios, redundantes com a prova) e
# TP_SIT_FUNC_ESC (η² = 0); notas, respostas, presença, CO_PROVA_* (vazamento ou sem significado de perfil).

_LETRAS = "ABCDEFGHIJKLMNOPQ"
DOMINIOS_EXTRAS = {
    "TP_SEXO": ["F", "M"],
    "TP_ESTADO_CIVIL": [str(i) for i in range(5)],
    "TP_COR_RACA": [str(i) for i in range(7)],
    "TP_NACIONALIDADE": [str(i) for i in range(5)],
    "TP_ANO_CONCLUIU": [str(i) for i in range(18)],
    "TP_ENSINO": ["1", "2"],
    "TP_DEPENDENCIA_ADM_ESC": ["1", "2", "3", "4"],
    "TP_LOCALIZACAO_ESC": ["1", "2"],
    "Q003": list("ABCDEF"),
    "Q004": list("ABCDEF"),
    "Q007": list("ABCD"),
    **{q: list("AB") for q in ("Q018", "Q020", "Q021", "Q023")},
    **{q: list("ABCDE") for q in BENS_BRUTOS if q not in ("Q007", "Q018", "Q020", "Q021", "Q023")},
}
PADRAO_MUNICIPIO = r"^\d{7}$"
# Ausência permitida: campos da escola só existem para parte dos concluintes de 2023.
EXTRAS_COM_AUSENCIA_ESPERADA = ["TP_ENSINO", "TP_DEPENDENCIA_ADM_ESC", "TP_LOCALIZACAO_ESC"]

# --- Mapeamentos das features derivadas -----------------------------------------------
ORDEM_OCUPACAO = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}  # F = "não sei"
OCUPACAO_INDETERMINADA = "INDETERMINADA"
NIVEL_RENDA_Q006 = {letra: i for i, letra in enumerate(_LETRAS)}
NIVEL_BEM = {letra: i for i, letra in enumerate("ABCDE")}  # A = não tem, B = 1, ...
BINARIOS_BENS = ["Q018", "Q020", "Q021", "Q023", "Q025"]
CONTAGENS_BENS = ["Q007", "Q008", "Q009", "Q010", "Q012", "Q013", "Q014", "Q015", "Q016", "Q017",
                  "Q019", "Q022", "Q024"]  # Q011 (moto) fica fora: Spearman ≈ 0 com a renda
QUARTOS_Q009 = {"B": 1, "C": 2, "D": 3, "E": 4}  # E = "4 ou mais" (limite inferior); A = sem quarto
DEPENDENCIA_ESCOLA = {"1": "FEDERAL", "2": "ESTADUAL", "3": "MUNICIPAL", "4": "PRIVADA"}
TP_ESCOLA_SEM_DEPENDENCIA = {1: "NAO_INFORMADA", 2: "PUBLICA_SEM_DETALHE", 3: "PRIVADA"}
TIPOS_ESCOLA_DETALHADO = ["NAO_INFORMADA", "PUBLICA_SEM_DETALHE", "ESTADUAL", "MUNICIPAL", "FEDERAL", "PRIVADA"]
FAIXA_ETARIA_ATRASO = 4  # 19 anos ou mais concluindo o EM em 2023
REGIAO_POR_UF = {
    **dict.fromkeys(["AC", "AM", "AP", "PA", "RO", "RR", "TO"], "N"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "NE"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "SE"),
    **dict.fromkeys(["PR", "RS", "SC"], "S"),
    **dict.fromkeys(["DF", "GO", "MS", "MT"], "CO"),
}

FEATURES_DERIVADAS_AMPLIADAS = [
    "OCUPACAO_MAX_RESP", "CAPITAL_CULTURAL_RESP", "N_RESP_NAO_SEI",
    "INDICE_BENS", "INDICE_BENS_POR_MORADOR", "MORADORES_POR_QUARTO",
    "TIPO_ESCOLA_DETALHADO", "ATRASO_ESCOLAR", "ANOS_DESDE_CONCLUSAO", "REGIAO",
    "INTER_RENDA_X_ESPANHOL", "INTER_RENDA_X_ESCOLARIDADE", "INTER_DIGITAL_X_ESCOLARIDADE",
]

# --- Tipo de cada coluna (orienta a codificação dentro do Pipeline) ----------------------
# ordinal: ordem do dicionário; "especiais" (não sei / indeterminada) não têm posição na ordem.
# categorica: sem ordem (one-hot nos modelos escalonados; códigos nas árvores).
# numerica: contínua/contagem; "log" = log1p antes de padronizar.
# binaria: 0/1.  municipio: alta cardinalidade (target encoding + porte, ajustados na dobra).


def _ordinal(ordem, especiais=()):
    return {"tipo": "ordinal", "ordem": list(ordem), "especiais": list(especiais)}


def _categorica(dominio):
    return {"tipo": "categorica", "dominio": [str(v) for v in dominio]}


_NUMERICA = {"tipo": "numerica", "log": False}
_NUMERICA_LOG = {"tipo": "numerica", "log": True}
_BINARIA = {"tipo": "binaria"}

TIPOS_COLUNAS = {
    # renda e moradores
    "Q006": _ordinal(_LETRAS),
    "Q005": _NUMERICA,
    "RENDA_PER_CAPITA_APROX": _NUMERICA_LOG,
    # família
    "Q001": _ordinal("ABCDEFG", ["H"]),
    "Q002": _ordinal("ABCDEFG", ["H"]),
    "Q003": _ordinal("ABCDE", ["F"]),
    "Q004": _ordinal("ABCDE", ["F"]),
    "ESCOLARIDADE_MAX_RESP": _ordinal("ABCDEFG", [cfg_dados.ESCOLARIDADE_INDETERMINADA]),
    "OCUPACAO_MAX_RESP": _ordinal("ABCDE", [OCUPACAO_INDETERMINADA]),
    "CAPITAL_CULTURAL_RESP": _NUMERICA,
    "N_RESP_NAO_SEI": _NUMERICA,
    # bens e domicílio
    "Q007": _ordinal("ABCD"),
    **{q: _ordinal("AB") for q in BINARIOS_BENS},
    **{q: _ordinal("ABCDE") for q in BENS_BRUTOS + ["Q024"] if q not in ("Q007", *BINARIOS_BENS)},
    "INDICE_BENS": _NUMERICA,
    "INDICE_BENS_POR_MORADOR": _NUMERICA_LOG,
    "MORADORES_POR_QUARTO": _NUMERICA_LOG,
    # digital
    "PERFIL_ACESSO_DIGITAL": _categorica([cfg_dados.PERFIL_SEM_COMPUTADOR_SEM_INTERNET, cfg_dados.PERFIL_SOMENTE_INTERNET,
                                          cfg_dados.PERFIL_SOMENTE_COMPUTADOR, cfg_dados.PERFIL_COMPUTADOR_E_INTERNET]),
    # escolar
    "TP_ESCOLA": _categorica([1, 2, 3]),
    "TP_ST_CONCLUSAO": _categorica([1, 2, 3, 4]),
    "IN_TREINEIRO": _BINARIA,
    "TP_ENSINO": _categorica(["1", "2", AUSENTE]),
    "TP_LOCALIZACAO_ESC": _categorica(["1", "2", AUSENTE]),
    "TIPO_ESCOLA_DETALHADO": _categorica(TIPOS_ESCOLA_DETALHADO),
    "ATRASO_ESCOLAR": _BINARIA,
    "ANOS_DESDE_CONCLUSAO": _NUMERICA,
    # demografia
    "TP_FAIXA_ETARIA": _categorica(range(1, 21)),
    "TP_SEXO": _categorica(["F", "M"]),
    "TP_COR_RACA": _categorica(range(7)),
    "TP_ESTADO_CIVIL": _categorica(range(5)),
    "TP_NACIONALIDADE": _categorica(range(5)),
    # localização
    "SG_UF_PROVA": _categorica(cfg_dados.UFS_VALIDAS),
    "REGIAO": _categorica(["SE", "N", "NE", "S", "CO"]),
    "CO_MUNICIPIO_PROVA": {"tipo": "municipio"},
    # língua e interações
    "TP_LINGUA": _BINARIA,
    "INTER_RENDA_X_ESPANHOL": _NUMERICA,
    "INTER_RENDA_X_ESCOLARIDADE": _NUMERICA,
    "INTER_DIGITAL_X_ESCOLARIDADE": _NUMERICA,
}

# Colunas lidas pelo agrupamento (C15): perfil socioeconômico resumido.
COLUNAS_CLUSTER = ["Q006", "RENDA_PER_CAPITA_APROX", "Q005", "ESCOLARIDADE_MAX_RESP", "OCUPACAO_MAX_RESP",
                   "Q024", "Q025"]

# --- Redundâncias exatas para modelos escalonados (linear/SVM) --------------------------
# (coluna, colunas que a determinam): se todas as determinantes estiverem no conjunto, a coluna sai,
# senão a matriz perde posto (ex.: IN_TREINEIRO ≡ TP_ST_CONCLUSAO == 3). Árvores mantêm tudo.
REDUNDANCIAS_ESCALONADAS = [
    ("IN_TREINEIRO", ["TP_ST_CONCLUSAO"]),
    ("Q025", ["PERFIL_ACESSO_DIGITAL"]),
    ("CAPITAL_CULTURAL_RESP", ["ESCOLARIDADE_MAX_RESP", "OCUPACAO_MAX_RESP"]),
    ("N_RESP_NAO_SEI", ["Q001", "Q002", "Q003", "Q004"]),
    ("INDICE_BENS", [*CONTAGENS_BENS, "Q018", "Q020", "Q021", "Q023"]),
    ("TP_ESCOLA", ["TIPO_ESCOLA_DETALHADO"]),
    ("REGIAO", ["SG_UF_PROVA"]),
]

# --- Conjuntos -------------------------------------------------------------------------------
ORIG12 = list(cfg_dados.FEATURES_ORIGINAIS)
DER3 = list(cfg_dados.FEATURES_DERIVADAS)
FAMILIA_NOVAS = ["Q003", "Q004", "OCUPACAO_MAX_RESP", "CAPITAL_CULTURAL_RESP", "N_RESP_NAO_SEI"]
BENS_RESUMIDOS = ["INDICE_BENS", "INDICE_BENS_POR_MORADOR", "MORADORES_POR_QUARTO", "Q008", "Q010", "Q018"]
TRAJETORIA = ["TIPO_ESCOLA_DETALHADO", "ATRASO_ESCOLAR", "ANOS_DESDE_CONCLUSAO"]
DEMOGRAFIA_NOVAS = ["TP_COR_RACA", "TP_SEXO", "TP_ESTADO_CIVIL", "TP_NACIONALIDADE"]
INTERACOES = ["INTER_RENDA_X_ESPANHOL", "INTER_RENDA_X_ESCOLARIDADE", "INTER_DIGITAL_X_ESCOLARIDADE"]


def _unir(*listas):
    vistas = []
    for lista in listas:
        vistas.extend(c for c in lista if c not in vistas)
    return vistas


_C02 = ORIG12 + DER3
_C04 = _unir(_C02, FAMILIA_NOVAS)
_C11 = _unir(_C04, BENS_BRUTOS, BENS_RESUMIDOS, TRAJETORIA, ["TP_ENSINO", "TP_LOCALIZACAO_ESC"],
             ["CO_MUNICIPIO_PROVA", "REGIAO"], DEMOGRAFIA_NOVAS, INTERACOES)

# etapas: transformações aprendidas aplicadas DEPOIS da codificação, dentro do Pipeline.
#   pca_bens: fração de variância do PCA só do bloco de bens (C14)
#   clusters: nº de grupos do MiniBatchKMeans (C15)
#   selecao:  k do SelectKBest(f_regression) (C12)
#   pca_global: fração de variância do PCA de todas as features codificadas (C13)
# CO_MUNICIPIO_PROVA, quando presente, sempre recebe target encoding + porte (ajustados na dobra).
CONJUNTOS = {
    "C01": {"nome": "ORIGINAIS", "colunas": ORIG12,
            "hipotese": "Representação inicial: as 12 features originais escolhidas pela equipe."},
    "C02": {"nome": "ATUAL", "colunas": _C02,
            "hipotese": "As 3 features derivadas (renda per capita, escolaridade máx., perfil digital) melhoram as originais."},
    "C03": {"nome": "ENXUTO",
            "colunas": ["Q006", "RENDA_PER_CAPITA_APROX", "ESCOLARIDADE_MAX_RESP", "Q024", "TP_ESCOLA", "TP_LINGUA",
                        "SG_UF_PROVA"],
            "hipotese": "Uma solução com 7 colunas mantém quase todo o desempenho com menor custo."},
    "C04": {"nome": "FAMILIA_AMPLIADA", "colunas": _C04,
            "hipotese": "A ocupação dos responsáveis (Q003/Q004) é o maior sinal ausente do conjunto atual."},
    "C05": {"nome": "BENS_BRUTOS", "colunas": _unir(_C04, BENS_BRUTOS),
            "hipotese": "Os bens do domicílio complementam a renda declarada em faixas."},
    "C06": {"nome": "BENS_RESUMIDOS", "colunas": _unir(_C04, BENS_RESUMIDOS),
            "hipotese": "Um índice de bens + 3 bens mais informativos mantém o ganho das 17 colunas de bens."},
    "C07": {"nome": "TRAJETORIA_ESCOLAR", "colunas": _unir(_C04, TRAJETORIA),
            "hipotese": "Rede federal × estadual e atraso escolar acrescentam sinal ao tipo de escola."},
    "C08": {"nome": "LOCALIZACAO_FINA", "colunas": _unir(_C04, ["CO_MUNICIPIO_PROVA"]),
            "hipotese": "O município da prova acrescenta informação além da UF."},
    "C08B": {"nome": "REGIAO_NO_LUGAR_DA_UF", "colunas": _unir([c for c in _C04 if c != "SG_UF_PROVA"], ["REGIAO"]),
             "hipotese": "A região (5 categorias) basta no lugar da UF (27)."},
    "C09": {"nome": "DEMOGRAFICO", "colunas": _unir(_C04, DEMOGRAFIA_NOVAS),
            "hipotese": "Cor/raça, sexo, estado civil e nacionalidade acrescentam pouco após o socioeconômico."},
    "C10": {"nome": "INTERACOES", "colunas": _unir(_C04, INTERACOES),
            "hipotese": "Interações renda × língua/escolaridade ajudam modelos lineares, mas não árvores."},
    "C11": {"nome": "COMPLETO", "colunas": _C11,
            "hipotese": "Teto de informação: todas as colunas permitidas e derivadas."},
    "C12": {"nome": "SELECIONADO", "colunas": _C11, "etapas": {"selecao": 20},
            "hipotese": "Uma seleção (SelectKBest, k ajustável) sobre o completo mantém o desempenho com menos atributos."},
    "C13": {"nome": "PCA_GLOBAL", "colunas": _C11, "etapas": {"pca_global": 0.80},
            "hipotese": "Componentes com ≥ 80% da variância substituem as features sem perda relevante."},
    "C14": {"nome": "PCA_BLOCO_BENS", "colunas": _unir(_C04, BENS_BRUTOS), "etapas": {"pca_bens": 0.80},
            "hipotese": "O bloco de bens, homogêneo e correlacionado, é bem resumido por poucos componentes."},
    "C15": {"nome": "CLUSTERS", "colunas": _C04, "etapas": {"clusters": 6},
            "hipotese": "Distâncias a perfis socioeconômicos típicos (k-means na dobra) acrescentam sinal."},
    "C16": {"nome": "EXTERNO", "colunas": [], "disponivel": False,
            "hipotese": "Indicadores municipais de 2023 (ex.: IDEB) descrevem o contexto. Aguardando os dados externos."},
}

# --- Ablação por blocos (a partir de C11) -------------------------------------------------------
BLOCOS_ABLACAO = {
    "RENDA": ["Q006", "Q005", "RENDA_PER_CAPITA_APROX", "INDICE_BENS_POR_MORADOR",
              "INTER_RENDA_X_ESPANHOL", "INTER_RENDA_X_ESCOLARIDADE"],
    "FAMILIA": ["Q001", "Q002", "Q003", "Q004", "ESCOLARIDADE_MAX_RESP", "OCUPACAO_MAX_RESP",
                "CAPITAL_CULTURAL_RESP", "N_RESP_NAO_SEI", "INTER_RENDA_X_ESCOLARIDADE", "INTER_DIGITAL_X_ESCOLARIDADE"],
    "BENS": [q for q in BENS_BRUTOS if q != "Q022"] + ["INDICE_BENS", "INDICE_BENS_POR_MORADOR", "MORADORES_POR_QUARTO"],
    "DIGITAL": ["Q024", "Q025", "Q022", "PERFIL_ACESSO_DIGITAL", "INTER_DIGITAL_X_ESCOLARIDADE"],
    "ESCOLAR": ["TP_ESCOLA", "TP_ST_CONCLUSAO", "IN_TREINEIRO", "TP_ENSINO", "TP_LOCALIZACAO_ESC",
                "TIPO_ESCOLA_DETALHADO", "ATRASO_ESCOLAR", "ANOS_DESDE_CONCLUSAO"],
    "DEMOGRAFIA": ["TP_FAIXA_ETARIA", "TP_SEXO", "TP_COR_RACA", "TP_ESTADO_CIVIL", "TP_NACIONALIDADE"],
    "LOCALIZACAO": ["SG_UF_PROVA", "REGIAO", "CO_MUNICIPIO_PROVA"],
    "LINGUA": ["TP_LINGUA", "INTER_RENDA_X_ESPANHOL"],
}
for _bloco, _retirar in BLOCOS_ABLACAO.items():
    CONJUNTOS[f"A_{_bloco}"] = {
        "nome": f"C11_SEM_{_bloco}",
        "colunas": [c for c in _C11 if c not in _retirar],
        "hipotese": f"Ablação: quanto o erro de C11 piora sem o bloco {_bloco}.",
    }
