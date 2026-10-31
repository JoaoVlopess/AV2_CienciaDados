"""Configuração central do tratamento ENEM 2023 (Linguagens).

Caminhos, colunas de entrada (lista positiva), domínios válidos e tabelas de
mapeamento definidos pela especificação em docs/documentacao_tratamento_enem_2023_lc.md.
Códigos conferidos com o Dicionário_Microdados_Enem_2023.xlsx do pacote oficial.
"""

from pathlib import Path

# --- Caminhos (relativos à raiz do projeto) -------------------------------------
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
CAMINHO_CSV_PADRAO = RAIZ_PROJETO / "microdados_enem_2023" / "DADOS" / "MICRODADOS_ENEM_2023.csv"
PASTA_SAIDA_PADRAO = RAIZ_PROJETO / "dados" / "processed"
ARQUIVO_DIAGNOSTICO = "enem_2023_lc_diagnostico.parquet"
ARQUIVO_ELEGIVEIS = "enem_2023_lc_elegiveis.parquet"
PASTA_RELATORIOS = "relatorios"

# --- Fonte ------------------------------------------------------------------------
ANO_FONTE = 2023
TOTAL_LINHAS_ESPERADO = 3_933_955  # informado no enunciado; usado só como conferência
VERSAO_REGRAS_TRATAMENTO = "1.0.0"
TAMANHO_BLOCO_PADRAO = 500_000
LINHAS_AMOSTRA_TIPO_BRUTO = 100_000
LIMITE_EXEMPLOS_ANOMALIA = 20

# --- Colunas ----------------------------------------------------------------------
COLUNA_ID = "NU_INSCRICAO"
COLUNA_ALVO = "NU_NOTA_LC"

FEATURES_ORIGINAIS = [
    "Q006",
    "Q001",
    "Q002",
    "TP_ESCOLA",
    "TP_ST_CONCLUSAO",
    "TP_FAIXA_ETARIA",
    "Q005",
    "Q024",
    "Q025",
    "IN_TREINEIRO",
    "SG_UF_PROVA",
    "TP_LINGUA",
]
FEATURES_DERIVADAS = [
    "RENDA_PER_CAPITA_APROX",
    "ESCOLARIDADE_MAX_RESP",
    "PERFIL_ACESSO_DIGITAL",
]

# Lista positiva: somente estas 14 colunas são lidas do CSV.
COLUNAS_ENTRADA = [COLUNA_ID, *FEATURES_ORIGINAIS, COLUNA_ALVO]
# Contrato de saída: 17 colunas.
COLUNAS_SAIDA = [COLUNA_ID, *FEATURES_ORIGINAIS, *FEATURES_DERIVADAS, COLUNA_ALVO]

# Padrão de colunas proibidas na saída (vazamento de informação).
PADRAO_COLUNAS_PROIBIDAS = r"^(NU_NOTA_(?!LC$)|TX_RESPOSTAS_|TX_GABARITO_|TP_PRESENCA_|TP_STATUS_REDACAO|CO_PROVA_)"

# Tipo lógico de cada coluna após o tratamento (dtypes anuláveis do pandas).
TIPOS_TRATADOS = {
    COLUNA_ID: "string",
    "Q006": "string",
    "Q001": "string",
    "Q002": "string",
    "TP_ESCOLA": "Int8",
    "TP_ST_CONCLUSAO": "Int8",
    "TP_FAIXA_ETARIA": "Int8",
    "Q005": "Int8",
    "Q024": "string",
    "Q025": "string",
    "IN_TREINEIRO": "Int8",
    "SG_UF_PROVA": "string",
    "TP_LINGUA": "Int8",
    "RENDA_PER_CAPITA_APROX": "Float64",
    "ESCOLARIDADE_MAX_RESP": "string",
    "PERFIL_ACESSO_DIGITAL": "string",
    COLUNA_ALVO: "Float64",
}

# --- Domínios válidos ---------------------------------------------------------------
DOMINIOS_LETRA = {
    "Q001": list("ABCDEFGH"),
    "Q002": list("ABCDEFGH"),
    "Q006": list("ABCDEFGHIJKLMNOPQ"),
    "Q024": list("ABCDE"),
    "Q025": list("AB"),
}
DOMINIOS_CODIGO = {
    "TP_ESCOLA": [1, 2, 3],
    "TP_ST_CONCLUSAO": [1, 2, 3, 4],
    "TP_FAIXA_ETARIA": list(range(1, 21)),
    "IN_TREINEIRO": [0, 1],
    "TP_LINGUA": [0, 1],
}
Q005_MIN, Q005_MAX = 1, 20
UFS_VALIDAS = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA",
    "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
]
NOTA_MIN_PLAUSIVEL, NOTA_MAX_PLAUSIVEL = 0.0, 1000.0  # fora disso: só alerta

# Colunas com frequências por categoria nos relatórios.
COLUNAS_CATEGORICAS = [
    *FEATURES_ORIGINAIS,
    "ESCOLARIDADE_MAX_RESP",
    "PERFIL_ACESSO_DIGITAL",
]
# Colunas com estatísticas numéricas (mín., máx., quantis).
COLUNAS_NUMERICAS = ["Q005", "RENDA_PER_CAPITA_APROX", COLUNA_ALVO]

# --- Feature 1: renda per capita aproximada (seção 7) ------------------------------
# Faixas mensais familiares do ENEM 2023 (R$): (limite_inferior, limite_superior).
FAIXAS_RENDA_Q006 = {
    "A": (0.00, 0.00),
    "B": (0.00, 1320.00),
    "C": (1320.01, 1980.00),
    "D": (1980.01, 2640.00),
    "E": (2640.01, 3300.00),
    "F": (3300.01, 3960.00),
    "G": (3960.01, 5280.00),
    "H": (5280.01, 6600.00),
    "I": (6600.01, 7920.00),
    "J": (7920.01, 9240.00),
    "K": (9240.01, 10560.00),
    "L": (10560.01, 11880.00),
    "M": (11880.01, 13200.00),
    "N": (13200.01, 15840.00),
    "O": (15840.01, 19800.00),
    "P": (19800.01, 26400.00),
    "Q": (26400.01, None),  # faixa aberta: sem ponto médio
}
RENDA_REPRESENTATIVA_Q006 = {
    codigo: (float("nan") if sup is None else (inf + sup) / 2)
    for codigo, (inf, sup) in FAIXAS_RENDA_Q006.items()
}

# --- Feature 2: escolaridade máxima dos responsáveis (seção 8) ---------------------
ORDEM_ESCOLARIDADE = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}  # H = "não sei"
ESCOLARIDADE_INDETERMINADA = "INDETERMINADA"

# --- Feature 3: perfil de acesso digital (seção 9) ---------------------------------
Q024_SEM_COMPUTADOR = ["A"]
Q024_COM_COMPUTADOR = ["B", "C", "D", "E"]
Q025_SEM_INTERNET = ["A"]
Q025_COM_INTERNET = ["B"]
PERFIL_SEM_COMPUTADOR_SEM_INTERNET = "SEM_COMPUTADOR_SEM_INTERNET"
PERFIL_SOMENTE_INTERNET = "SOMENTE_INTERNET"
PERFIL_SOMENTE_COMPUTADOR = "SOMENTE_COMPUTADOR"
PERFIL_COMPUTADOR_E_INTERNET = "COMPUTADOR_E_INTERNET"
PERFIL_INDETERMINADO = "INDETERMINADO"

# --- Rótulos explicativos (somente para relatórios; os códigos originais são mantidos)
_ROTULOS_ESCOLARIDADE = {
    "A": "Nunca estudou",
    "B": "Não completou a 4ª série/5º ano do EF",
    "C": "Completou a 4ª série/5º ano, mas não a 8ª série/9º ano do EF",
    "D": "Completou a 8ª série/9º ano do EF, mas não o EM",
    "E": "Completou o EM, mas não a faculdade",
    "F": "Completou a faculdade, mas não a pós-graduação",
    "G": "Completou a pós-graduação",
    "H": "Não sei",
}
_ROTULOS_FAIXA_ETARIA = {
    1: "Menor de 17 anos", 2: "17 anos", 3: "18 anos", 4: "19 anos", 5: "20 anos",
    6: "21 anos", 7: "22 anos", 8: "23 anos", 9: "24 anos", 10: "25 anos",
    11: "Entre 26 e 30 anos", 12: "Entre 31 e 35 anos", 13: "Entre 36 e 40 anos",
    14: "Entre 41 e 45 anos", 15: "Entre 46 e 50 anos", 16: "Entre 51 e 55 anos",
    17: "Entre 56 e 60 anos", 18: "Entre 61 e 65 anos", 19: "Entre 66 e 70 anos",
    20: "Maior de 70 anos",
}


def _reais(valor):
    return "R$ " + f"{valor:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _rotulo_renda(codigo):
    inf, sup = FAIXAS_RENDA_Q006[codigo]
    if codigo == "A":
        return "Nenhuma renda"
    if sup is None:
        return f"Acima de {_reais(inf - 0.01)}"
    if codigo == "B":
        return f"Até {_reais(sup)}"
    return f"De {_reais(inf)} até {_reais(sup)}"


ROTULOS = {
    "Q001": _ROTULOS_ESCOLARIDADE,
    "Q002": _ROTULOS_ESCOLARIDADE,
    "ESCOLARIDADE_MAX_RESP": {**_ROTULOS_ESCOLARIDADE, ESCOLARIDADE_INDETERMINADA: "H, ausente ou inválido em algum responsável"},
    "Q006": {c: _rotulo_renda(c) for c in FAIXAS_RENDA_Q006},
    "PERFIL_ACESSO_DIGITAL": {
        PERFIL_SEM_COMPUTADOR_SEM_INTERNET: "Q024 = A e Q025 = A",
        PERFIL_SOMENTE_INTERNET: "Q024 = A e Q025 = B",
        PERFIL_SOMENTE_COMPUTADOR: "Q024 em B–E e Q025 = A",
        PERFIL_COMPUTADOR_E_INTERNET: "Q024 em B–E e Q025 = B",
        PERFIL_INDETERMINADO: "Q024 ou Q025 ausente/inválido",
    },
    "Q024": {"A": "Não", "B": "Sim, um", "C": "Sim, dois", "D": "Sim, três", "E": "Sim, quatro ou mais"},
    "Q025": {"A": "Não", "B": "Sim"},
    "TP_ESCOLA": {1: "Não respondeu", 2: "Pública", 3: "Privada"},
    "TP_ST_CONCLUSAO": {
        1: "Já concluí o EM",
        2: "Estou cursando e concluirei o EM em 2023",
        3: "Estou cursando e concluirei o EM após 2023",
        4: "Não concluí e não estou cursando o EM",
    },
    "TP_FAIXA_ETARIA": _ROTULOS_FAIXA_ETARIA,
    "IN_TREINEIRO": {0: "Não", 1: "Sim"},
    "TP_LINGUA": {0: "Inglês", 1: "Espanhol"},
}
