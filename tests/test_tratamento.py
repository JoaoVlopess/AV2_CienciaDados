"""Testes mínimos da seção 13 da especificação.

Rodar a partir da raiz do projeto:  python -m pytest -q
"""

import json

import pandas as pd
import pytest

from src.dados import configuracao as cfg
from src.dados.engenharia_features import (
    adicionar_features,
    escolaridade_max_resp,
    perfil_acesso_digital,
    renda_per_capita_aprox,
)
from src.dados.executar_tratamento import executar
from src.dados.tratamento import tratar_bloco


def _s(valores, dtype="string"):
    return pd.Series(valores, dtype=dtype)


# --- Seção 7: renda per capita ---------------------------------------------------
def test_renda_exemplos_da_especificacao():
    resultado = renda_per_capita_aprox(_s(["A", "B", "C", "Q", "D"]), _s([4, 2, 3, 4, 0], "Int8"))
    assert resultado[0] == 0.0
    assert resultado[1] == pytest.approx(330.0)
    assert resultado[2] == pytest.approx(550.0, abs=0.01)
    assert pd.isna(resultado[3])  # faixa Q não recebe ponto médio fictício
    assert pd.isna(resultado[4])  # Q005 = 0 não divide por zero


def test_renda_ausencias():
    resultado = renda_per_capita_aprox(_s([None, "B"]), _s([3, None], "Int8"))
    assert resultado.isna().all()


# --- Seção 8: escolaridade -------------------------------------------------------
def test_escolaridade_exemplos_da_especificacao():
    resultado = escolaridade_max_resp(_s(["B", "F", "A", "H", "E"]), _s(["G", "D", "A", "F", None]))
    assert list(resultado) == ["G", "F", "A", "INDETERMINADA", "INDETERMINADA"]


def test_escolaridade_h_nunca_e_maximo():
    assert escolaridade_max_resp(_s(["H"]), _s(["H"]))[0] == "INDETERMINADA"


# --- Seção 9: acesso digital -----------------------------------------------------
def test_perfil_digital_exemplos_da_especificacao():
    resultado = perfil_acesso_digital(_s(["A", "A", "C", "E", None]), _s(["A", "B", "A", "B", "B"]))
    assert list(resultado) == [
        "SEM_COMPUTADOR_SEM_INTERNET",
        "SOMENTE_INTERNET",
        "SOMENTE_COMPUTADOR",
        "COMPUTADOR_E_INTERNET",
        "INDETERMINADO",
    ]


# --- Etapas 4 e 5: padronização e domínios ---------------------------------------
def _bloco_bruto(**sobrescrever):
    base = {
        "NU_INSCRICAO": ["1", "2", "3"],
        "Q006": [" b ", "Z", None],
        "Q001": ["h", "A", "X"],
        "Q002": ["G", "", "B"],
        "TP_ESCOLA": ["1", "2", "9"],
        "TP_ST_CONCLUSAO": ["1", "4", "5"],
        "TP_FAIXA_ETARIA": ["1", "20", "21"],
        "Q005": ["0", "2.5", "20"],
        "Q024": ["E", "a", "F"],
        "Q025": ["A", "B", "C"],
        "IN_TREINEIRO": ["0", "1", "2"],
        "SG_UF_PROVA": ["ce", "XX", "SP"],
        "TP_LINGUA": ["0", "1", "x"],
        "NU_NOTA_LC": ["0", "abc", None],
    }
    base.update(sobrescrever)
    return pd.DataFrame(base, dtype=object)


def test_padronizacao_e_codigos_invalidos():
    tratado, invalidos = tratar_bloco(_bloco_bruto())
    assert tratado["Q006"].iloc[0] == "B"
    assert pd.isna(tratado["Q006"].iloc[1]) and invalidos["Q006"].iloc[1]
    assert tratado["Q001"].iloc[0] == "H"  # categoria especial preservada
    assert pd.isna(tratado["Q002"].iloc[1]) and not invalidos["Q002"].iloc[1]  # vazio = ausente, não inválido
    assert tratado["TP_ESCOLA"].iloc[0] == 1  # "não respondeu" preservado
    assert pd.isna(tratado["TP_ESCOLA"].iloc[2]) and invalidos["TP_ESCOLA"].iloc[2]
    assert tratado["Q005"].isna().tolist() == [True, True, False]  # 0 e 2.5 inválidos
    assert tratado["Q024"].iloc[1] == "A"
    assert tratado["SG_UF_PROVA"].tolist()[0] == "CE" and pd.isna(tratado["SG_UF_PROVA"].iloc[1])
    assert tratado["NU_NOTA_LC"].iloc[0] == 0.0  # zero é observação
    assert pd.isna(tratado["NU_NOTA_LC"].iloc[1]) and invalidos["NU_NOTA_LC"].iloc[1]  # nunca vira zero
    assert str(tratado["Q005"].dtype) == "Int8"


# --- Teste recomendado: independência do alvo -----------------------------------
def test_features_nao_dependem_do_alvo():
    tratado, _ = tratar_bloco(_bloco_bruto())
    original = adicionar_features(tratado)[cfg.FEATURES_DERIVADAS]
    tratado["NU_NOTA_LC"] = pd.array([999.9, 1.0, 500.0], dtype="Float64")
    alterado = adicionar_features(tratado)[cfg.FEATURES_DERIVADAS]
    pd.testing.assert_frame_equal(original, alterado)


# --- Ponta a ponta com CSV fictício ----------------------------------------------
LINHAS_FICTICIAS = [
    # id, Q006, Q001, Q002, ESC, CONC, FAIXA, Q005, Q024, Q025, TREIN, UF, LINGUA, NOTA_LC, NOTA_MT
    ["100000000001", "B", "B", "G", "2", "1", "3", "2", "A", "B", "0", "CE", "1", "512.3", "600.1"],
    ["100000000002", "Q", "H", "F", "3", "1", "4", "4", "E", "B", "0", "SP", "0", "0", "0"],
    ["100000000003", "A", "", "E", "1", "3", "1", "4", "", "A", "1", "RS", "0", "", ""],
    ["100000000004", "D", "C", "D", "2", "2", "2", "0", "B", "A", "1", "AM", "1", "430.0", ""],
    ["100000000005", "", "", "", "", "", "", "", "", "", "0", "DF", "0", "", ""],
    ["100000000006", "Z", "A", "A", "7", "1", "5", "3", "C", "B", "0", "BA", "0", "678.9", "700"],
    ["100000000007", "C", "F", "D", "2", "1", "11", "3", "D", "B", "1", "CE", "1", "0", "321"],
]


def _escrever_csv_ficticio(caminho, linhas=LINHAS_FICTICIAS):
    cabecalho = ["NU_INSCRICAO", "NU_ANO", "Q006", "Q001", "Q002", "TP_ESCOLA", "TP_ST_CONCLUSAO",
                 "TP_FAIXA_ETARIA", "Q005", "Q024", "Q025", "IN_TREINEIRO", "SG_UF_PROVA", "TP_LINGUA",
                 "NU_NOTA_LC", "NU_NOTA_MT", "TX_RESPOSTAS_LC", "TP_PRESENCA_LC", "NO_MUNICIPIO_PROVA"]
    with open(caminho, "w", encoding="latin-1", newline="") as f:
        f.write(";".join(cabecalho) + "\n")
        for l in linhas:
            f.write(";".join([l[0], "2023", *l[1:14], l[14], "ABCDE", "1", "São João"]) + "\n")


def test_ponta_a_ponta(tmp_path):
    csv = tmp_path / "MICRODADOS_ENEM_2023.csv"
    _escrever_csv_ficticio(csv)
    resumo = executar(csv, tmp_path / "saida", tamanho_bloco=3, total_esperado=len(LINHAS_FICTICIAS), verbose=False)

    assert resumo["status"] == "OK", resumo["regras_e_alertas"]
    diag = pd.read_parquet(tmp_path / "saida" / cfg.ARQUIVO_DIAGNOSTICO)
    eleg = pd.read_parquet(tmp_path / "saida" / cfg.ARQUIVO_ELEGIVEIS)

    assert len(diag) == 7  # nenhuma linha perdida
    assert list(diag.columns) == cfg.COLUNAS_SAIDA and len(diag.columns) == 17
    assert not any(c.startswith(("NU_NOTA_MT", "TX_", "TP_PRESENCA")) for c in diag.columns)
    assert len(eleg) == 5 and eleg["NU_NOTA_LC"].notna().all()
    assert (eleg["NU_NOTA_LC"] == 0).sum() == 2  # zeros preservados
    assert (eleg["IN_TREINEIRO"] == 1).sum() == 2  # treineiros mantidos
    assert "100000000004" in set(eleg["NU_INSCRICAO"])  # Q005 = 0 não remove o participante
    assert diag.loc[diag["NU_INSCRICAO"] == "100000000002", "RENDA_PER_CAPITA_APROX"].isna().all()
    assert resumo["total_notas_lc_zero"] == 2 and resumo["invalidos_por_coluna"]["Q006"] == 1

    relatorios = tmp_path / "saida" / "relatorios"
    for nome in ["resumo_processamento.json", "qualidade_colunas.csv",
                 "frequencias_categorias.csv", "exemplos_anomalias.csv"]:
        assert (relatorios / nome).is_file()
    assert json.loads((relatorios / "resumo_processamento.json").read_text("utf-8"))["status"] == "OK"


def test_blocos_dao_mesmo_resultado(tmp_path):
    csv = tmp_path / "MICRODADOS_ENEM_2023.csv"
    _escrever_csv_ficticio(csv)
    executar(csv, tmp_path / "um_bloco", tamanho_bloco=1000, total_esperado=None, verbose=False)
    executar(csv, tmp_path / "varios", tamanho_bloco=2, total_esperado=None, verbose=False)
    for arquivo in (cfg.ARQUIVO_DIAGNOSTICO, cfg.ARQUIVO_ELEGIVEIS):
        a = pd.read_parquet(tmp_path / "um_bloco" / arquivo).sort_values("NU_INSCRICAO", ignore_index=True)
        b = pd.read_parquet(tmp_path / "varios" / arquivo).sort_values("NU_INSCRICAO", ignore_index=True)
        pd.testing.assert_frame_equal(a, b)


def test_duplicata_interrompe_finalizacao(tmp_path):
    csv = tmp_path / "MICRODADOS_ENEM_2023.csv"
    _escrever_csv_ficticio(csv, LINHAS_FICTICIAS + [LINHAS_FICTICIAS[0]])
    resumo = executar(csv, tmp_path / "saida", tamanho_bloco=3, total_esperado=None, verbose=False)
    assert resumo["status"] == "FALHOU"
    assert resumo["total_identificadores_duplicados"] == 1
    assert not (tmp_path / "saida" / cfg.ARQUIVO_DIAGNOSTICO).exists()
    assert (tmp_path / "saida" / "relatorios" / "resumo_processamento.json").exists()
