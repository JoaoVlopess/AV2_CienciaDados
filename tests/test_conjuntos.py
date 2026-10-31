"""Testes dos conjuntos de features (src/conjuntos) com dados fictícios.

Rodar a partir da raiz do projeto:  python -m pytest -q
"""

import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.conjuntos import configuracao as cfg
from src.conjuntos.base_ampliada import criar_ou_carregar_particoes
from src.conjuntos.engenharia import (
    adicionar_features_ampliadas,
    anos_desde_conclusao,
    atraso_escolar,
    indice_bens,
    moradores_por_quarto,
    n_resp_nao_sei,
    ocupacao_max_resp,
    tipo_escola_detalhado,
)
from src.conjuntos.exportar_conjunto import montar_tabela
from src.conjuntos.gerar_conjuntos import selecionar_desenvolvimento
from src.conjuntos.preprocessamento import (
    FAMILIAS,
    RemoverColinearidade,
    colunas_para_familia,
    dobras_da_coluna,
    montar_preprocessador,
    preparar_para_modelos,
)
from src.dados import configuracao as cfg_dados
from src.dados.engenharia_features import escolaridade_max_resp, perfil_acesso_digital, renda_per_capita_aprox

DISPONIVEIS = [c for c, spec in cfg.CONJUNTOS.items() if spec.get("disponivel", True)]


def _s(valores, dtype="string"):
    return pd.Series(valores, dtype=dtype)


# --- Features derivadas por linha ----------------------------------------------------------
def test_ocupacao_max_segue_regra_da_escolaridade():
    resultado = ocupacao_max_resp(_s(["A", "E", "F", "C"]), _s(["D", "B", "A", None]))
    assert list(resultado) == ["D", "E", "INDETERMINADA", "INDETERMINADA"]


def test_n_resp_nao_sei_conta_h_e_f():
    resultado = n_resp_nao_sei(_s(["H", "A"]), _s(["H", "B"]), _s(["F", "C"]), _s(["F", "F"]))
    assert list(resultado) == [4, 1]


def test_moradores_por_quarto_sem_quarto_fica_vazio():
    resultado = moradores_por_quarto(_s([4, 3, 6], "Int8"), _s(["C", "A", "E"]))
    assert resultado[0] == 2.0 and pd.isna(resultado[1]) and resultado[2] == 1.5


def test_tipo_escola_detalhado_usa_dependencia_e_depois_tp_escola():
    resultado = tipo_escola_detalhado(_s(["1", None, None, None, "4"]), _s([2, 2, 3, 1, 3], "Int8"))
    assert list(resultado) == ["FEDERAL", "PUBLICA_SEM_DETALHE", "PRIVADA", "NAO_INFORMADA", "PRIVADA"]


def test_atraso_escolar_so_para_concluintes_de_2023_com_19_ou_mais():
    resultado = atraso_escolar(_s([2, 2, 1, 3], "Int8"), _s([3, 4, 10, 6], "Int8"))
    assert list(resultado) == [0, 1, 0, 0]


def test_anos_desde_conclusao_zero_e_nao_informado():
    resultado = anos_desde_conclusao(_s(["0", "1", "17"]))
    assert pd.isna(resultado[0]) and resultado[1] == 1 and resultado[2] == 17


def test_indice_bens_soma_contagens_e_binarios():
    linha = {c: ["A"] for c in cfg.CONTAGENS_BENS + cfg.BINARIOS_BENS}
    linha.update({"Q008": ["C"], "Q024": ["B"], "Q018": ["B"]})  # 2 + 1 + 1
    assert indice_bens(pd.DataFrame(linha, dtype="string"))[0] == 4


# --- Definição dos conjuntos ---------------------------------------------------------------
def test_toda_coluna_dos_conjuntos_tem_tipo_definido():
    for conjunto in DISPONIVEIS:
        sem_tipo = [c for c in cfg.CONJUNTOS[conjunto]["colunas"] if c not in cfg.TIPOS_COLUNAS]
        assert not sem_tipo, f"{conjunto}: {sem_tipo}"


def test_nenhum_conjunto_usa_colunas_proibidas():
    for conjunto in DISPONIVEIS:
        proibidas = [c for c in cfg.CONJUNTOS[conjunto]["colunas"]
                     if re.search(cfg_dados.PADRAO_COLUNAS_PROIBIDAS, c) or c in (cfg.COLUNA_ID, cfg.COLUNA_ALVO)]
        assert not proibidas, f"{conjunto}: {proibidas}"


def test_redundancias_saem_so_dos_modelos_escalonados():
    linear = colunas_para_familia("C02", "linear")
    assert "IN_TREINEIRO" not in linear and "Q025" not in linear
    assert colunas_para_familia("C02", "arvore") == cfg.CONJUNTOS["C02"]["colunas"]


def test_ablacao_retira_o_bloco_inteiro():
    assert not set(cfg.BLOCOS_ABLACAO["RENDA"]) & set(cfg.CONJUNTOS["A_RENDA"]["colunas"])
    assert set(cfg.CONJUNTOS["A_RENDA"]["colunas"]) < set(cfg.CONJUNTOS["C11"]["colunas"])


def test_amostra_cobre_as_tres_dobras_oficiais():
    # Partições criadas como no notebook: KFold(random_state=42) não pode coincidir com o sorteio da amostra.
    ids = pd.Series([f"{i:012d}" for i in range(9000)], dtype="string")
    particoes = criar_ou_carregar_particoes(ids, caminho=Path(tempfile.mkdtemp()) / "particoes.parquet")
    base = pd.DataFrame({cfg.COLUNA_ID: ids, cfg.COLUNA_ALVO: 500.0})
    amostra = selecionar_desenvolvimento(base, particoes, amostra=900)
    assert amostra[cfg.COLUNA_DOBRA].value_counts().to_dict() == {1: 300, 2: 300, 3: 300}
    assert set(amostra[cfg.COLUNA_ID]) <= set(particoes.loc[particoes["CONJUNTO"] == "desenvolvimento", cfg.COLUNA_ID])


def test_remover_colinearidade_tira_zeros_e_combinacoes_exatas():
    X = pd.DataFrame({"a": [1.0, 0, 1, 0, 1], "b": [0.0, 1, 0, 1, 1], "zeros": 0.0,
                      "a_mais_b": [1.0, 1, 1, 1, 2], "c": [3.0, 1, 4, 1, 5]})
    filtro = RemoverColinearidade().fit(X)
    assert filtro.removidas_ == ["zeros", "a_mais_b"]
    assert list(filtro.get_feature_names_out()) == ["a", "b", "c"]


def test_exportar_conjunto_traz_desenvolvimento_e_teste_com_dobras():
    base, y = _base_ficticia(n=900, semente=2)
    base = base.assign(**{cfg.COLUNA_ID: pd.Series([f"{i:012d}" for i in range(len(base))], dtype="string"),
                          cfg.COLUNA_ALVO: y})
    particoes = criar_ou_carregar_particoes(base[cfg.COLUNA_ID], caminho=Path(tempfile.mkdtemp()) / "p.parquet")
    tabela = montar_tabela(base, particoes, "C07")
    assert list(tabela.columns) == [cfg.COLUNA_ID, "CONJUNTO", cfg.COLUNA_DOBRA, *cfg.CONJUNTOS["C07"]["colunas"],
                                    cfg.COLUNA_ALVO]
    assert len(tabela) == len(base) and list(tabela[cfg.COLUNA_ID]) == list(base[cfg.COLUNA_ID])
    assert (tabela.loc[tabela["CONJUNTO"] == "teste", cfg.COLUNA_DOBRA] == 0).all()
    assert set(tabela.loc[tabela["CONJUNTO"] == "desenvolvimento", cfg.COLUNA_DOBRA]) == {1, 2, 3}


def test_dobras_da_coluna():
    dobras = dobras_da_coluna([1, 2, 3, 1])
    assert [list(v) for _, v in dobras] == [[0, 3], [1], [2]]
    assert list(dobras[0][0]) == [1, 2]


# --- Pipeline em dados fictícios -----------------------------------------------------------
def _base_ficticia(n=4000, semente=0):
    rng = np.random.default_rng(semente)

    def letras(dominio):
        return pd.Series(rng.choice(list(dominio), n), dtype="string")

    base = pd.DataFrame({
        "Q006": letras("ABCDEFGHIJKLMNOPQ"), "Q001": letras("ABCDEFGH"), "Q002": letras("ABCDEFGH"),
        "TP_ESCOLA": pd.Series(rng.integers(1, 4, n), dtype="Int8"),
        "TP_ST_CONCLUSAO": pd.Series(rng.integers(1, 5, n), dtype="Int8"),
        "TP_FAIXA_ETARIA": pd.Series(rng.integers(1, 21, n), dtype="Int8"),
        "Q005": pd.Series(rng.integers(1, 9, n), dtype="Int8"),
        "Q024": letras("ABCDE"), "Q025": letras("AB"),
        "SG_UF_PROVA": letras(cfg_dados.UFS_VALIDAS),
        "TP_LINGUA": pd.Series(rng.integers(0, 2, n), dtype="Int8"),
        "CO_MUNICIPIO_PROVA": letras([f"23{i:05d}" for i in range(40)]),
    })
    base["IN_TREINEIRO"] = (base["TP_ST_CONCLUSAO"] == 3).astype("Int8")
    for coluna, dominio in cfg.DOMINIOS_EXTRAS.items():
        base[coluna] = letras(dominio)
        if coluna in cfg.EXTRAS_COM_AUSENCIA_ESPERADA:
            base[coluna] = base[coluna].mask(rng.random(n) < 0.5)
    base["RENDA_PER_CAPITA_APROX"] = renda_per_capita_aprox(base["Q006"], base["Q005"])
    base["ESCOLARIDADE_MAX_RESP"] = escolaridade_max_resp(base["Q001"], base["Q002"])
    base["PERFIL_ACESSO_DIGITAL"] = perfil_acesso_digital(base["Q024"], base["Q025"])
    base = adicionar_features_ampliadas(base)
    y = 500 + 10 * base["Q006"].map(cfg.NIVEL_RENDA_Q006).to_numpy() + rng.normal(0, 50, n)
    return base, y


@pytest.fixture(scope="module")
def dados_ficticios():
    base, y = _base_ficticia()
    return base, y


@pytest.mark.parametrize("familia", list(FAMILIAS))
@pytest.mark.parametrize("conjunto", DISPONIVEIS)
def test_todo_conjunto_roda_em_toda_familia(dados_ficticios, conjunto, familia):
    base, y = dados_ficticios
    X = preparar_para_modelos(base[cfg.CONJUNTOS[conjunto]["colunas"]])
    preparacao = montar_preprocessador(conjunto, familia)
    Xt = preparacao.fit_transform(X.iloc[:3000], y[:3000])
    Xv = preparacao.transform(X.iloc[3000:])
    assert len(Xt) == 3000 and len(Xv) == 1000 and list(Xt.columns) == list(Xv.columns)
    forca_escalonado = {"pca_global", "selecao"} & set(cfg.CONJUNTOS[conjunto].get("etapas", {}))
    if FAMILIAS[familia] == "escalonado" or forca_escalonado:
        assert not Xt.isna().any().any(), "modelos escalonados não aceitam NaN"
    if familia == "boosting" and not forca_escalonado:
        assert (Xt.dtypes == "category").any()


@pytest.mark.parametrize("conjunto", [c for c in DISPONIVEIS if not cfg.CONJUNTOS[c].get("etapas")])
def test_matriz_linear_tem_posto_completo(dados_ficticios, conjunto):
    base, y = dados_ficticios
    X = preparar_para_modelos(base[cfg.CONJUNTOS[conjunto]["colunas"]])
    Xt = montar_preprocessador(conjunto, "linear").fit_transform(X, y).to_numpy(dtype=float)
    Xt = np.column_stack([np.ones(len(Xt)), Xt])
    assert np.linalg.matrix_rank(Xt.T @ Xt) == Xt.shape[1]
