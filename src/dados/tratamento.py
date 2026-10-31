"""Padronização de formatos e validação de domínios (etapas 4 e 5).

Nenhuma imputação: valores fora do domínio viram ausentes e são contabilizados.
Cada função devolve (série_tratada, máscara_de_inválidos).
"""

import numpy as np
import pandas as pd

from . import configuracao as cfg


def normalizar_texto(bruto):
    """Remove espaços periféricos, coloca em maiúsculas e trata texto vazio como ausente."""
    serie = bruto.astype("string").str.strip().str.upper()
    return serie.replace("", pd.NA)


def _para_numero(serie_texto):
    numero = pd.to_numeric(serie_texto, errors="coerce").astype("Float64")
    finito = np.isfinite(numero.to_numpy(dtype="float64", na_value=np.nan))
    return numero.where(finito)


def tratar_letra(bruto, dominio):
    serie = normalizar_texto(bruto)
    valido = serie.isin(dominio)
    invalido = serie.notna() & ~valido
    return serie.where(valido), invalido


def tratar_codigo(bruto, dominio, tipo="Int8"):
    serie = normalizar_texto(bruto)
    numero = _para_numero(serie)
    inteiro = (numero % 1 == 0).fillna(False)
    valido = inteiro & numero.isin(dominio).fillna(False)
    invalido = serie.notna() & ~valido
    return numero.where(valido).astype(tipo), invalido


def tratar_q005(bruto):
    return tratar_codigo(bruto, list(range(cfg.Q005_MIN, cfg.Q005_MAX + 1)))


def tratar_uf(bruto):
    return tratar_letra(bruto, cfg.UFS_VALIDAS)


def tratar_nota(bruto):
    """Converte a nota para Float64. Texto não numérico vira ausente (nunca zero)."""
    serie = normalizar_texto(bruto).str.replace(",", ".", regex=False)
    numero = _para_numero(serie)
    invalido = serie.notna() & numero.isna()
    return numero, invalido


def tratar_id(bruto):
    serie = bruto.astype("string").str.strip().replace("", pd.NA)
    return serie, pd.Series(False, index=bruto.index)


def tratar_bloco(bruto):
    """Aplica a padronização às 14 colunas de entrada de um bloco.

    Retorna o DataFrame tratado e um dicionário {coluna: máscara de valores inválidos}.
    """
    tratado, invalidos = {}, {}
    for coluna in cfg.COLUNAS_ENTRADA:
        serie = bruto[coluna]
        if coluna == cfg.COLUNA_ID:
            resultado = tratar_id(serie)
        elif coluna == cfg.COLUNA_ALVO:
            resultado = tratar_nota(serie)
        elif coluna == "Q005":
            resultado = tratar_q005(serie)
        elif coluna == "SG_UF_PROVA":
            resultado = tratar_uf(serie)
        elif coluna in cfg.DOMINIOS_LETRA:
            resultado = tratar_letra(serie, cfg.DOMINIOS_LETRA[coluna])
        else:
            resultado = tratar_codigo(serie, cfg.DOMINIOS_CODIGO[coluna])
        tratado[coluna] = resultado[0].astype(cfg.TIPOS_TRATADOS[coluna])
        invalidos[coluna] = resultado[1]
    return pd.DataFrame(tratado, index=bruto.index), invalidos
