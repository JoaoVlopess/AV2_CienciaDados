"""As três features derivadas (seções 7, 8 e 9).

Funções puras e determinísticas: recebem apenas colunas de perfil já padronizadas
e nunca consultam NU_NOTA_LC.
"""

import numpy as np
import pandas as pd

from . import configuracao as cfg


def renda_per_capita_aprox(q006, q005):
    """Ponto médio da faixa de renda familiar (Q006) dividido pelo nº de moradores (Q005).

    Q006 = Q (faixa aberta), Q006 ausente/inválido ou Q005 ausente/inválido/zero -> ausente.
    Sem arredondamento.
    """
    renda_familiar = q006.map(cfg.RENDA_REPRESENTATIVA_Q006).astype("Float64")
    moradores = q005.astype("Float64")
    moradores = moradores.where((moradores >= 1).fillna(False))  # nunca divide por zero
    return (renda_familiar / moradores).astype("Float64")


def escolaridade_max_resp(q001, q002):
    """Maior escolaridade entre os responsáveis, somente quando ambos estão entre A e G.

    Qualquer H ("não sei"), ausente ou inválido -> INDETERMINADA (regra conservadora).
    """
    ordem_pai = q001.map(cfg.ORDEM_ESCOLARIDADE)
    ordem_mae = q002.map(cfg.ORDEM_ESCOLARIDADE)
    ambos_conhecidos = q001.isin(cfg.ORDEM_ESCOLARIDADE) & q002.isin(cfg.ORDEM_ESCOLARIDADE)
    maior = pd.Series(np.where(ordem_pai >= ordem_mae, q001, q002), index=q001.index, dtype="string")
    return maior.where(ambos_conhecidos, cfg.ESCOLARIDADE_INDETERMINADA)


def perfil_acesso_digital(q024, q025):
    """Combinação de computador (Q024) e internet (Q025) no domicílio."""
    sem_computador = q024.isin(cfg.Q024_SEM_COMPUTADOR)
    com_computador = q024.isin(cfg.Q024_COM_COMPUTADOR)
    sem_internet = q025.isin(cfg.Q025_SEM_INTERNET)
    com_internet = q025.isin(cfg.Q025_COM_INTERNET)
    perfil = np.select(
        [
            sem_computador & sem_internet,
            sem_computador & com_internet,
            com_computador & sem_internet,
            com_computador & com_internet,
        ],
        [
            cfg.PERFIL_SEM_COMPUTADOR_SEM_INTERNET,
            cfg.PERFIL_SOMENTE_INTERNET,
            cfg.PERFIL_SOMENTE_COMPUTADOR,
            cfg.PERFIL_COMPUTADOR_E_INTERNET,
        ],
        default=cfg.PERFIL_INDETERMINADO,
    )
    return pd.Series(perfil, index=q024.index, dtype="string")


def adicionar_features(tratado):
    """Acrescenta as três features derivadas e devolve as 17 colunas na ordem do contrato."""
    saida = tratado.copy()
    saida["RENDA_PER_CAPITA_APROX"] = renda_per_capita_aprox(tratado["Q006"], tratado["Q005"])
    saida["ESCOLARIDADE_MAX_RESP"] = escolaridade_max_resp(tratado["Q001"], tratado["Q002"])
    saida["PERFIL_ACESSO_DIGITAL"] = perfil_acesso_digital(tratado["Q024"], tratado["Q025"])
    return saida[cfg.COLUNAS_SAIDA]
