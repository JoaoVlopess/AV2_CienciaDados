"""Features derivadas por linha (D04–D13 e interações de docs/conjuntos_features_lc.md).

Funções puras e determinísticas: usam só colunas da própria linha e nunca consultam NU_NOTA_LC.
Features que dependem de estatísticas da base (target encoding, porte do município, PCA,
clusters) ficam em preprocessamento.py, para serem ajustadas no treino de cada dobra.
"""

import numpy as np
import pandas as pd

from ..dados import configuracao as cfg_dados
from . import configuracao as cfg


def _nivel(serie, mapa):
    return serie.map(mapa).astype("Float64")


def ocupacao_max_resp(q003, q004):
    """Maior grupo de ocupação entre os responsáveis, só quando ambos estão entre A e E.

    F ("não sei"), ausente ou inválido em algum dos dois -> INDETERMINADA (mesma regra de ESCOLARIDADE_MAX_RESP).
    """
    nivel_pai, nivel_mae = q003.map(cfg.ORDEM_OCUPACAO), q004.map(cfg.ORDEM_OCUPACAO)
    ambos_conhecidos = q003.isin(cfg.ORDEM_OCUPACAO) & q004.isin(cfg.ORDEM_OCUPACAO)
    maior = pd.Series(np.where(nivel_pai >= nivel_mae, q003, q004), index=q003.index, dtype="string")
    return maior.where(ambos_conhecidos, cfg.OCUPACAO_INDETERMINADA)


def capital_cultural_resp(escolaridade_max, ocupacao_max):
    """Nível da escolaridade máxima (0–6) + nível da ocupação máxima (0–4); vazio se algum for indeterminado."""
    return _nivel(escolaridade_max, cfg_dados.ORDEM_ESCOLARIDADE) + _nivel(ocupacao_max, cfg.ORDEM_OCUPACAO)


def n_resp_nao_sei(q001, q002, q003, q004):
    """Quantas das 4 perguntas sobre os responsáveis foram respondidas com "não sei" (0–4)."""
    return ((q001 == "H").astype("Int8") + (q002 == "H").astype("Int8")
            + (q003 == "F").astype("Int8") + (q004 == "F").astype("Int8")).astype("Int8")


def indice_bens(df):
    """Soma das quantidades (A = 0, B = 1, …) dos bens contáveis + 1 por bem binário presente.

    Q011 (motocicleta) fica fora por não acompanhar a renda. Qualquer item ausente -> vazio.
    """
    total = sum(_nivel(df[c], cfg.NIVEL_BEM) for c in cfg.CONTAGENS_BENS)
    total = total + sum((df[c] == "B").astype("Float64").where(df[c].notna()) for c in cfg.BINARIOS_BENS)
    return total.astype("Float64")


def indice_bens_por_morador(indice, q005):
    moradores = q005.astype("Float64")
    return (indice / moradores.where((moradores >= 1).fillna(False))).astype("Float64")


def moradores_por_quarto(q005, q009):
    """Q005 ÷ nº de quartos. Q009 = A (sem quarto) não tem denominador válido -> vazio.

    Q009 = E significa "4 ou mais": o valor usa 4, então a densidade é um limite superior.
    """
    quartos = q009.map(cfg.QUARTOS_Q009).astype("Float64")
    return (q005.astype("Float64") / quartos).astype("Float64")


def tipo_escola_detalhado(tp_dependencia, tp_escola):
    """Dependência administrativa quando informada; senão o TP_ESCOLA (pública sem detalhe/privada/não informada)."""
    detalhado = tp_dependencia.map(cfg.DEPENDENCIA_ESCOLA)
    reserva = tp_escola.map(cfg.TP_ESCOLA_SEM_DEPENDENCIA)
    return detalhado.fillna(reserva).fillna("NAO_INFORMADA").astype("string")


def atraso_escolar(tp_st_conclusao, tp_faixa_etaria):
    """1 se conclui o EM em 2023 (TP_ST_CONCLUSAO = 2) com 19 anos ou mais; 0 caso contrário."""
    atraso = (tp_st_conclusao == 2) & (tp_faixa_etaria >= cfg.FAIXA_ETARIA_ATRASO)
    return atraso.fillna(False).astype("Int8")


def anos_desde_conclusao(tp_ano_concluiu):
    """Código 1 (= 2022) … 16 (= 2007) vira 1 … 16 anos; 17 ("antes de 2007") vira 17 (limite inferior).

    0 ("não informado", inclui quem ainda não concluiu) -> vazio.
    """
    codigo = pd.to_numeric(tp_ano_concluiu, errors="coerce").astype("Float64")
    return codigo.where(codigo >= 1)


def regiao(sg_uf_prova):
    return sg_uf_prova.map(cfg.REGIAO_POR_UF).astype("string")


def interacoes(df):
    """Produtos de níveis numéricos (não estão no espaço gerado pelos efeitos principais).

    Escolaridade INDETERMINADA entra como 0: o indicador de ESCOLARIDADE_MAX_RESP já cobre esse grupo.
    """
    renda = _nivel(df["Q006"], cfg.NIVEL_RENDA_Q006)
    escolaridade = _nivel(df["ESCOLARIDADE_MAX_RESP"], cfg_dados.ORDEM_ESCOLARIDADE).fillna(0)
    completo = (df["PERFIL_ACESSO_DIGITAL"] == cfg_dados.PERFIL_COMPUTADOR_E_INTERNET).astype("Float64")
    return pd.DataFrame({
        "INTER_RENDA_X_ESPANHOL": renda * df["TP_LINGUA"].astype("Float64"),
        "INTER_RENDA_X_ESCOLARIDADE": renda * escolaridade,
        "INTER_DIGITAL_X_ESCOLARIDADE": completo * escolaridade,
    }, index=df.index)


def adicionar_features_ampliadas(base):
    """Acrescenta as 13 features de cfg.FEATURES_DERIVADAS_AMPLIADAS a uma base com as colunas brutas."""
    saida = base.copy()
    saida["OCUPACAO_MAX_RESP"] = ocupacao_max_resp(base["Q003"], base["Q004"])
    saida["CAPITAL_CULTURAL_RESP"] = capital_cultural_resp(base["ESCOLARIDADE_MAX_RESP"], saida["OCUPACAO_MAX_RESP"])
    saida["N_RESP_NAO_SEI"] = n_resp_nao_sei(base["Q001"], base["Q002"], base["Q003"], base["Q004"])
    saida["INDICE_BENS"] = indice_bens(base)
    saida["INDICE_BENS_POR_MORADOR"] = indice_bens_por_morador(saida["INDICE_BENS"], base["Q005"])
    saida["MORADORES_POR_QUARTO"] = moradores_por_quarto(base["Q005"], base["Q009"])
    saida["TIPO_ESCOLA_DETALHADO"] = tipo_escola_detalhado(base["TP_DEPENDENCIA_ADM_ESC"], base["TP_ESCOLA"])
    saida["ATRASO_ESCOLAR"] = atraso_escolar(base["TP_ST_CONCLUSAO"], base["TP_FAIXA_ETARIA"])
    saida["ANOS_DESDE_CONCLUSAO"] = anos_desde_conclusao(base["TP_ANO_CONCLUIU"])
    saida["REGIAO"] = regiao(base["SG_UF_PROVA"])
    saida[cfg.INTERACOES] = interacoes(saida)
    return saida
