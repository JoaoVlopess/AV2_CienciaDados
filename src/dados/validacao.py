"""Inspeção, auditoria e invariantes (etapas 3, 5, 6 e seção 12).

Como o arquivo é lido em blocos, as contagens são acumuladas aqui e os relatórios
são montados no final, sempre sobre a totalidade dos registros.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd

from . import configuracao as cfg

VAZIO = "<vazio>"
AUSENTE = "NaN"
CONJUNTOS = ("diagnostico", "elegiveis")


def _contar(serie):
    return Counter(serie.value_counts(dropna=False).to_dict())


def _rotular(serie):
    """Converte para texto com ausentes como 'NaN', para contagem por categoria."""
    return serie.astype("string").fillna(AUSENTE)


def mascarar_id(valor):
    valor = str(valor)
    if len(valor) <= 6:
        return "*" * len(valor)
    return valor[:4] + "*" * (len(valor) - 7) + valor[-3:]


def classificar_anomalia(coluna, valor_bruto):
    texto = str(valor_bruto).strip()
    if coluna == cfg.COLUNA_ALVO:
        return "nota_nao_numerica"
    if coluna in cfg.DOMINIOS_CODIGO or coluna == "Q005":
        try:
            numero = float(texto.replace(",", "."))
        except ValueError:
            return "codigo_nao_numerico"
        if numero % 1 != 0:
            return "codigo_nao_inteiro"
        if coluna == "Q005" and numero <= 0:
            return "moradores_zero_ou_negativo"
        return "codigo_fora_do_dominio"
    if coluna == "SG_UF_PROVA":
        return "uf_invalida"
    return "codigo_fora_do_dominio"


def estatisticas_de_contagens(contagem):
    """Mínimo, máximo, média e quantis a partir de {valor: frequência}."""
    if not contagem:
        return {"n": 0}
    valores = np.array(sorted(contagem), dtype="float64")
    pesos = np.array([contagem[v] for v in sorted(contagem)], dtype="float64")
    acumulado = np.cumsum(pesos)
    total = acumulado[-1]

    def quantil(q):
        return float(valores[np.searchsorted(acumulado, q * total, side="left")])

    return {
        "n": int(total),
        "minimo": float(valores[0]),
        "p25": quantil(0.25),
        "mediana": quantil(0.50),
        "p75": quantil(0.75),
        "maximo": float(valores[-1]),
        "media": float(np.average(valores, weights=pesos)),
    }


class AuditoriaQualidade:
    """Acumula contagens de todos os blocos."""

    def __init__(self):
        colunas_sem_id = [c for c in cfg.COLUNAS_ENTRADA if c != cfg.COLUNA_ID]
        self.total_linhas_origem = 0
        self.valores_brutos = {c: Counter() for c in colunas_sem_id}
        self.ausentes_brutos = Counter()
        self.invalidos = {c: Counter() for c in cfg.COLUNAS_ENTRADA}
        self.linhas = Counter()
        self.ausentes_tratados = {n: Counter() for n in CONJUNTOS}
        self.frequencias = {n: {c: Counter() for c in cfg.COLUNAS_CATEGORICAS} for n in CONJUNTOS}
        self.numericos = {n: {c: Counter() for c in cfg.COLUNAS_NUMERICAS} for n in CONJUNTOS}
        self.notas = Counter()
        self.elegiveis_com_feature_ausente = 0
        self._ids = []
        self.tipos_tratados = None

    # --- Etapa 3: inspeção antes da correção -------------------------------------
    def registrar_bruto(self, bruto):
        self.total_linhas_origem += len(bruto)
        for coluna in cfg.COLUNAS_ENTRADA:
            self.ausentes_brutos[coluna] += int(bruto[coluna].isna().sum())
            if coluna != cfg.COLUNA_ID:
                self.valores_brutos[coluna].update(_contar(bruto[coluna].fillna(VAZIO)))

    # --- Etapa 5: códigos fora do domínio ---------------------------------------
    def registrar_invalidos(self, bruto, mascaras):
        for coluna, mascara in mascaras.items():
            if mascara.any():
                self.invalidos[coluna].update(_contar(bruto.loc[mascara.to_numpy(), coluna]))

    # --- Etapa 8: conjuntos de saída -------------------------------------------
    def registrar_conjunto(self, nome, df):
        if self.tipos_tratados is None:
            self.tipos_tratados = {c: str(df[c].dtype) for c in df.columns}
        self.linhas[nome] += len(df)
        self.ausentes_tratados[nome].update({c: int(df[c].isna().sum()) for c in df.columns})
        for coluna in cfg.COLUNAS_CATEGORICAS:
            self.frequencias[nome][coluna].update(_contar(_rotular(df[coluna])))
        for coluna in cfg.COLUNAS_NUMERICAS:
            self.numericos[nome][coluna].update(_contar(df[coluna].dropna().astype("float64")))

        nota = df[cfg.COLUNA_ALVO]
        if nome == "diagnostico":
            self._ids.append(df[cfg.COLUNA_ID].astype("string"))
            self.notas["observadas"] += int(nota.notna().sum())
            self.notas["ausentes"] += int(nota.isna().sum())
            self.notas["zero"] += int((nota == 0).fillna(False).sum())
            fora = (nota < cfg.NOTA_MIN_PLAUSIVEL) | (nota > cfg.NOTA_MAX_PLAUSIVEL)
            self.notas["fora_0_1000"] += int(fora.fillna(False).sum())
        else:
            self.notas["zero_elegiveis"] += int((nota == 0).fillna(False).sum())
            colunas_features = cfg.FEATURES_ORIGINAIS + ["RENDA_PER_CAPITA_APROX"]
            self.elegiveis_com_feature_ausente += int(df[colunas_features].isna().any(axis=1).sum())

    # --- Etapa 6: duplicatas -------------------------------------------------------
    def verificar_ids(self):
        ids = pd.concat(self._ids, ignore_index=True) if self._ids else pd.Series(dtype="string")
        duplicados = ids[ids.duplicated(keep=False) & ids.notna()]
        contagem = duplicados.value_counts()
        return {
            "total_ids": int(len(ids)),
            "ids_ausentes": int(ids.isna().sum()),
            "ids_distintos": int(ids.nunique(dropna=True)),
            "total_identificadores_duplicados": int(len(contagem)),
            "linhas_com_id_duplicado": int(len(duplicados)),
            "exemplos_duplicados": {mascarar_id(k): int(v) for k, v in contagem.head(cfg.LIMITE_EXEMPLOS_ANOMALIA).items()},
        }

    # --- Relatórios ------------------------------------------------------------------
    def _distintos(self, coluna):
        if coluna == cfg.COLUNA_ID:
            return None
        if coluna in cfg.COLUNAS_NUMERICAS:
            return len(self.numericos["diagnostico"][coluna])
        return len([k for k in self.frequencias["diagnostico"][coluna] if k != AUSENTE])

    def tabela_qualidade(self, tipos_brutos, info_ids):
        total = self.linhas["diagnostico"] or 1
        linhas = []
        for coluna in cfg.COLUNAS_SAIDA:
            if coluna == cfg.COLUNA_ID:
                papel = "identificador"
            elif coluna == cfg.COLUNA_ALVO:
                papel = "alvo"
            elif coluna in cfg.FEATURES_DERIVADAS:
                papel = "feature_derivada"
            else:
                papel = "feature_original"
            ausentes = self.ausentes_tratados["diagnostico"][coluna]
            distintos = info_ids["ids_distintos"] if coluna == cfg.COLUNA_ID else self._distintos(coluna)
            linhas.append({
                "coluna": coluna,
                "papel": papel,
                "tipo_bruto": tipos_brutos.get(coluna, "(derivada)"),
                "tipo_tratado": (self.tipos_tratados or {}).get(coluna, cfg.TIPOS_TRATADOS[coluna]),
                "ausentes_brutos": self.ausentes_brutos[coluna] if coluna in self.ausentes_brutos else None,
                "distintos_brutos": len(self.valores_brutos[coluna]) if coluna in self.valores_brutos else None,
                "total_invalidos": sum(self.invalidos[coluna].values()) if coluna in self.invalidos else 0,
                "total_ausentes": ausentes,
                "percentual_ausentes": round(100 * ausentes / total, 4),
                "valores_distintos": distintos,
                "ausentes_elegiveis": self.ausentes_tratados["elegiveis"][coluna],
            })
        return pd.DataFrame(linhas)

    def tabela_frequencias(self):
        linhas = []
        for nome in CONJUNTOS:
            total = self.linhas[nome] or 1
            for coluna in cfg.COLUNAS_CATEGORICAS:
                rotulos = cfg.ROTULOS.get(coluna, {})
                contagem = self.frequencias[nome][coluna]
                for categoria in sorted(contagem, key=_ordem_categoria):
                    chave = int(categoria) if categoria.isdigit() else categoria
                    linhas.append({
                        "conjunto": nome,
                        "coluna": coluna,
                        "categoria": categoria,
                        "rotulo": "Ausente/inválido" if categoria == AUSENTE else rotulos.get(chave, ""),
                        "contagem": contagem[categoria],
                        "percentual": round(100 * contagem[categoria] / total, 4),
                    })
        return pd.DataFrame(linhas)

    def tabela_anomalias(self, info_ids):
        linhas = []
        for coluna, contagem in self.invalidos.items():
            for valor, qtd in contagem.most_common(cfg.LIMITE_EXEMPLOS_ANOMALIA):
                linhas.append({
                    "coluna": coluna,
                    "valor_bruto": valor,
                    "tipo_anomalia": classificar_anomalia(coluna, valor),
                    "quantidade_total": qtd,
                })
        if info_ids["ids_ausentes"]:
            linhas.append({"coluna": cfg.COLUNA_ID, "valor_bruto": VAZIO,
                           "tipo_anomalia": "inscricao_ausente", "quantidade_total": info_ids["ids_ausentes"]})
        for id_mascarado, qtd in info_ids["exemplos_duplicados"].items():
            linhas.append({"coluna": cfg.COLUNA_ID, "valor_bruto": id_mascarado,
                           "tipo_anomalia": "inscricao_duplicada", "quantidade_total": qtd})
        if self.notas["fora_0_1000"]:
            linhas.append({"coluna": cfg.COLUNA_ALVO, "valor_bruto": "(vários)",
                           "tipo_anomalia": "nota_fora_de_0_a_1000_mantida", "quantidade_total": self.notas["fora_0_1000"]})
        return pd.DataFrame(linhas, columns=["coluna", "valor_bruto", "tipo_anomalia", "quantidade_total"])

    def cobertura_elegiveis(self):
        """Confere que treineiros, faixas etárias e UFs continuam presentes nos elegíveis."""
        resultado = {}
        for coluna in ("IN_TREINEIRO", "TP_FAIXA_ETARIA", "SG_UF_PROVA"):
            diag = {k for k in self.frequencias["diagnostico"][coluna] if k != AUSENTE}
            eleg = {k for k in self.frequencias["elegiveis"][coluna] if k != AUSENTE}
            resultado[coluna] = {
                "categorias_diagnostico": len(diag),
                "categorias_elegiveis": len(eleg),
                "ausentes_nos_elegiveis": sorted(diag - eleg, key=_ordem_categoria),
            }
        resultado["treineiros_elegiveis"] = self.frequencias["elegiveis"]["IN_TREINEIRO"].get("1", 0)
        return resultado


def _ordem_categoria(categoria):
    if categoria == AUSENTE:
        return (2, 0, "")
    if categoria.isdigit():
        return (0, int(categoria), "")
    return (1, 0, categoria)


def verificar_invariantes(resumo, colunas_escritas):
    """Seção 12: retorna a lista de invariantes violadas (vazia = tudo certo)."""
    falhas = []
    if resumo["total_linhas_diagnostico"] != resumo["total_linhas_origem"]:
        falhas.append("total_linhas_diagnostico != total_linhas_origem")
    if resumo["total_linhas_elegiveis"] != resumo["total_notas_lc_observadas"]:
        falhas.append("total_linhas_elegiveis != total_notas_lc_observadas")
    if resumo["total_notas_lc_observadas"] + resumo["total_notas_lc_ausentes"] != resumo["total_linhas_origem"]:
        falhas.append("observadas + ausentes != total_linhas_origem")
    if resumo["total_notas_lc_zero_elegiveis"] != resumo["total_notas_lc_zero"]:
        falhas.append("notas zero não foram todas preservadas no conjunto elegível")
    for nome, colunas in colunas_escritas.items():
        if list(colunas) != cfg.COLUNAS_SAIDA:
            falhas.append(f"colunas do arquivo {nome} diferem do contrato de 17 colunas")
        proibidas = [c for c in colunas if re.match(cfg.PADRAO_COLUNAS_PROIBIDAS, c)]
        if proibidas:
            falhas.append(f"colunas proibidas no arquivo {nome}: {proibidas}")
    if resumo["linhas_parquet_diagnostico"] != resumo["total_linhas_diagnostico"]:
        falhas.append("parquet de diagnóstico não contém todas as linhas processadas")
    if resumo["linhas_parquet_elegiveis"] != resumo["total_linhas_elegiveis"]:
        falhas.append("parquet de elegíveis não contém todas as linhas processadas")
    if resumo["total_identificadores_duplicados"] > 0:
        falhas.append(f"{resumo['total_identificadores_duplicados']} NU_INSCRICAO duplicados (falha de integridade)")
    if resumo["total_identificadores_ausentes"] > 0:
        falhas.append(f"{resumo['total_identificadores_ausentes']} NU_INSCRICAO ausentes (falha de integridade)")
    return falhas
