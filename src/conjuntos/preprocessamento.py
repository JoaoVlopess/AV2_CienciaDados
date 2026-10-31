"""Preparação de cada conjunto para cada família de modelo, sempre dentro de um Pipeline do scikit-learn.

Uso típico (a validação cruzada reajusta tudo só no treino de cada dobra):

    from src.conjuntos.preprocessamento import carregar_conjunto, montar_pipeline
    X, y, dobras = carregar_conjunto("C04")
    pipe = montar_pipeline("C04", "linear", Ridge())
    cross_validate(pipe, X, y, cv=dobras, scoring="neg_root_mean_squared_error")

Famílias: "linear" e "svm" usam codificação escalonada (níveis ordinais padronizados, one-hot,
imputação + indicador); "arvore" e "boosting" usam códigos ordinais e mantêm os ausentes (NaN).
"""

import json

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from . import configuracao as cfg

FAMILIAS = {"linear": "escalonado", "svm": "escalonado", "arvore": "arvore", "boosting": "arvore"}


# --- Transformadores próprios ------------------------------------------------------------------
class CodificadorNivel(TransformerMixin, BaseEstimator):
    """Converte colunas em números sem aprender nada dos dados.

    Colunas em `ordens` (letras do questionário) viram a posição na ordem do dicionário (A = 0, B = 1, …).
    modo="nivel": categorias especiais ("não sei", INDETERMINADA) viram NaN, para imputação + indicador.
    modo="codigo": especiais recebem códigos depois da ordem (as árvores separam esses grupos sozinhas).
    Demais colunas são convertidas para número; as listadas em `log` recebem log1p.
    """

    def __init__(self, ordens=None, especiais=None, modo="nivel", log=()):
        self.ordens = ordens
        self.especiais = especiais
        self.modo = modo
        self.log = log

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        ordens, especiais = self.ordens or {}, self.especiais or {}
        saida = {}
        for coluna in self.feature_names_in_:
            valores = X[coluna]
            if coluna in ordens:
                mapa = {v: float(i) for i, v in enumerate(ordens[coluna])}
                if self.modo == "codigo":
                    mapa.update({v: float(len(mapa) + j) for j, v in enumerate(especiais.get(coluna, []))})
                saida[coluna] = valores.map(mapa).astype(float)
            else:
                numero = pd.to_numeric(valores, errors="coerce").astype(float)
                saida[coluna] = np.log1p(numero) if coluna in self.log else numero
        return pd.DataFrame(saida, index=X.index).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_.copy()


class ContagemPorCategoria(TransformerMixin, BaseEstimator):
    """log(1 + nº de participantes da categoria no treino); categoria não vista = 0. Não usa o alvo."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.contagens_ = {c: X[c].value_counts() for c in self.feature_names_in_}
        return self

    def transform(self, X):
        return np.column_stack([np.log1p(X[c].map(self.contagens_[c]).astype(float).fillna(0.0))
                                for c in self.feature_names_in_])

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_PORTE" for c in self.feature_names_in_], dtype=object)


class RemoverColinearidade(TransformerMixin, BaseEstimator):
    """Remove, no treino da dobra, colunas que são combinação linear exata do intercepto e das anteriores.

    Casos reais na base: categoria do dicionário ausente nos dados (TP_COR_RACA = 6, coluna só de zeros) e
    relações estruturais entre códigos (TIPO_ESCOLA_DETALHADO = NAO_INFORMADA ⇔ TP_ST_CONCLUSAO ≠ 2).
    Sem isso a regressão linear sem regularização não tem coeficientes identificáveis. Percorre as colunas
    na ordem de saída e descarta a que não acrescenta variância residual (relativa) acima de `tolerancia`.
    """

    def __init__(self, tolerancia=1e-9):
        self.tolerancia = tolerancia

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        Z = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        gram = Z.T @ Z / len(Z)
        mantidas = [0]  # intercepto
        for j in range(1, gram.shape[0]):
            g = gram[mantidas, j]
            residuo = gram[j, j] - g @ np.linalg.lstsq(gram[np.ix_(mantidas, mantidas)], g, rcond=None)[0]
            if gram[j, j] > 0 and residuo / gram[j, j] > self.tolerancia:
                mantidas.append(j)
        self.mantidas_ = np.asarray(mantidas[1:]) - 1
        self.removidas_ = [n for i, n in enumerate(self.feature_names_in_) if i not in set(self.mantidas_)]
        return self

    def transform(self, X):
        return X.to_numpy(dtype=float)[:, self.mantidas_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_[self.mantidas_]


class MarcarCategoricas(TransformerMixin, BaseEstimator):
    """Marca como `category` as colunas do bloco categórico (prefixo "cat__").

    Assim o HistGradientBoostingRegressor(categorical_features="from_dtype") trata UF, perfil digital etc.
    como categorias, e não como números ordenados.
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        X = X.copy()
        for coluna in X.columns:
            if coluna.startswith("cat__"):
                X[coluna] = X[coluna].astype("category")
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_.copy()


# --- Conjuntos e colunas ---------------------------------------------------------------------------
def obter_conjunto(conjunto):
    if conjunto not in cfg.CONJUNTOS:
        raise KeyError(f"Conjunto desconhecido: {conjunto}. Disponíveis: {', '.join(cfg.CONJUNTOS)}")
    spec = cfg.CONJUNTOS[conjunto]
    if not spec.get("disponivel", True):
        raise ValueError(f"Conjunto {conjunto} ainda não está disponível: {spec['hipotese']}")
    return {"etapas": {}, **spec}


def colunas_para_familia(conjunto, familia):
    """Colunas de entrada do conjunto; nos modelos escalonados, sem as redundâncias exatas."""
    colunas = list(obter_conjunto(conjunto)["colunas"])
    if _modo(conjunto, familia) == "escalonado":
        presentes = set(colunas)
        removidas = {col for col, determinantes in cfg.REDUNDANCIAS_ESCALONADAS
                     if col in presentes and presentes.issuperset(determinantes)}
        colunas = [c for c in colunas if c not in removidas]
    return colunas


def _modo(conjunto, familia):
    if familia not in FAMILIAS:
        raise KeyError(f"Família desconhecida: {familia}. Use uma de: {', '.join(FAMILIAS)}")
    # PCA e SelectKBest exigem features numéricas sem NaN (padronizadas), qualquer que seja o regressor.
    etapas = obter_conjunto(conjunto)["etapas"]
    return "escalonado" if {"pca_global", "selecao"} & set(etapas) else FAMILIAS[familia]


def preparar_para_modelos(df):
    """Tipos previsíveis para o scikit-learn: categóricas/ordinais como texto (ausente = "AUSENTE"),
    numéricas e binárias como float64 (ausente = NaN). Pode ser aplicada mais de uma vez."""
    saida = df.copy()
    for coluna in saida.columns:
        tipo = cfg.TIPOS_COLUNAS.get(coluna, {}).get("tipo")
        if tipo in ("ordinal", "categorica", "municipio"):
            texto = saida[coluna].astype("string")
            saida[coluna] = texto.fillna(cfg.AUSENTE).astype(object)
        elif tipo in ("numerica", "binaria"):
            saida[coluna] = pd.to_numeric(saida[coluna], errors="coerce").astype("float64")
    return saida


# --- Montagem do pré-processamento -------------------------------------------------------------
def _bloco_ordinal(colunas, modo):
    ordens = {c: cfg.TIPOS_COLUNAS[c]["ordem"] for c in colunas}
    especiais = {c: cfg.TIPOS_COLUNAS[c]["especiais"] for c in colunas}
    if modo == "arvore":
        return CodificadorNivel(ordens, especiais, modo="codigo")
    return make_pipeline(CodificadorNivel(ordens, especiais, modo="nivel"),
                         SimpleImputer(strategy="median", add_indicator=True), StandardScaler())


def _bloco_numerico(colunas, modo):
    log = tuple(c for c in colunas if cfg.TIPOS_COLUNAS[c].get("log"))
    if modo == "arvore":
        return CodificadorNivel(log=log)
    return make_pipeline(CodificadorNivel(log=log), SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler())


def _bloco_categorico(colunas, modo):
    dominios = [cfg.TIPOS_COLUNAS[c]["dominio"] for c in colunas]
    if modo == "arvore":
        return OrdinalEncoder(categories=dominios, handle_unknown="use_encoded_value", unknown_value=np.nan)
    return OneHotEncoder(categories=dominios, drop="first", handle_unknown="ignore", sparse_output=False)


def _bloco_municipio(modo):
    # TargetEncoder faz cross-fitting no fit_transform: cada linha de treino recebe a média sem o próprio alvo.
    if _versao_sklearn() >= (1, 9):
        alvo = TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=cfg.SEMENTE))
    else:
        alvo = TargetEncoder(target_type="continuous", shuffle=True, random_state=cfg.SEMENTE)
    porte = ContagemPorCategoria()
    if modo == "escalonado":
        return [("mun_te", make_pipeline(alvo, StandardScaler())), ("mun_porte", make_pipeline(porte, StandardScaler()))]
    return [("mun_te", alvo), ("mun_porte", porte)]


def _versao_sklearn():
    return tuple(int(parte) for parte in sklearn.__version__.split(".")[:2])


def _bloco_niveis_padronizados(colunas):
    """Todas as colunas como números padronizados (entrada do PCA de bens e do k-means)."""
    ordens = {c: cfg.TIPOS_COLUNAS[c]["ordem"] for c in colunas if cfg.TIPOS_COLUNAS[c]["tipo"] == "ordinal"}
    log = tuple(c for c in colunas if cfg.TIPOS_COLUNAS[c].get("log"))
    return [CodificadorNivel(ordens, modo="nivel", log=log), SimpleImputer(strategy="median"), StandardScaler()]


def montar_preprocessador(conjunto, familia):
    """Pipeline de preparação do conjunto para a família ("linear", "svm", "arvore" ou "boosting").

    Saída em pandas, com nomes de colunas legíveis (ex.: "ord__Q006", "cat__SG_UF_PROVA_CE").
    """
    spec = obter_conjunto(conjunto)
    etapas = spec["etapas"]
    modo = _modo(conjunto, familia)
    colunas = colunas_para_familia(conjunto, familia)

    em_blocos_proprios = set(cfg.BENS_BRUTOS) if "pca_bens" in etapas else set()
    por_tipo = {"ordinal": [], "numerica": [], "binaria": [], "categorica": [], "municipio": []}
    for coluna in colunas:
        if coluna not in em_blocos_proprios:
            por_tipo[cfg.TIPOS_COLUNAS[coluna]["tipo"]].append(coluna)

    blocos = []
    if por_tipo["categorica"]:  # primeiro, para o prefixo "cat__" valer no boosting
        blocos.append(("cat", _bloco_categorico(por_tipo["categorica"], modo), por_tipo["categorica"]))
    if por_tipo["ordinal"]:
        blocos.append(("ord", _bloco_ordinal(por_tipo["ordinal"], modo), por_tipo["ordinal"]))
    if por_tipo["numerica"]:
        blocos.append(("num", _bloco_numerico(por_tipo["numerica"], modo), por_tipo["numerica"]))
    if por_tipo["binaria"]:
        blocos.append(("bin", CodificadorNivel(), por_tipo["binaria"]))
    for nome, transformador in (_bloco_municipio(modo) if por_tipo["municipio"] else []):
        blocos.append((nome, transformador, por_tipo["municipio"]))
    if "pca_bens" in etapas:
        bens = [c for c in colunas if c in em_blocos_proprios]
        pca = PCA(n_components=etapas["pca_bens"], svd_solver="full", random_state=cfg.SEMENTE)
        blocos.append(("pca_bens", make_pipeline(*_bloco_niveis_padronizados(bens), pca), bens))
    if "clusters" in etapas:
        kmeans = MiniBatchKMeans(n_clusters=etapas["clusters"], random_state=cfg.SEMENTE, n_init=3, batch_size=4096)
        blocos.append(("cluster", make_pipeline(*_bloco_niveis_padronizados(cfg.COLUNAS_CLUSTER), kmeans),
                       cfg.COLUNAS_CLUSTER))

    passos = [("preparacao", ColumnTransformer(blocos, remainder="drop"))]
    if modo == "escalonado":
        passos.append(("colinearidade", RemoverColinearidade()))
    if "selecao" in etapas:
        passos.append(("selecao", SelectKBest(f_regression, k=etapas["selecao"])))
    if "pca_global" in etapas:
        passos.append(("pca", PCA(n_components=etapas["pca_global"], svd_solver="full", random_state=cfg.SEMENTE)))
    elif familia == "boosting":
        passos.append(("categorias", MarcarCategoricas()))
    return Pipeline(passos).set_output(transform="pandas")


def montar_pipeline(conjunto, familia, regressor):
    """Pipeline completo (preparação + regressor), pronto para cross_validate / GridSearchCV.

    Hiperparâmetros do regressor ficam em "modelo__<param>"; os da preparação em "features__…"
    (ex.: "features__selecao__k" no C12).
    """
    return Pipeline([("features", montar_preprocessador(conjunto, familia)), ("modelo", regressor)])


# --- Leitura dos arquivos gerados por gerar_conjuntos ---------------------------------------------
def dobras_da_coluna(dobra):
    """[(treino, validação)] das 3 dobras fixas, a partir da coluna DOBRA (1–3), como no notebook."""
    dobra = np.asarray(dobra)
    return [(np.flatnonzero(dobra != k), np.flatnonzero(dobra == k)) for k in (1, 2, 3)]


def carregar_conjunto(conjunto, pasta=None):
    """Lê o parquet do conjunto gerado por gerar_conjuntos e devolve (X, y, dobras).

    `pasta`: pasta da geração (padrão: a mais recente em dados/conjuntos/). X contém só as colunas do
    conjunto; NU_INSCRICAO e DOBRA nunca entram em X.
    """
    if pasta is None:
        candidatas = sorted(cfg.PASTA_CONJUNTOS.glob("*/manifesto.json"), key=lambda p: p.stat().st_mtime)
        if not candidatas:
            raise FileNotFoundError("Nenhum conjunto gerado. Rode: python -m src.conjuntos.gerar_conjuntos")
        pasta = candidatas[-1].parent
    with open(pasta / "manifesto.json", encoding="utf-8") as f:
        manifesto = json.load(f)
    info = manifesto["conjuntos"][conjunto]
    df = pd.read_parquet(pasta / info["arquivo"])
    X = preparar_para_modelos(df[info["colunas_entrada"]])
    return X, df[cfg.COLUNA_ALVO].to_numpy(dtype=float), dobras_da_coluna(df[cfg.COLUNA_DOBRA])
