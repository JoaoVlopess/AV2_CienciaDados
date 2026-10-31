"""Triagem dos conjuntos de features com 3 regressores (linear, árvore de decisão e boosting).

Uso (a partir da raiz do projeto, depois de python -m src.conjuntos.gerar_conjuntos):
    python -m src.modelagem.triagem_conjuntos                       # todos os conjuntos x 3 modelos
    python -m src.modelagem.triagem_conjuntos --conjuntos C02 C04 --modelos linear boosting
    python -m src.modelagem.triagem_conjuntos --refazer             # ignora resultados já gravados

Cada conjunto é avaliado nas 3 dobras fixas do desenvolvimento, com o pipeline completo (preparação +
regressor) reajustado no treino de cada dobra. Os resultados vão para resultados/experimentos_triagem.csv,
com as mesmas colunas de experimentos.csv do notebook. Experimentos já gravados são pulados, então a
execução pode ser interrompida e retomada. O teste nunca é lido.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from ..conjuntos import configuracao as cfg
from ..conjuntos.preprocessamento import carregar_conjunto, colunas_para_familia, montar_pipeline

PASTA_RESULTADOS = cfg.RAIZ_PROJETO / "resultados"
ARQUIVO_EXPERIMENTOS = PASTA_RESULTADOS / "experimentos_triagem.csv"
ARQUIVO_RESUMO = PASTA_RESULTADOS / "triagem_conjuntos_lc.csv"
PASTA_FIGURAS = PASTA_RESULTADOS / "figuras"
METRICAS = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
CONJUNTO_REFERENCIA = "C02"  # conjunto usado hoje: os ganhos são medidos contra ele

# Configurações iniciais (antes de qualquer busca de hiperparâmetros), iguais para todos os conjuntos.
# A árvore sem limite decoraria o treino (RMSE de treino ≈ 0); profundidade 12 e folhas de 100 participantes
# são um ponto de partida comedido. O boosting usa os padrões do scikit-learn, com parada antecipada
# automática numa parcela interna (10%) do treino da dobra; a validação da dobra não é usada nela.
MODELOS = {
    "linear": {"sigla": "LIN", "rotulo": "Regressão linear", "familia": "linear", "regressor": LinearRegression()},
    "arvore": {"sigla": "ARV", "rotulo": "Árvore de decisão", "familia": "arvore",
               "regressor": DecisionTreeRegressor(max_depth=12, min_samples_leaf=100, random_state=cfg.SEMENTE)},
    "boosting": {"sigla": "HGB", "rotulo": "Boosting (HistGB)", "familia": "boosting",
                 "regressor": HistGradientBoostingRegressor(random_state=cfg.SEMENTE)},
}
PREPARACAO = {
    "linear": ("níveis ordinais padronizados; 'não sei' -> mediana + indicador; one-hot sem a 1ª categoria; "
               "log1p em renda/densidades; sem redundâncias nem colinearidade exata"),
    "arvore": "códigos ordinais (especiais no fim da ordem); one-hot não usado; NaN mantido",
    "boosting": "códigos ordinais; categóricas como category (categorical_features='from_dtype'); NaN mantido",
}

# Estilo igual ao do notebook; cores 1–3 da paleta categórica de referência (validadas para 3 séries).
CORES = {"linear": "#2a78d6", "arvore": "#eb6834", "boosting": "#1baf7a"}
COR_BASELINE = "#8a8984"
COR_TEXTO_SEC = "#52514e"
FONTE_DADOS = "Fonte: Inep, Microdados ENEM 2023. Elaboração da equipe."
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.8,
    "axes.edgecolor": "#b5b4ae", "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.labelcolor": COR_TEXTO_SEC, "xtick.color": COR_TEXTO_SEC, "ytick.color": COR_TEXTO_SEC,
    "legend.frameon": False, "font.size": 10,
})


def _ler_experimentos(arquivo):
    return pd.read_csv(arquivo, dtype={"conclusao": "object"}) if arquivo.exists() else pd.DataFrame()


def salvar_experimento(registro, arquivo):
    tabela = _ler_experimentos(arquivo)
    if len(tabela):
        tabela = tabela[tabela["id_experimento"] != registro["id_experimento"]]
    tabela = pd.concat([tabela, pd.DataFrame([registro])], ignore_index=True)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(arquivo, index=False)


def avaliar(id_experimento, id_conjunto, modelo, pipeline, X, y, dobras, hipotese, colunas, preparacao,
            conjunto_avaliado, arquivo):
    """Valida o pipeline nas 3 dobras e grava uma linha no formato de experimentos.csv."""
    inicio = time.perf_counter()
    res = cross_validate(pipeline, X, y, cv=dobras, scoring=METRICAS, return_train_score=True,
                         return_estimator=True, n_jobs=1, error_score="raise")
    tempo_total = time.perf_counter() - inicio
    regressor = pipeline.steps[-1][1]
    n_saida = [e.steps[-1][1].n_features_in_ for e in res["estimator"]] if colunas else [0]
    registro = {
        "id_experimento": id_experimento, "data_hora": datetime.now().isoformat(timespec="seconds"),
        "alvo": cfg.COLUNA_ALVO, "hipotese": hipotese, "features": "|".join(colunas),
        "n_features_entrada": len(colunas), "preparacao": preparacao, "modelo": type(regressor).__name__,
        "parametros": json.dumps(regressor.get_params(), default=str), "conjunto_avaliado": conjunto_avaliado,
        "semente": cfg.SEMENTE, "id_conjunto": id_conjunto, "familia": modelo, "linhas_avaliadas": len(y),
        "n_features_saida": int(np.median(n_saida)),
    }
    for metrica in METRICAS:
        sinal = 1 if metrica == "r2" else -1
        validacao, treino = sinal * res[f"test_{metrica}"], sinal * res[f"train_{metrica}"]
        for k, valor in enumerate(validacao, start=1):
            registro[f"{metrica}_dobra{k}"] = valor
        registro[f"{metrica}_media"] = validacao.mean()
        registro[f"{metrica}_dp"] = validacao.std(ddof=1)
        registro[f"{metrica}_treino_media"] = treino.mean()
    registro["tempo_ajuste_medio_s"] = res["fit_time"].mean()
    registro["tempo_total_s"] = tempo_total
    registro["conclusao"] = ""
    salvar_experimento(registro, arquivo)
    print(f"  {id_experimento:<22} RMSE = {registro['rmse_media']:6.2f} ± {registro['rmse_dp']:.2f} | "
          f"MAE = {registro['mae_media']:6.2f} | R² = {registro['r2_media']:.4f} | "
          f"treino {registro['rmse_treino_media']:6.2f} | {tempo_total:5.0f} s", flush=True)
    return registro


def _prefixo(pasta):
    """TRI-A300000 para a amostra de 300 mil, TRI-DEV para o desenvolvimento completo."""
    nome = pasta.name
    if nome == "dev_completo":
        return "TRI-DEV"
    return "TRI-A" + nome.removeprefix("dev_amostra_") if nome.startswith("dev_amostra_") else f"TRI-{nome.upper()}"


def executar(pasta, conjuntos, modelos, arquivo=ARQUIVO_EXPERIMENTOS, refazer=False):
    """Roda baseline + conjuntos x modelos e devolve a tabela de experimentos da triagem desta pasta."""
    conjunto_avaliado = f"validacao_cruzada_{pasta.name}_3_dobras"
    prefixo = _prefixo(pasta)
    feitos = _ler_experimentos(arquivo)
    feitos = set() if refazer or feitos.empty else set(feitos["id_experimento"])

    tarefas = [(f"{prefixo}-BASE", conjuntos[0], None)] + [
        (f"{prefixo}-{c}-{MODELOS[m]['sigla']}", c, m) for c in conjuntos for m in modelos]
    pendentes = [t for t in tarefas if t[0] not in feitos]
    print(f"{len(tarefas)} experimentos, {len(tarefas) - len(pendentes)} já gravados, {len(pendentes)} a rodar")
    for id_experimento, conjunto, modelo in pendentes:
        X, y, dobras = carregar_conjunto(conjunto, pasta)
        if modelo is None:  # o baseline não usa features: prevê a média do treino de cada dobra
            baseline = Pipeline([("modelo", DummyRegressor(strategy="mean"))])
            avaliar(id_experimento, "-", "baseline", baseline, X.iloc[:, :1], y, dobras,
                    "Referência: média do treino de cada dobra, sem features.", [], "nenhuma",
                    conjunto_avaliado, arquivo)
            continue
        spec = cfg.CONJUNTOS[conjunto]
        etapas = spec.get("etapas", {})
        preparacao = PREPARACAO[modelo] + (f"; etapas: {json.dumps(etapas)}" if etapas else "")
        if "CO_MUNICIPIO_PROVA" in spec["colunas"]:
            preparacao += "; município: target encoding com cross-fitting + porte (log de inscritos)"
        pipeline = montar_pipeline(conjunto, MODELOS[modelo]["familia"], MODELOS[modelo]["regressor"])
        avaliar(id_experimento, conjunto, modelo, pipeline, X, y, dobras,
                f"{spec['nome']}: {spec['hipotese']}", colunas_para_familia(conjunto, MODELOS[modelo]["familia"]),
                preparacao, conjunto_avaliado, arquivo)

    tabela = _ler_experimentos(arquivo)
    return tabela[(tabela["conjunto_avaliado"] == conjunto_avaliado)
                  & tabela["id_experimento"].isin([t[0] for t in tarefas])]


def resumir(tabela):
    """Uma linha por conjunto: RMSE médio ± dp de cada modelo e diferença para o conjunto de referência."""
    modelos = tabela[tabela["familia"] != "baseline"]
    resumo = modelos.pivot(index="id_conjunto", columns="familia",
                           values=["rmse_media", "rmse_dp", "mae_media", "r2_media", "rmse_treino_media",
                                   "n_features_saida", "tempo_total_s"])
    resumo.columns = [f"{metrica}_{familia}" for metrica, familia in resumo.columns]
    ordem = [c for c in cfg.CONJUNTOS if c in resumo.index]
    resumo = resumo.loc[ordem]
    resumo.insert(0, "nome", [cfg.CONJUNTOS[c]["nome"] for c in ordem])
    for familia in modelos["familia"].unique():
        if CONJUNTO_REFERENCIA in resumo.index:
            resumo[f"delta_rmse_vs_{CONJUNTO_REFERENCIA}_{familia}"] = (
                resumo[f"rmse_media_{familia}"] - resumo.loc[CONJUNTO_REFERENCIA, f"rmse_media_{familia}"])
        if "C11" in resumo.index:
            resumo[f"delta_rmse_vs_C11_{familia}"] = resumo[f"rmse_media_{familia}"] - resumo.loc["C11", f"rmse_media_{familia}"]
    resumo = resumo.round(4)
    baseline = tabela.loc[tabela["familia"] == "baseline", "rmse_media"]
    resumo.attrs["rmse_baseline"] = float(baseline.iloc[0]) if len(baseline) else np.nan
    return resumo


def _familias(resumo):
    return [m for m in MODELOS if f"rmse_media_{m}" in resumo.columns]


def figura_conjuntos(resumo, destino, titulo_amostra):
    """RMSE de validação (média ± dp nas 3 dobras) por conjunto, um marcador por modelo."""
    principais = resumo[~resumo.index.str.startswith("A_")]
    familias = _familias(principais)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(principais) + 1.6))
    posicoes = np.arange(len(principais))[::-1]
    deslocamento = {m: d for m, d in zip(familias, np.linspace(0.22, -0.22, len(familias)))}  # ordem da legenda
    for familia in familias:
        ax.errorbar(principais[f"rmse_media_{familia}"], posicoes + deslocamento[familia],
                    xerr=principais[f"rmse_dp_{familia}"], fmt="o", ms=6, color=CORES[familia],
                    ecolor=CORES[familia], elinewidth=1.5, capsize=0, label=MODELOS[familia]["rotulo"],
                    markeredgecolor="white", markeredgewidth=1)
    if CONJUNTO_REFERENCIA in principais.index:
        ax.axhspan(posicoes[principais.index.get_loc(CONJUNTO_REFERENCIA)] - 0.45,
                   posicoes[principais.index.get_loc(CONJUNTO_REFERENCIA)] + 0.45, color="#f1f0ec", zorder=0)
    ax.set_yticks(posicoes, [f"{c}  {n.replace('_', ' ').lower()}" for c, n in zip(principais.index, principais["nome"])])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RMSE na validação (pontos da nota; menor é melhor)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08 - 0.6 / len(principais)), ncol=len(familias))
    ax.set_title("Triagem dos conjuntos de features — validação cruzada (3 dobras)", loc="left", pad=30)
    baseline = resumo.attrs.get("rmse_baseline", np.nan)
    ax.text(0.0, 1.01, f"{titulo_amostra}. Baseline pela média: RMSE {baseline:.2f}.\n"
            f"Faixa cinza = conjunto atual ({CONJUNTO_REFERENCIA}). Barras = ±1 dp entre dobras.",
            transform=ax.transAxes, fontsize=8.5, color=COR_TEXTO_SEC, va="bottom")
    fig.text(0.0, -0.02, FONTE_DADOS, fontsize=8, color=COR_TEXTO_SEC, va="top")
    fig.savefig(destino)
    plt.close(fig)


def figura_ablacao(resumo, destino, titulo_amostra):
    """Quanto o RMSE de C11 piora ao retirar cada bloco (barras horizontais, uma por modelo)."""
    ablacoes = resumo[resumo.index.str.startswith("A_")]
    familias = [m for m in _familias(ablacoes) if f"delta_rmse_vs_C11_{m}" in ablacoes.columns]
    if ablacoes.empty or not familias:
        return False
    ordem = ablacoes[f"delta_rmse_vs_C11_{familias[-1]}"].sort_values().index
    ablacoes = ablacoes.loc[ordem]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(ablacoes) + 1.6))
    altura = 0.8 / len(familias)
    posicoes = np.arange(len(ablacoes))
    for i, familia in enumerate(familias):
        ax.barh(posicoes + ((len(familias) - 1) / 2 - i) * altura, ablacoes[f"delta_rmse_vs_C11_{familia}"],
                height=altura - 0.04, color=CORES[familia], label=MODELOS[familia]["rotulo"],
                edgecolor="white", linewidth=1)
    ax.axvline(0, color=COR_BASELINE, lw=1)
    ax.set_yticks(posicoes, [c.removeprefix("A_").lower() for c in ablacoes.index])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Variação do RMSE ao retirar o bloco (pontos; positivo = o bloco ajudava)")
    ax.legend(loc="lower right")
    ax.set_title("Ablação por blocos a partir do conjunto completo (C11)", loc="left", pad=20)
    ax.text(0.0, 1.01, f"{titulo_amostra}. Média das 3 dobras de validação.", transform=ax.transAxes,
            fontsize=8.5, color=COR_TEXTO_SEC, va="bottom")
    fig.text(0.0, -0.02, FONTE_DADOS, fontsize=8, color=COR_TEXTO_SEC, va="top")
    fig.savefig(destino)
    plt.close(fig)
    return True


def _pasta_mais_recente():
    candidatas = sorted(cfg.PASTA_CONJUNTOS.glob("*/manifesto.json"), key=lambda p: p.stat().st_mtime)
    return candidatas[-1].parent if candidatas else None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Triagem dos conjuntos de features com linear, árvore e boosting.")
    parser.add_argument("--pasta", type=Path, default=None,
                        help="pasta gerada por src.conjuntos.gerar_conjuntos (padrão: a mais recente)")
    parser.add_argument("--conjuntos", nargs="+", default=None, metavar="ID", help="padrão: todos os da pasta")
    parser.add_argument("--modelos", nargs="+", default=list(MODELOS), choices=list(MODELOS))
    parser.add_argument("--experimentos", type=Path, default=ARQUIVO_EXPERIMENTOS,
                        help="CSV de experimentos (padrão: resultados/experimentos_triagem.csv)")
    parser.add_argument("--refazer", action="store_true", help="roda de novo mesmo o que já foi gravado")
    args = parser.parse_args(argv)

    pasta = args.pasta or _pasta_mais_recente()
    if pasta is None or not (pasta / "manifesto.json").exists():
        print("ERRO: rode antes  python -m src.conjuntos.gerar_conjuntos", file=sys.stderr)
        return 2
    with open(pasta / "manifesto.json", encoding="utf-8") as f:
        manifesto = json.load(f)
    disponiveis = list(manifesto["conjuntos"])
    conjuntos = args.conjuntos or disponiveis
    faltando = [c for c in conjuntos if c not in disponiveis]
    if faltando:
        print(f"ERRO: conjuntos não gerados em {pasta}: {', '.join(faltando)}", file=sys.stderr)
        return 2

    amostra = manifesto["amostra"]
    titulo_amostra = (f"{amostra['tipo'].capitalize()}: {amostra['linhas']:,} participantes".replace(",", "."))
    print(f"Pasta: {pasta}\n{titulo_amostra}")
    inicio = time.perf_counter()
    tabela = executar(pasta, conjuntos, args.modelos, args.experimentos, args.refazer)

    resumo = resumir(tabela)
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(ARQUIVO_RESUMO)
    figura_conjuntos(resumo, PASTA_FIGURAS / "lc_triagem_conjuntos.png", titulo_amostra)
    tem_ablacao = figura_ablacao(resumo, PASTA_FIGURAS / "lc_triagem_ablacao.png", titulo_amostra)

    colunas = [f"rmse_media_{m}" for m in _familias(resumo)]
    print(f"\nRMSE médio na validação (baseline {resumo.attrs['rmse_baseline']:.2f}):")
    print(resumo[["nome", *colunas]].to_string())
    print(f"\nResumo: {ARQUIVO_RESUMO}\nFiguras: {PASTA_FIGURAS / 'lc_triagem_conjuntos.png'}"
          + (f", {PASTA_FIGURAS / 'lc_triagem_ablacao.png'}" if tem_ablacao else "")
          + f"\nConcluído em {time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
