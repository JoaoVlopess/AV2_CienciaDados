"""Gera um parquet por conjunto de features (C01–C15 e ablações) para a triagem dos modelos.

Uso (a partir da raiz do projeto):
    python -m src.conjuntos.gerar_conjuntos                    # amostra de 300.000 do desenvolvimento
    python -m src.conjuntos.gerar_conjuntos --amostra 0        # desenvolvimento completo (~2,26 milhões)
    python -m src.conjuntos.gerar_conjuntos --conjuntos C02 C04 C08
    python -m src.conjuntos.gerar_conjuntos --refazer-base     # relê o CSV e reconstrói a base ampliada

Somente o DESENVOLVIMENTO (80%) é exportado; o teste continua reservado.
A amostra serve para comparar conjuntos; o vencedor deve ser confirmado no desenvolvimento completo,
pois o enunciado não permite trocar a base nacional por uma amostra na comparação principal.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..dados import configuracao as cfg_dados
from . import configuracao as cfg
from .base_ampliada import construir_base_ampliada, criar_ou_carregar_particoes
from .preprocessamento import FAMILIAS, colunas_para_familia, obter_conjunto, preparar_para_modelos


def selecionar_desenvolvimento(base, particoes, amostra, semente=cfg.SEMENTE):
    """Linhas do desenvolvimento (com DOBRA), opcionalmente uma amostra aleatória sem reposição.

    A amostra é sorteada dentro de cada dobra (1/3 do total em cada) e preserva a dobra de cada participante:
    as 3 dobras da triagem são subconjuntos das dobras oficiais. Usa default_rng: RandomState(42).choice
    repetiria a permutação do KFold(random_state=42) e sortearia só participantes da dobra 1.
    """
    dev = particoes.loc[particoes["CONJUNTO"] == "desenvolvimento", [cfg.COLUNA_ID, cfg.COLUNA_DOBRA, "ORDEM"]]
    if amostra and amostra < len(dev):
        rng = np.random.default_rng(semente)
        dobra = dev[cfg.COLUNA_DOBRA].to_numpy()
        posicoes = [rng.choice(np.flatnonzero(dobra == k), size=n, replace=False)
                    for k, n in zip((1, 2, 3), np.diff(np.linspace(0, amostra, 4).round().astype(int)))]
        dev = dev.iloc[np.sort(np.concatenate(posicoes))]
    dev = dev.merge(base, on=cfg.COLUNA_ID, how="inner", validate="one_to_one").sort_values("ORDEM")
    return dev.drop(columns="ORDEM").reset_index(drop=True)


def exportar(dev, pasta, conjuntos, descricao_amostra):
    pasta.mkdir(parents=True, exist_ok=True)
    manifesto = {
        "versao_conjuntos": cfg.VERSAO_CONJUNTOS,
        "data_hora_execucao": datetime.now().astimezone().isoformat(timespec="seconds"),
        "alvo": cfg.COLUNA_ALVO,
        "amostra": descricao_amostra,
        "linhas_por_dobra": {str(k): int(v) for k, v in dev[cfg.COLUNA_DOBRA].value_counts().sort_index().items()},
        "como_usar": "X, y, dobras = src.conjuntos.preprocessamento.carregar_conjunto(id, pasta)",
        "tipos_colunas": cfg.TIPOS_COLUNAS,
        "redundancias_escalonadas": cfg.REDUNDANCIAS_ESCALONADAS,
        "conjuntos": {},
        "indisponiveis": {},
    }
    controle = [cfg.COLUNA_ID, cfg.COLUNA_DOBRA, cfg.COLUNA_ALVO]
    for conjunto in conjuntos:
        spec = cfg.CONJUNTOS[conjunto]
        if not spec.get("disponivel", True):
            manifesto["indisponiveis"][conjunto] = spec["hipotese"]
            print(f"  {conjunto:<14} indisponível: {spec['hipotese']}")
            continue
        spec = obter_conjunto(conjunto)
        arquivo = f"{conjunto}.parquet"
        dados = preparar_para_modelos(dev[controle + spec["colunas"]])
        dados.to_parquet(pasta / arquivo, index=False, compression="zstd")
        manifesto["conjuntos"][conjunto] = {
            "nome": spec["nome"],
            "hipotese": spec["hipotese"],
            "arquivo": arquivo,
            "colunas_entrada": spec["colunas"],
            "n_colunas_entrada": len(spec["colunas"]),
            "etapas_no_pipeline": spec["etapas"],
            "colunas_por_familia": {f: colunas_para_familia(conjunto, f) for f in ("linear", "arvore")},
            "megabytes": round((pasta / arquivo).stat().st_size / 1e6, 1),
        }
        print(f"  {conjunto:<14} {spec['nome']:<24} {len(spec['colunas']):>3} colunas  "
              f"{manifesto['conjuntos'][conjunto]['megabytes']:>6.1f} MB")
    with open(pasta / "manifesto.json", "w", encoding="utf-8") as f:
        json.dump(manifesto, f, ensure_ascii=False, indent=2)
    return manifesto


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gera os conjuntos de features para a triagem dos modelos (NU_NOTA_LC).")
    parser.add_argument("--amostra", type=int, default=cfg.AMOSTRA_PADRAO,
                        help=f"linhas do desenvolvimento; 0 = todas (padrão: {cfg.AMOSTRA_PADRAO})")
    parser.add_argument("--conjuntos", nargs="+", default=list(cfg.CONJUNTOS), metavar="ID",
                        help="ids a gerar (padrão: todos). Ex.: C02 C04 A_RENDA")
    parser.add_argument("--saida", type=Path, default=None,
                        help="pasta de saída (padrão: dados/conjuntos/dev_amostra_<N> ou dev_completo)")
    parser.add_argument("--entrada", type=Path, default=cfg_dados.CAMINHO_CSV_PADRAO, help="CSV original do Inep")
    parser.add_argument("--refazer-base", action="store_true", help="reconstrói enem_2023_lc_ampliada.parquet")
    args = parser.parse_args(argv)

    desconhecidos = [c for c in args.conjuntos if c not in cfg.CONJUNTOS]
    if desconhecidos:
        print(f"ERRO: conjuntos desconhecidos: {', '.join(desconhecidos)}. Disponíveis: {', '.join(cfg.CONJUNTOS)}",
              file=sys.stderr)
        return 2
    if not cfg.ARQUIVO_ELEGIVEIS.exists():
        print("ERRO: rode antes  python -m src.dados.executar_tratamento", file=sys.stderr)
        return 2

    inicio = time.perf_counter()
    print("[1/3] Base ampliada")
    if args.refazer_base or not cfg.ARQUIVO_BASE_AMPLIADA.exists():
        base = construir_base_ampliada(args.entrada)
    else:
        base = pd.read_parquet(cfg.ARQUIVO_BASE_AMPLIADA)
        print(f"      reutilizando {cfg.ARQUIVO_BASE_AMPLIADA.relative_to(cfg.RAIZ_PROJETO)}")

    print("[2/3] Partições e seleção do desenvolvimento")
    particoes = criar_ou_carregar_particoes(base[cfg.COLUNA_ID])
    dev = selecionar_desenvolvimento(base, particoes, args.amostra)
    del base
    total_dev = int((particoes["CONJUNTO"] == "desenvolvimento").sum())
    completo = not args.amostra or args.amostra >= total_dev
    descricao = {
        "tipo": "desenvolvimento completo" if completo else "amostra aleatória do desenvolvimento, 1/3 por dobra",
        "linhas": len(dev),
        "linhas_desenvolvimento": total_dev,
        "semente": cfg.SEMENTE,
        "particoes": cfg.ARQUIVO_PARTICOES.relative_to(cfg.RAIZ_PROJETO).as_posix(),
    }
    print(f"      {descricao['tipo']}: {len(dev):,} de {total_dev:,} linhas".replace(",", "."))

    pasta = args.saida or cfg.PASTA_CONJUNTOS / ("dev_completo" if completo else f"dev_amostra_{len(dev)}")
    print(f"[3/3] Exportando para {pasta}")
    exportar(dev, pasta, args.conjuntos, descricao)
    print(f"Concluído em {time.perf_counter() - inicio:.0f} s. Famílias aceitas em montar_pipeline: {', '.join(FAMILIAS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
