"""Exporta um conjunto de features com TODOS os participantes elegíveis (desenvolvimento + teste).

Uso (a partir da raiz do projeto):
    python -m src.conjuntos.exportar_conjunto C07        # -> dados/processed/enem_2023_lc_c07.parquet

Diferente de gerar_conjuntos (triagem, só desenvolvimento e em amostra), este arquivo serve à modelagem
final: cada linha traz CONJUNTO ("desenvolvimento"/"teste") e DOBRA (1–3 no desenvolvimento, 0 no teste),
as mesmas partições de dados/splits/particoes_lc.parquet. O teste só deve ser lido na avaliação final.
"""

import argparse
import sys
import time

import pandas as pd
import pyarrow.parquet as pq

from ..dados import configuracao as cfg_dados
from . import configuracao as cfg
from .base_ampliada import construir_base_ampliada, criar_ou_carregar_particoes
from .preprocessamento import obter_conjunto, preparar_para_modelos


def montar_tabela(base, particoes, conjunto):
    """NU_INSCRICAO, CONJUNTO, DOBRA, features do conjunto e alvo, na ordem da base de elegíveis."""
    colunas = obter_conjunto(conjunto)["colunas"]
    tabela = base[[cfg.COLUNA_ID, *colunas, cfg.COLUNA_ALVO]].merge(
        particoes[[cfg.COLUNA_ID, "CONJUNTO", cfg.COLUNA_DOBRA]], on=cfg.COLUNA_ID, how="left", validate="one_to_one")
    if tabela["CONJUNTO"].isna().any():
        raise RuntimeError("Há participantes elegíveis sem partição; apague dados/splits/particoes_lc.parquet e refaça.")
    tabela = tabela[[cfg.COLUNA_ID, "CONJUNTO", cfg.COLUNA_DOBRA, *colunas, cfg.COLUNA_ALVO]]
    return preparar_para_modelos(tabela)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Exporta um conjunto com todos os participantes elegíveis.")
    parser.add_argument("conjunto", help="id do conjunto (ex.: C07)")
    args = parser.parse_args(argv)
    conjunto = args.conjunto.upper()
    try:
        colunas = obter_conjunto(conjunto)["colunas"]
    except (KeyError, ValueError) as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        return 2

    inicio = time.perf_counter()
    if not cfg.ARQUIVO_BASE_AMPLIADA.exists():
        construir_base_ampliada(cfg_dados.CAMINHO_CSV_PADRAO)
    base = pd.read_parquet(cfg.ARQUIVO_BASE_AMPLIADA, columns=[cfg.COLUNA_ID, *colunas, cfg.COLUNA_ALVO])
    particoes = criar_ou_carregar_particoes(base[cfg.COLUNA_ID])
    tabela = montar_tabela(base, particoes, conjunto)

    destino = cfg_dados.PASTA_SAIDA_PADRAO / f"enem_2023_lc_{conjunto.lower()}.parquet"
    temporario = destino.with_suffix(".parquet.tmp")
    tabela.to_parquet(temporario, index=False, compression="zstd")
    temporario.replace(destino)
    linhas = f"{len(tabela):,}".replace(",", ".")
    print(f"{conjunto}: {linhas} linhas x {tabela.shape[1]} colunas -> {destino} "
          f"({destino.stat().st_size / 1e6:.0f} MB, {time.perf_counter() - inicio:.0f} s)")
    print(tabela["CONJUNTO"].value_counts().to_string())
    return 0 if pq.ParquetFile(destino).metadata.num_rows == len(base) else 1


if __name__ == "__main__":
    sys.exit(main())
