"""Base ampliada (elegíveis + colunas extras + features derivadas) e partições desenvolvimento/teste.

A base ampliada parte de enem_2023_lc_elegiveis.parquet (já validado por src/dados) e acrescenta
as colunas de cfg.COLUNAS_EXTRAS lidas do CSV original, com a mesma regra de domínio:
código fora do dicionário vira ausente e é contabilizado; nada é imputado.
"""

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
from sklearn.model_selection import KFold, train_test_split

from ..dados import configuracao as cfg_dados
from ..dados.leitura import conferir_arquivo
from . import configuracao as cfg
from .engenharia import adicionar_features_ampliadas


def _padronizar_texto(serie):
    texto = serie.astype("string").str.strip().str.upper()
    return texto.mask(texto == "")


def ler_colunas_extras(caminho_csv, ids_elegiveis):
    """Lê NU_INSCRICAO + colunas extras do CSV, só das linhas elegíveis, e valida os domínios.

    Retorna (DataFrame com as colunas extras como texto, auditoria por coluna).
    """
    origem = conferir_arquivo(caminho_csv, calcular_hash=False)
    colunas = [cfg.COLUNA_ID, *cfg.COLUNAS_EXTRAS]
    tabela = pcsv.read_csv(
        caminho_csv,
        read_options=pcsv.ReadOptions(encoding=origem["encoding_detectado"], block_size=64 << 20),
        parse_options=pcsv.ParseOptions(delimiter=origem["separador_detectado"]),
        convert_options=pcsv.ConvertOptions(include_columns=colunas, column_types={c: pa.string() for c in colunas},
                                            strings_can_be_null=True),
    )
    tabela = tabela.filter(pc.is_in(tabela[cfg.COLUNA_ID], value_set=pa.array(ids_elegiveis, pa.string())))
    extras = tabela.to_pandas(types_mapper={pa.string(): pd.StringDtype("pyarrow")}.get)

    auditoria = {}
    for coluna in cfg.COLUNAS_EXTRAS:
        valores = _padronizar_texto(extras[coluna])
        if coluna == "CO_MUNICIPIO_PROVA":
            validos = valores.str.fullmatch(cfg.PADRAO_MUNICIPIO).fillna(False)
        else:
            validos = valores.isin(cfg.DOMINIOS_EXTRAS[coluna])
        invalidos = valores.notna() & ~validos
        auditoria[coluna] = {
            "ausentes": int(valores.isna().sum()),
            "invalidos_tratados_como_ausentes": int(invalidos.sum()),
            "exemplos_invalidos": valores[invalidos].value_counts().head(5).to_dict(),
        }
        extras[coluna] = valores.mask(invalidos)
    return extras, auditoria


def construir_base_ampliada(caminho_csv=cfg_dados.CAMINHO_CSV_PADRAO, arquivo_elegiveis=cfg.ARQUIVO_ELEGIVEIS,
                            destino=cfg.ARQUIVO_BASE_AMPLIADA, verbose=True):
    """Gera dados/processed/enem_2023_lc_ampliada.parquet (mesmas linhas e ordem dos elegíveis)."""
    def log(msg):
        if verbose:
            print(msg, flush=True)

    inicio = time.perf_counter()
    log(f"Lendo elegíveis: {arquivo_elegiveis}")
    elegiveis = pd.read_parquet(arquivo_elegiveis)
    log(f"Lendo {len(cfg.COLUNAS_EXTRAS)} colunas extras de {caminho_csv}")
    extras, auditoria = ler_colunas_extras(caminho_csv, elegiveis[cfg.COLUNA_ID].to_numpy())

    base = elegiveis.merge(extras, on=cfg.COLUNA_ID, how="left", validate="one_to_one", indicator=True)
    sem_correspondencia = int((base.pop("_merge") != "both").sum())
    if sem_correspondencia or len(base) != len(elegiveis):
        raise RuntimeError(f"{sem_correspondencia} inscrições dos elegíveis não foram encontradas no CSV.")

    base = adicionar_features_ampliadas(base)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(".parquet.tmp")
    base.to_parquet(temporario, index=False, compression="zstd")
    temporario.replace(destino)

    resumo = {
        "arquivo": destino.relative_to(cfg.RAIZ_PROJETO).as_posix(),
        "linhas": len(base),
        "colunas": list(base.columns),
        "colunas_extras": cfg.COLUNAS_EXTRAS,
        "features_derivadas_ampliadas": cfg.FEATURES_DERIVADAS_AMPLIADAS,
        "auditoria_colunas_extras": auditoria,
        "ausentes_derivadas": {c: int(base[c].isna().sum()) for c in cfg.FEATURES_DERIVADAS_AMPLIADAS},
        "versao_conjuntos": cfg.VERSAO_CONJUNTOS,
        "data_hora_execucao": datetime.now().astimezone().isoformat(timespec="seconds"),
        "duracao_segundos": round(time.perf_counter() - inicio, 1),
    }
    cfg.ARQUIVO_RESUMO_AMPLIADA.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.ARQUIVO_RESUMO_AMPLIADA, "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    log(f"Base ampliada: {len(base):,} linhas x {base.shape[1]} colunas -> {destino}".replace(",", "."))
    return base


def criar_ou_carregar_particoes(ids, caminho=cfg.ARQUIVO_PARTICOES):
    """Divisão 80/20 e dobras do desenvolvimento, idênticas às do notebook regressao_linear_lc.ipynb.

    `ids` deve estar na ordem de enem_2023_lc_elegiveis.parquet. Se o arquivo já existir
    (gerado aqui ou no Colab), ele é reutilizado: todos os experimentos usam as mesmas partições.
    """
    if caminho.exists():
        particoes = pd.read_parquet(caminho)
    else:
        ids_dev, ids_teste = train_test_split(pd.Series(ids), test_size=0.20, random_state=cfg.SEMENTE, shuffle=True)
        dobra = np.zeros(len(ids_dev), dtype="int8")
        kfold = KFold(n_splits=3, shuffle=True, random_state=cfg.SEMENTE)
        for k, (_, idx_validacao) in enumerate(kfold.split(ids_dev), start=1):
            dobra[idx_validacao] = k
        particoes = pd.concat([
            pd.DataFrame({cfg.COLUNA_ID: ids_dev.to_numpy(), "CONJUNTO": "desenvolvimento",
                          "DOBRA": dobra, "ORDEM": np.arange(len(ids_dev))}),
            pd.DataFrame({cfg.COLUNA_ID: ids_teste.to_numpy(), "CONJUNTO": "teste",
                          "DOBRA": np.int8(0), "ORDEM": np.arange(len(ids_teste))}),
        ], ignore_index=True)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        particoes.to_parquet(caminho, index=False)
    if not (particoes[cfg.COLUNA_ID].is_unique and len(particoes) == len(ids)):
        raise RuntimeError(f"{caminho} não corresponde aos participantes elegíveis atuais.")
    return particoes
