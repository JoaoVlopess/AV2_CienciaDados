"""Conferência de origem e leitura em blocos do CSV de microdados (etapas 1 e 2)."""

import codecs
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import configuracao as cfg


class ErroArquivoFonte(Exception):
    """Arquivo de origem ausente, ilegível, de outra edição ou sem colunas obrigatórias."""


def _caminho_exibicao(caminho):
    """Caminho relativo à raiz do projeto, quando possível (para os relatórios)."""
    try:
        return Path(caminho).resolve().relative_to(cfg.RAIZ_PROJETO).as_posix()
    except ValueError:
        return Path(caminho).resolve().as_posix()


def detectar_encoding(caminho, n_bytes=1_048_576):
    """Retorna 'utf-8' se o início do arquivo for UTF-8 válido; senão 'latin-1'."""
    with open(caminho, "rb") as f:
        inicio = f.read(n_bytes)
    if inicio.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    try:
        codecs.getincrementaldecoder("utf-8")().decode(inicio, final=False)
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def ler_cabecalho(caminho, encoding):
    with open(caminho, "r", encoding=encoding, newline="") as f:
        return f.readline().rstrip("\r\n")


def detectar_separador(cabecalho):
    candidatos = [";", ",", "\t", "|"]
    return max(candidatos, key=cabecalho.count)


def hash_e_linhas(caminho, tamanho_bloco=8 * 1_048_576):
    """SHA-256 do arquivo e número de linhas físicas de dados (sem o cabeçalho)."""
    sha = hashlib.sha256()
    quebras = 0
    ultimo = b""
    with open(caminho, "rb") as f:
        while bloco := f.read(tamanho_bloco):
            sha.update(bloco)
            quebras += bloco.count(b"\n")
            ultimo = bloco[-1:]
    linhas = quebras + (1 if ultimo not in (b"\n", b"") else 0)
    return sha.hexdigest(), max(linhas - 1, 0)


def conferir_arquivo(caminho, calcular_hash=True):
    """Etapa 1: valida existência, edição, encoding, separador e colunas obrigatórias."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise ErroArquivoFonte(f"Arquivo de origem não encontrado: {caminho}")

    encoding = detectar_encoding(caminho)
    cabecalho = ler_cabecalho(caminho, encoding)
    separador = detectar_separador(cabecalho)
    colunas = [c.strip().strip('"') for c in cabecalho.split(separador)]

    faltantes = [c for c in cfg.COLUNAS_ENTRADA if c not in colunas]
    if faltantes:
        raise ErroArquivoFonte(f"Colunas obrigatórias ausentes no CSV: {', '.join(faltantes)}")

    alertas = []
    if "NU_ANO" in colunas:
        anos = pd.read_csv(caminho, sep=separador, encoding=encoding, usecols=["NU_ANO"],
                           dtype=str, nrows=1000)["NU_ANO"].dropna().unique()
        if set(anos) != {str(cfg.ANO_FONTE)}:
            raise ErroArquivoFonte(f"NU_ANO esperado {cfg.ANO_FONTE}, encontrado {sorted(anos)}")
    else:
        alertas.append("Coluna NU_ANO ausente: edição 2023 não pôde ser confirmada pelo conteúdo.")

    estat = caminho.stat()
    sha256, linhas_fisicas = hash_e_linhas(caminho) if calcular_hash else (None, None)
    return {
        "arquivo_fonte": _caminho_exibicao(caminho),
        "tamanho_bytes": estat.st_size,
        "modificado_em": datetime.fromtimestamp(estat.st_mtime).astimezone().isoformat(timespec="seconds"),
        "hash_sha256": sha256,
        "linhas_fisicas_dados": linhas_fisicas,
        "encoding_detectado": encoding,
        "separador_detectado": separador,
        "total_colunas_arquivo": len(colunas),
        "alertas": alertas,
        "_mtime": estat.st_mtime,
    }


def inferir_tipos_brutos(caminho, encoding, separador, nrows=cfg.LINHAS_AMOSTRA_TIPO_BRUTO):
    """Tipo que o pandas inferiria para cada coluna de entrada (amostra das primeiras linhas)."""
    amostra = pd.read_csv(caminho, sep=separador, encoding=encoding,
                          usecols=cfg.COLUNAS_ENTRADA, nrows=nrows)
    return {col: str(amostra[col].dtype) for col in cfg.COLUNAS_ENTRADA}


def ler_em_blocos(caminho, encoding, separador, tamanho_bloco=cfg.TAMANHO_BLOCO_PADRAO):
    """Etapa 2: lê TODAS as linhas, somente as 14 colunas da lista positiva, como texto bruto.

    Somente células vazias viram ausentes ("NA", "NULL" etc. continuam como texto
    para aparecerem como códigos inválidos na auditoria).
    """
    return pd.read_csv(
        caminho,
        sep=separador,
        encoding=encoding,
        usecols=cfg.COLUNAS_ENTRADA,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        chunksize=tamanho_bloco,
    )
