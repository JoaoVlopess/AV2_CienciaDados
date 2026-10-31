"""Orquestração do tratamento ENEM 2023 (Linguagens).

Uso (a partir da raiz do projeto):
    python -m src.dados.executar_tratamento
    python -m src.dados.executar_tratamento --entrada caminho/MICRODADOS_ENEM_2023.csv --bloco 250000
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import configuracao as cfg
from .engenharia_features import adicionar_features
from .leitura import ErroArquivoFonte, conferir_arquivo, inferir_tipos_brutos, ler_em_blocos
from .tratamento import tratar_bloco
from .validacao import AuditoriaQualidade, estatisticas_de_contagens, verificar_invariantes

_TIPOS_ARROW = {"string": pa.string(), "Int8": pa.int8(), "Float64": pa.float64()}

REGRAS_APLICADAS = [
    "Leitura por lista positiva: somente as 14 colunas de entrada (12 features + NU_INSCRICAO + NU_NOTA_LC).",
    "Base nacional completa, sem filtro de UF, idade, escola ou treineiro.",
    "Textos: espaços removidos, maiúsculas, vazio = ausente. Códigos fora do dicionário = ausente e contabilizado.",
    "Nenhuma imputação, padronização, codificação aprendida ou exclusão por ausência de feature.",
    "RENDA_PER_CAPITA_APROX = ponto médio da faixa Q006 (2023) / Q005; A = 0; Q, Q005 inválido/ausente = ausente; sem arredondamento.",
    "ESCOLARIDADE_MAX_RESP = maior entre Q001 e Q002 só quando ambos em A–G; caso contrário INDETERMINADA. Ordem: A<B<C<D<E<F<G.",
    "PERFIL_ACESSO_DIGITAL = combinação Q024 (A = sem computador; B–E = com) e Q025 (A = não; B = sim); demais = INDETERMINADO.",
    "Conjunto elegível = NU_NOTA_LC observada e numérica, incluindo nota zero.",
]


def esquema_parquet():
    return pa.schema([(c, _TIPOS_ARROW[cfg.TIPOS_TRATADOS[c]]) for c in cfg.COLUNAS_SAIDA])


def _gravar(escritor, caminho, df, esquema):
    tabela = pa.Table.from_pandas(df, schema=esquema, preserve_index=False)
    if escritor is None:
        escritor = pq.ParquetWriter(caminho, esquema, compression="zstd")
    escritor.write_table(tabela)
    return escritor


def _salvar_csv(df, caminho):
    # ';' e vírgula decimal para abrir direto no Excel em português; UTF-8 com BOM.
    df.to_csv(caminho, sep=";", decimal=",", index=False, encoding="utf-8-sig")


def executar(caminho_csv=cfg.CAMINHO_CSV_PADRAO, pasta_saida=cfg.PASTA_SAIDA_PADRAO,
             tamanho_bloco=cfg.TAMANHO_BLOCO_PADRAO, total_esperado=cfg.TOTAL_LINHAS_ESPERADO,
             calcular_hash=True, verbose=True):
    """Executa todas as etapas. Retorna o dicionário de resumo (status 'OK' ou 'FALHOU')."""
    def log(msg):
        if verbose:
            print(msg, flush=True)

    inicio = time.perf_counter()
    data_hora = datetime.now().astimezone().isoformat(timespec="seconds")
    caminho_csv, pasta_saida = Path(caminho_csv), Path(pasta_saida)
    pasta_relatorios = pasta_saida / cfg.PASTA_RELATORIOS
    pasta_relatorios.mkdir(parents=True, exist_ok=True)

    # Etapa 1 — conferência de origem
    log(f"[1/6] Conferindo arquivo de origem: {caminho_csv}")
    if calcular_hash:
        log("      calculando SHA-256 e contando linhas (pode levar ~1 min)...")
    origem = conferir_arquivo(caminho_csv, calcular_hash=calcular_hash)
    log(f"      encoding={origem['encoding_detectado']}  separador='{origem['separador_detectado']}'  "
        f"colunas={origem['total_colunas_arquivo']}  tamanho={origem['tamanho_bytes'] / 1e9:.2f} GB")
    tipos_brutos = inferir_tipos_brutos(caminho_csv, origem["encoding_detectado"], origem["separador_detectado"])

    # Etapas 2 a 8 — leitura em blocos, tratamento, features e gravação incremental
    log(f"[2/6] Lendo o arquivo completo em blocos de {tamanho_bloco:,} linhas".replace(",", "."))
    auditoria = AuditoriaQualidade()
    esquema = esquema_parquet()
    destinos = {
        "diagnostico": pasta_saida / cfg.ARQUIVO_DIAGNOSTICO,
        "elegiveis": pasta_saida / cfg.ARQUIVO_ELEGIVEIS,
    }
    temporarios = {n: p.with_suffix(".parquet.tmp") for n, p in destinos.items()}
    escritores = {n: None for n in destinos}
    try:
        for i, bruto in enumerate(ler_em_blocos(caminho_csv, origem["encoding_detectado"],
                                                origem["separador_detectado"], tamanho_bloco), start=1):
            auditoria.registrar_bruto(bruto)
            tratado, invalidos = tratar_bloco(bruto)
            auditoria.registrar_invalidos(bruto, invalidos)
            completo = adicionar_features(tratado)
            elegiveis = completo[completo[cfg.COLUNA_ALVO].notna().to_numpy()]

            for nome, df in (("diagnostico", completo), ("elegiveis", elegiveis)):
                escritores[nome] = _gravar(escritores[nome], temporarios[nome], df, esquema)
                auditoria.registrar_conjunto(nome, df)
            log(f"      bloco {i:>3}: {auditoria.total_linhas_origem:>10,} linhas lidas".replace(",", "."))
        for nome in destinos:  # garante arquivo mesmo se o CSV não tiver linhas
            if escritores[nome] is None:
                escritores[nome] = pq.ParquetWriter(temporarios[nome], esquema)
    finally:
        for escritor in escritores.values():
            if escritor is not None:
                escritor.close()

    # Etapa 6 — duplicatas
    log("[3/6] Verificando identificadores")
    info_ids = auditoria.verificar_ids()

    # Seção 12 — resumo e invariantes
    log("[4/6] Conferindo invariantes")
    estat_origem = Path(caminho_csv).stat()
    alertas = list(origem["alertas"])
    resumo = {
        "status": None,
        "ano_fonte": cfg.ANO_FONTE,
        "arquivo_fonte": origem["arquivo_fonte"],
        "hash_sha256": origem["hash_sha256"],
        "tamanho_bytes_fonte": origem["tamanho_bytes"],
        "arquivo_fonte_inalterado": (estat_origem.st_size == origem["tamanho_bytes"]
                                     and estat_origem.st_mtime == origem["_mtime"]),
        "encoding_detectado": origem["encoding_detectado"],
        "separador_detectado": origem["separador_detectado"],
        "total_linhas_esperado_enunciado": total_esperado,
        "linhas_fisicas_dados_arquivo": origem["linhas_fisicas_dados"],
        "total_linhas_origem": auditoria.total_linhas_origem,
        "total_linhas_diagnostico": auditoria.linhas["diagnostico"],
        "total_notas_lc_observadas": auditoria.notas["observadas"],
        "total_notas_lc_ausentes": auditoria.notas["ausentes"],
        "total_notas_lc_zero": auditoria.notas["zero"],
        "total_notas_lc_zero_elegiveis": auditoria.notas["zero_elegiveis"],
        "total_notas_lc_invalidas": sum(auditoria.invalidos[cfg.COLUNA_ALVO].values()),
        "total_linhas_elegiveis": auditoria.linhas["elegiveis"],
        "elegiveis_com_alguma_feature_ausente": auditoria.elegiveis_com_feature_ausente,
        "total_identificadores_duplicados": info_ids["total_identificadores_duplicados"],
        "total_identificadores_ausentes": info_ids["ids_ausentes"],
        "linhas_parquet_diagnostico": pq.ParquetFile(temporarios["diagnostico"]).metadata.num_rows,
        "linhas_parquet_elegiveis": pq.ParquetFile(temporarios["elegiveis"]).metadata.num_rows,
        "colunas_saida": cfg.COLUNAS_SAIDA,
        "features_originais": cfg.FEATURES_ORIGINAIS,
        "features_derivadas": cfg.FEATURES_DERIVADAS,
        "ordem_escolaridade_max_resp": list(cfg.ORDEM_ESCOLARIDADE),
        "renda_representativa_q006": {k: (None if v != v else v) for k, v in cfg.RENDA_REPRESENTATIVA_Q006.items()},
        "estatisticas_numericas": {
            nome: {c: estatisticas_de_contagens(auditoria.numericos[nome][c]) for c in cfg.COLUNAS_NUMERICAS}
            for nome in ("diagnostico", "elegiveis")
        },
        "cobertura_elegiveis": auditoria.cobertura_elegiveis(),
        "invalidos_por_coluna": {c: sum(v.values()) for c, v in auditoria.invalidos.items()},
        "versao_regras_tratamento": cfg.VERSAO_REGRAS_TRATAMENTO,
        "data_hora_execucao": data_hora,
        "duracao_segundos": None,
        "regras_e_alertas": {"regras": REGRAS_APLICADAS, "alertas": alertas, "invariantes_violadas": []},
    }

    if total_esperado is not None and resumo["total_linhas_origem"] != total_esperado:
        alertas.append(f"Total de linhas ({resumo['total_linhas_origem']}) difere do esperado no enunciado ({total_esperado}).")
    if resumo["linhas_fisicas_dados_arquivo"] not in (None, resumo["total_linhas_origem"]):
        alertas.append("Número de linhas físicas difere do número de registros lidos (verificar quebras de linha em campos).")
    if not resumo["arquivo_fonte_inalterado"]:
        alertas.append("O arquivo de origem mudou de tamanho ou data durante a execução.")
    if resumo["total_notas_lc_invalidas"]:
        alertas.append(f"{resumo['total_notas_lc_invalidas']} valores de NU_NOTA_LC não numéricos tratados como ausentes.")
    if auditoria.notas["fora_0_1000"]:
        alertas.append(f"{auditoria.notas['fora_0_1000']} notas de LC fora de 0–1000 (mantidas).")
    for coluna in ("IN_TREINEIRO", "TP_FAIXA_ETARIA", "SG_UF_PROVA"):
        faltando = resumo["cobertura_elegiveis"][coluna]["ausentes_nos_elegiveis"]
        if faltando:
            alertas.append(f"Categorias de {coluna} sem registros elegíveis: {faltando}")

    colunas_escritas = {n: pq.ParquetFile(p).schema_arrow.names for n, p in temporarios.items()}
    falhas = verificar_invariantes(resumo, colunas_escritas)
    resumo["regras_e_alertas"]["invariantes_violadas"] = falhas
    resumo["status"] = "FALHOU" if falhas else "OK"

    # Finalização dos arquivos somente se todas as invariantes passarem
    log("[5/6] Finalizando arquivos de dados")
    for nome, temporario in temporarios.items():
        if falhas:
            temporario.unlink(missing_ok=True)
        else:
            temporario.replace(destinos[nome])

    # Relatórios (gravados sempre, para registrar o problema em caso de falha)
    log("[6/6] Gravando relatórios")
    _salvar_csv(auditoria.tabela_qualidade(tipos_brutos, info_ids), pasta_relatorios / "qualidade_colunas.csv")
    _salvar_csv(auditoria.tabela_frequencias(), pasta_relatorios / "frequencias_categorias.csv")
    _salvar_csv(auditoria.tabela_anomalias(info_ids), pasta_relatorios / "exemplos_anomalias.csv")
    resumo["duracao_segundos"] = round(time.perf_counter() - inicio, 1)
    with open(pasta_relatorios / "resumo_processamento.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    return resumo


def _imprimir_resumo(resumo, pasta_saida):
    def n(valor):
        return f"{valor:,}".replace(",", ".")

    print()
    print("=" * 64)
    print(f" STATUS: {resumo['status']}   ({resumo['duracao_segundos']} s)")
    print("=" * 64)
    print(f" Linhas na origem ............ {n(resumo['total_linhas_origem'])}"
          f"  (esperado: {n(resumo['total_linhas_esperado_enunciado'] or 0)})")
    print(f" Linhas no diagnóstico ....... {n(resumo['total_linhas_diagnostico'])}")
    print(f" Notas LC observadas ......... {n(resumo['total_notas_lc_observadas'])}")
    print(f" Notas LC ausentes ........... {n(resumo['total_notas_lc_ausentes'])}")
    print(f" Notas LC iguais a zero ...... {n(resumo['total_notas_lc_zero'])}")
    print(f" Linhas elegíveis ............ {n(resumo['total_linhas_elegiveis'])}")
    print(f" Inscrições duplicadas ....... {n(resumo['total_identificadores_duplicados'])}")
    print(f" Inválidos por coluna ........ "
          + (", ".join(f"{c}={n(v)}" for c, v in resumo["invalidos_por_coluna"].items() if v) or "nenhum"))
    for alerta in resumo["regras_e_alertas"]["alertas"]:
        print(f" ALERTA: {alerta}")
    for falha in resumo["regras_e_alertas"]["invariantes_violadas"]:
        print(f" FALHA:  {falha}")
    print(f" Saídas em: {pasta_saida}")
    print("=" * 64)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tratamento dos microdados ENEM 2023 para prever NU_NOTA_LC.")
    parser.add_argument("--entrada", type=Path, default=cfg.CAMINHO_CSV_PADRAO,
                        help="CSV original do Inep (padrão: microdados_enem_2023/DADOS/MICRODADOS_ENEM_2023.csv)")
    parser.add_argument("--saida", type=Path, default=cfg.PASTA_SAIDA_PADRAO,
                        help="pasta de saída (padrão: dados/processed)")
    parser.add_argument("--bloco", type=int, default=cfg.TAMANHO_BLOCO_PADRAO,
                        help="linhas por bloco de leitura; diminua se faltar memória (padrão: 500000)")
    parser.add_argument("--total-esperado", type=int, default=cfg.TOTAL_LINHAS_ESPERADO,
                        help="total de linhas esperado para conferência (padrão: 3933955)")
    parser.add_argument("--sem-hash", action="store_true", help="não calcula o SHA-256 (mais rápido)")
    args = parser.parse_args(argv)

    try:
        resumo = executar(args.entrada, args.saida, args.bloco, args.total_esperado,
                          calcular_hash=not args.sem_hash)
    except ErroArquivoFonte as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        return 2
    _imprimir_resumo(resumo, args.saida)
    return 0 if resumo["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
