"""Teste de ponta a ponta da triagem (src/modelagem) com dados fictícios.

Rodar a partir da raiz do projeto:  python -m pytest -q
"""

import numpy as np
import pandas as pd

from src.conjuntos import configuracao as cfg
from src.conjuntos.gerar_conjuntos import exportar
from src.modelagem import triagem_conjuntos as triagem
from tests.test_conjuntos import _base_ficticia

CONJUNTOS = ["C01", "C02", "C11", "A_RENDA"]


def test_triagem_grava_um_experimento_por_conjunto_e_modelo(tmp_path):
    base, y = _base_ficticia(n=1500, semente=1)
    dev = base.assign(**{cfg.COLUNA_ID: [f"{i:012d}" for i in range(len(base))],
                         cfg.COLUNA_DOBRA: np.resize([1, 2, 3], len(base)).astype("int8"),
                         cfg.COLUNA_ALVO: y})
    pasta = tmp_path / "dev_amostra_1500"
    exportar(dev, pasta, CONJUNTOS, {"tipo": "teste", "linhas": len(dev)})
    arquivo = tmp_path / "experimentos.csv"

    tabela = triagem.executar(pasta, CONJUNTOS, list(triagem.MODELOS), arquivo)
    assert len(tabela) == 1 + len(CONJUNTOS) * len(triagem.MODELOS)
    assert tabela["id_experimento"].str.startswith("TRI-A1500-").all()
    assert tabela[["rmse_media", "mae_media", "r2_media"]].notna().all().all()
    assert set(tabela["familia"]) == {"baseline", "linear", "arvore", "boosting"}

    # Retomada: nada é refeito na segunda chamada.
    gravado = pd.read_csv(arquivo)["data_hora"].tolist()
    triagem.executar(pasta, CONJUNTOS, list(triagem.MODELOS), arquivo)
    assert pd.read_csv(arquivo)["data_hora"].tolist() == gravado

    resumo = triagem.resumir(tabela)
    assert list(resumo.index) == CONJUNTOS
    assert resumo.loc["C02", "delta_rmse_vs_C02_linear"] == 0
    triagem.figura_conjuntos(resumo, tmp_path / "conjuntos.png", "teste")
    assert triagem.figura_ablacao(resumo, tmp_path / "ablacao.png", "teste")
    assert (tmp_path / "conjuntos.png").stat().st_size > 0 and (tmp_path / "ablacao.png").stat().st_size > 0
