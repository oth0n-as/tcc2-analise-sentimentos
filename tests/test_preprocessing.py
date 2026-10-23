"""Testes unitários para o módulo de pré-processamento textual do TCC II.

Garante que as transformações de texto, remoção de emoticons,
preservação de negações e stemming funcionem com precisão metodológica.
"""

import pytest
from preprocessing import (
    limpar_texto_pt,
    remover_emoticons,
    aplicar_pipeline_pt,
    preprocessar_por_condicao,
    NEGACOES_PT,
    STOPWORDS_PT,
)


def test_limpar_texto_pt_urls_mencoes():
    raw = "Veja isso https://t.co/abc1234 e fale com @amigo!"
    limpo = limpar_texto_pt(raw)
    assert "https" not in limpo
    assert "@amigo" not in limpo
    assert "amigo" not in limpo
    assert "veja isso e fale com" in limpo


def test_limpar_texto_pt_hashtags():
    raw = "Notícia urgente sobre #fato e #oportunidade"
    limpo = limpar_texto_pt(raw)
    assert "#" not in limpo
    assert "fato" in limpo
    assert "oportunidade" in limpo


def test_limpar_texto_pt_alongamento():
    raw = "Estou muitoooooooo cansadooooo"
    limpo = limpar_texto_pt(raw)
    # Redução para no máximo 2 repetições
    assert "muitoo" in limpo
    assert "cansadoo" in limpo
    assert "muitoooooooo" not in limpo


def test_limpar_texto_pt_nao_alfa():
    raw = "Hoje é dia 25/10/2026! Preço: R$ 50,00 *top*."
    limpo = limpar_texto_pt(raw)
    assert "/" not in limpo
    assert "$" not in limpo
    assert "25" not in limpo
    assert "hoje é dia" in limpo
    assert "preço r" in limpo


def test_remover_emoticons():
    raw = "Adorei o produto :) muito bom mesmo :( :D ;) XD"
    sem_emo = remover_emoticons(raw)
    assert ":)" not in sem_emo
    assert ":(" not in sem_emo
    assert ":D" not in sem_emo
    assert ";)" not in sem_emo
    assert "Adorei o produto muito bom mesmo" in sem_emo


def test_remover_emoticons_preserva_urls():
    raw = "http://example.com/page com texto :)"
    sem_emo = remover_emoticons(raw)
    assert "http://example.com/page" in sem_emo
    assert ":)" not in sem_emo


def test_preservacao_negacoes_em_stopwords():
    for negacao in ["não", "nao", "nunca", "jamais", "nem"]:
        assert negacao in NEGACOES_PT
        assert negacao not in STOPWORDS_PT


def test_aplicar_pipeline_pt_com_negacao():
    texto_base = "eu não gostei do produto nunca mais compro"
    processado = aplicar_pipeline_pt(texto_base, usar_stopwords=True, usar_stemming=False)
    # As palavras de negação devem permanecer
    assert "não" in processado or "nao" in processado
    assert "nunca" in processado
    # A palavra 'do' (< 3 letras ou stopword) deve ter sido removida
    assert " do " not in f" {processado} "


def test_aplicar_pipeline_pt_com_stemming():
    texto_base = "gostei do atendimento maravilhoso"
    processado = aplicar_pipeline_pt(texto_base, usar_stopwords=True, usar_stemming=True)
    # RSLPStemmer reduz 'gostei' -> 'gost'
    assert "gost" in processado


def test_preprocessar_por_condicao_quatro_modos():
    texto = "Adorei @loja https://loja.com :) muitooooo bom!"
    
    bruto = preprocessar_por_condicao(texto, "Texto Bruto")
    assert bruto == texto.strip()

    basica = preprocessar_por_condicao(texto, "Limpeza Básica")
    assert "https" not in basica
    assert "@loja" not in basica
    assert "muitoo" in basica

    completo = preprocessar_por_condicao(texto, "Pipeline Completo")
    assert len(completo) > 0

    sem_emo = preprocessar_por_condicao(texto, "Sem Emoticons")
    assert ":)" not in sem_emo
    assert "Adorei @loja" in sem_emo


def test_preprocessar_por_condicao_entradas_invalidas():
    assert preprocessar_por_condicao("", "Texto Bruto") == ""
    assert preprocessar_por_condicao(None, "Limpeza Básica") == ""
    with pytest.raises(ValueError):
        preprocessar_por_condicao("teste", "Condição Inexistente")
