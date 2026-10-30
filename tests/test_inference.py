"""Testes para o módulo de inferência."""

import pytest
from inference import classificar_texto, comparar_todas_as_condicoes


def test_classificar_texto_vazio():
    res = classificar_texto("", "Texto Bruto")
    assert res["sucesso"] is False
    assert "vazio" in res["erro"].lower()

    res_spaces = classificar_texto("   \n\t ", "Limpeza Básica")
    assert res_spaces["sucesso"] is False


def test_classificar_condicao_invalida():
    res = classificar_texto("Adorei o produto", "CondicaoInexistente")
    assert res["sucesso"] is False


def test_comparar_todas_as_condicoes_estrutura():
    texto = "Muito feliz com a novidade! Parabéns pelo trabalho."
    resultado = comparar_todas_as_condicoes(texto, "Regressão Logística")
    assert "texto_original" in resultado
    assert "resultados" in resultado
    assert len(resultado["resultados"]) == 4


def test_inferencia_real_com_modelos_treinados():
    # Sentença claramente positiva
    res_pos = classificar_texto("Estou muito feliz com isso, parabéns!", "Texto Bruto", "Regressão Logística")
    assert res_pos["sucesso"] is True
    assert res_pos["classe_predita"] == "Positivo"
    assert "probabilidades" in res_pos
    assert res_pos["probabilidades"]["Positivo"] > 0.5

    # Sentença claramente negativa
    res_neg = classificar_texto("Estou muito triste e com medo, infelizmente", "Texto Bruto", "Regressão Logística")
    assert res_neg["sucesso"] is True
    assert res_neg["classe_predita"] == "Negativo"
    assert res_neg["probabilidades"]["Negativo"] > 0.5


def test_inferencia_neutro_noticioso():
    # Sentença neutra / noticiosa
    res_neu = classificar_texto("G1 noticia nova oportunidade de trabalho em São Paulo", "Texto Bruto", "Regressão Logística")
    assert res_neu["sucesso"] is True
    assert res_neu["classe_predita"] == "Neutro"


def test_inferencia_svm():
    # Testa o modelo SVM
    res_svm_pos = classificar_texto("Estou muito feliz e contente!", "Texto Bruto", "SVM")
    assert res_svm_pos["sucesso"] is True
    assert res_svm_pos["classe_predita"] == "Positivo"


def test_classificar_texto_vazio_apos_processamento():
    # Texto composto apenas de emoticons na condição 'Sem Emoticons'
    res_emoticon = classificar_texto(":-)", "Sem Emoticons", "Regressão Logística")
    assert res_emoticon["sucesso"] is True
    assert res_emoticon["texto_vazio"] is True
    assert res_emoticon["classe_predita"] == "Indeterminado"
    assert "vazio" in res_emoticon["diagnostico"]["aviso"].lower()

    # Texto composto apenas de stopwords na condição 'Pipeline Completo'
    res_stop = classificar_texto("de para com", "Pipeline Completo", "Regressão Logística")
    assert res_stop["sucesso"] is True
    assert res_stop["texto_vazio"] is True
    assert res_stop["classe_predita"] == "Indeterminado"


def test_diagnostico_tfidf_contribuicao_efetiva():
    res = classificar_texto("Adorei o atendimento rápido", "Texto Bruto", "Regressão Logística")
    assert res["sucesso"] is True
    diagnostico = res["diagnostico"]
    pesos = diagnostico.get("pesos_features", [])
    assert len(pesos) > 0

    primeiro_termo = pesos[0]
    assert "token" in primeiro_termo
    assert "tipo" in primeiro_termo
    assert primeiro_termo["tipo"] in ["Unigrama", "Bigrama"]
    assert "tfidf" in primeiro_termo
    assert "coeficiente" in primeiro_termo
    assert "contribuicao" in primeiro_termo
    # Valida relação matemática: contribuicao ~= tfidf * coeficiente
    assert abs(primeiro_termo["contribuicao"] - round(primeiro_termo["tfidf"] * primeiro_termo["coeficiente"], 4)) < 0.001


def test_comparar_todas_as_condicoes_com_texto_vazio_parcial():
    # Texto composto apenas de emoticons: na condição 'Sem Emoticons' fica vazio
    res = comparar_todas_as_condicoes(":-)", "Regressão Logística")
    assert res["total_condicoes"] == 4
    # Pelo menos a condição 'Sem Emoticons' ficou vazia
    condicoes_vazias = [r for r in res["resultados"] if r.get("texto_vazio")]
    assert len(condicoes_vazias) >= 1
    # Verifica que condicoes_validas exclui a vazia
    assert "Sem Emoticons" not in res["condicoes_validas"]


