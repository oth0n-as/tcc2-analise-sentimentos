"""Módulo de inferência dos modelos de classificação de sentimentos do TCC II.

Responsável por carregar os artefatos salvos em 'models/' e executar predições
de forma rápida e desacoplada da interface gráfica, garantindo compatibilidade
com o Streamlit e cache otimizado.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np

from config import (
    CLASS_ID_TO_LABEL,
    CLASS_LABEL_TO_ID,
    CONDICOES,
    MODELS_DIR,
)
from preprocessing import preprocessar_por_condicao

# Mapeamento dos nomes dos arquivos de modelo
MODEL_FILES = {
    ("Regressão Logística", "Texto Bruto"): "lr_texto_bruto.joblib",
    ("Regressão Logística", "Limpeza Básica"): "lr_limpeza_basica.joblib",
    ("Regressão Logística", "Pipeline Completo"): "lr_pipeline_completo.joblib",
    ("Regressão Logística", "Sem Emoticons"): "lr_sem_emoticons.joblib",
    ("SVM", "Texto Bruto"): "svm_texto_bruto.joblib",
    ("SVM", "Limpeza Básica"): "svm_limpeza_basica.joblib",
    ("SVM", "Pipeline Completo"): "svm_pipeline_completo.joblib",
    ("SVM", "Sem Emoticons"): "svm_sem_emoticons.joblib",
}

# Cache local para quando não estiver sob o contexto de st.cache_resource
_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def carregar_modelo(nome_modelo: str, condicao: str) -> Optional[Any]:
    """Carrega o pipeline scikit-learn salvo via joblib."""
    chave = (nome_modelo, condicao)
    if chave in _MODEL_CACHE:
        return _MODEL_CACHE[chave]

    nome_arquivo = MODEL_FILES.get(chave)
    if not nome_arquivo:
        return None

    caminho = MODELS_DIR / nome_arquivo
    if not caminho.exists():
        return None

    try:
        pipeline = joblib.load(caminho)
        _MODEL_CACHE[chave] = pipeline
        return pipeline
    except Exception as e:
        print(f"Erro ao carregar modelo {caminho}: {e}")
        return None


def verificar_modelos_disponiveis() -> Dict[str, List[str]]:
    """Verifica quais modelos e condições possuem arquivos .joblib prontos."""
    disponibilidade = {}
    for (modelo, condicao), arquivo in MODEL_FILES.items():
        if (MODELS_DIR / arquivo).exists():
            if modelo not in disponibilidade:
                disponibilidade[modelo] = []
            disponibilidade[modelo].append(condicao)
    return disponibilidade


def classificar_texto(
    texto: str,
    condicao: str,
    nome_modelo: str = "Regressão Logística",
) -> Dict[str, Any]:
    """Realiza a inferência real de sentimento para um determinado texto e condição.

    Passos:
    1. Aplica o pré-processamento correspondente à condição;
    2. Alimenta o pipeline treinado (TF-IDF + Classificador);
    3. Retorna a classe prevista, probabilidades (se disponíveis) e textos.
    """
    if not texto or not texto.strip():
        return {
            "sucesso": False,
            "erro": "Texto de entrada vazio.",
            "texto_original": texto,
            "texto_processado": "",
            "condicao": condicao,
            "modelo": nome_modelo,
        }

    pipeline = carregar_modelo(nome_modelo, condicao)
    if pipeline is None:
        return {
            "sucesso": False,
            "erro": (
                f"Modelo '{nome_modelo}' para a condição '{condicao}' não foi encontrado. "
                "Execute 'python train_models.py' para gerar os arquivos."
            ),
            "texto_original": texto,
            "texto_processado": "",
            "condicao": condicao,
            "modelo": nome_modelo,
        }

    # Aplica o pré-processamento estrito da condição
    texto_processado = preprocessar_por_condicao(texto, condicao)

    # Caso o pré-processamento remova todos os caracteres (ex: texto composto apenas por emoticons na condição 'Sem Emoticons')
    entrada_modelo = texto_processado if texto_processado.strip() else ""

    try:
        pred_enc = int(pipeline.predict([entrada_modelo])[0])
        classe_predita = CLASS_ID_TO_LABEL.get(pred_enc, f"Classe {pred_enc}")

        # Extração de probabilidades quando o classificador suportar (ex: Regressão Logística)
        probabilidades = {}
        if hasattr(pipeline.named_steps["clf"], "predict_proba"):
            probs = pipeline.predict_proba([entrada_modelo])[0]
            for idx, prob in enumerate(probs):
                nome_classe = CLASS_ID_TO_LABEL.get(idx, f"Classe {idx}")
                probabilidades[nome_classe] = float(prob)

        # Diagnóstico da representação textual (Tokens no TF-IDF e OOV)
        diagnostico = {
            "tokens_reconhecidos": [],
            "tokens_fora_vocabulario": [],
            "pesos_features": [],
        }

        if "tfidf" in pipeline.named_steps:
            tfidf = pipeline.named_steps["tfidf"]
            analyzer = tfidf.build_analyzer()
            tokens_extraidos = analyzer(entrada_modelo)
            vocab = tfidf.vocabulary_

            reconhecidos = []
            fora_vocab = []
            for t in tokens_extraidos:
                if t in vocab:
                    if t not in reconhecidos:
                        reconhecidos.append(t)
                else:
                    if t not in fora_vocab:
                        fora_vocab.append(t)

            diagnostico["tokens_reconhecidos"] = reconhecidos
            diagnostico["tokens_fora_vocabulario"] = fora_vocab

            clf = pipeline.named_steps.get("clf")
            if clf is not None and hasattr(clf, "coef_"):
                for t in reconhecidos:
                    col_idx = vocab[t]
                    peso_pred = float(clf.coef_[pred_enc, col_idx])
                    diagnostico["pesos_features"].append({
                        "token": t,
                        "peso": round(peso_pred, 4),
                        "influencia": "Favorece a classe" if peso_pred > 0 else "Penaliza a classe",
                    })

        return {
            "sucesso": True,
            "texto_original": texto,
            "texto_processado": texto_processado,
            "condicao": condicao,
            "modelo": nome_modelo,
            "classe_id": pred_enc,
            "classe_predita": classe_predita,
            "probabilidades": probabilidades,
            "diagnostico": diagnostico,
        }
    except Exception as e:
        return {
            "sucesso": False,
            "erro": f"Erro durante a inferência: {str(e)}",
            "texto_original": texto,
            "texto_processado": texto_processado,
            "condicao": condicao,
            "modelo": nome_modelo,
        }


def comparar_todas_as_condicoes(
    texto: str,
    nome_modelo: str = "Regressão Logística",
) -> Dict[str, Any]:
    """Executa a inferência nas 4 condições experimentais do TCC II para o mesmo texto.

    Permite observar visualmente a questão central da pesquisa:
    Como o pré-processamento altera o texto e se ele chega a modificar
    a decisão final do classificador.
    """
    resultados_condicoes = []
    classes_preditas = []

    for condicao in CONDICOES:
        res = classificar_texto(texto, condicao, nome_modelo=nome_modelo)
        resultados_condicoes.append(res)
        if res.get("sucesso"):
            classes_preditas.append(res["classe_predita"])

    divergencia = len(set(classes_preditas)) > 1

    return {
        "texto_original": texto,
        "modelo": nome_modelo,
        "resultados": resultados_condicoes,
        "classes_preditas": classes_preditas,
        "divergencia_detectada": divergencia,
    }
