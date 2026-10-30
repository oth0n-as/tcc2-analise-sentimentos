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

    # Caso o pré-processamento remova todos os caracteres (ex: texto composto apenas por emoticons na condição 'Sem Emoticons' ou apenas stopwords no 'Pipeline Completo')
    if not texto_processado or not texto_processado.strip():
        return {
            "sucesso": True,
            "texto_vazio": True,
            "erro": None,
            "texto_original": texto,
            "texto_processado": "",
            "condicao": condicao,
            "modelo": nome_modelo,
            "classe_id": None,
            "classe_predita": "Indeterminado",
            "probabilidades": {},
            "diagnostico": {
                "tokens_reconhecidos": [],
                "tokens_fora_vocabulario": [],
                "pesos_features": [],
                "aviso": (
                    f"O texto ficou vazio após a aplicação do pré-processamento da condição '{condicao}' "
                    "(todos os elementos foram filtrados pelas regras). "
                    "Nenhum token restou para alimentar o vetor TF-IDF."
                ),
            },
        }

    entrada_modelo = texto_processado

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

        # Diagnóstico da representação textual (Tokens no TF-IDF, OOV e Contribuição Efetiva)
        diagnostico = {
            "tokens_reconhecidos": [],
            "tokens_fora_vocabulario": [],
            "pesos_features": [],
            "aviso": None,
        }

        if "tfidf" in pipeline.named_steps:
            tfidf = pipeline.named_steps["tfidf"]
            clf = pipeline.named_steps.get("clf")
            vocab = tfidf.vocabulary_

            # Vetorização real do texto de entrada para capturar pesos TF-IDF efetivos
            X_tfidf = tfidf.transform([entrada_modelo])

            analyzer = tfidf.build_analyzer()
            tokens_extraidos = analyzer(entrada_modelo)

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

            if not reconhecidos:
                diagnostico["aviso"] = (
                    "Nenhum termo gerado pela tokenização foi encontrado no vocabulário de treino (Out-of-Vocabulary). "
                    "A predição foi baseada exclusivamente no viés (intercept) a priori do modelo."
                )

            # Contribuição efetiva: TF-IDF(termo) * Coeficiente(classe, termo)
            if clf is not None and hasattr(clf, "coef_"):
                for t in reconhecidos:
                    col_idx = vocab[t]
                    tfidf_val = float(X_tfidf[0, col_idx])
                    coef = float(clf.coef_[pred_enc, col_idx])
                    contribuicao = tfidf_val * coef
                    tipo_token = "Unigrama" if len(t.split()) == 1 else "Bigrama"

                    if contribuicao > 0.0001:
                        influencia = "Favorece a classe prevista"
                    elif contribuicao < -0.0001:
                        influencia = "Penaliza a classe prevista"
                    else:
                        influencia = "Impacto neutro / residual"

                    diagnostico["pesos_features"].append({
                        "token": t,
                        "tipo": tipo_token,
                        "tfidf": round(tfidf_val, 4),
                        "coeficiente": round(coef, 4),
                        "peso": round(contribuicao, 4),  # Mantido para compatibilidade retroativa
                        "contribuicao": round(contribuicao, 4),
                        "influencia": influencia,
                    })

                # Ordena os termos pela magnitude absoluta da contribuição efetiva na decisão
                diagnostico["pesos_features"].sort(
                    key=lambda item: abs(item["contribuicao"]),
                    reverse=True,
                )

        return {
            "sucesso": True,
            "texto_vazio": False,
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
            "texto_vazio": False,
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
    condicoes_validas = []

    for condicao in CONDICOES:
        res = classificar_texto(texto, condicao, nome_modelo=nome_modelo)
        resultados_condicoes.append(res)
        if res.get("sucesso") and not res.get("texto_vazio"):
            classes_preditas.append(res["classe_predita"])
            condicoes_validas.append(condicao)

    # Divergência detectada apenas se houver mais de uma classe distinta entre as condições com texto válido
    classes_unicas = set(classes_preditas)
    divergencia = len(classes_unicas) > 1

    return {
        "texto_original": texto,
        "modelo": nome_modelo,
        "resultados": resultados_condicoes,
        "classes_preditas": classes_preditas,
        "condicoes_validas": condicoes_validas,
        "divergencia_detectada": divergencia,
        "total_validas": len(classes_preditas),
        "total_condicoes": len(CONDICOES),
    }
