"""Script de treinamento e exportação dos modelos do TCC II.

Reproduz rigorosamente os passos do notebook original (Células 3, 5, 7 e 11):
1. Carregamento e auditoria do corpus Train3Classes.csv;
2. Remoção de conflitos de rótulo e duplicatas exatas;
3. Split estratificado 80/20 com SEED=42;
4. Geração das 4 representações textuais (Bruto, Limpeza Básica, Completo, Sem Emoticons);
5. Construção de sklearn.pipeline.Pipeline (TF-IDF + Classificador);
6. Treinamento da Regressão Logística (e SVM LinearSVC);
7. Avaliação e validação das métricas no conjunto de teste;
8. Exportação dos pipelines para a pasta 'models/' via joblib.
"""

import json
import logging
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import (
    CLASS_ID_TO_LABEL,
    CONDICAO_COLUNAS,
    CONDICOES,
    DATA_DIR,
    LOGISTIC_REGRESSION_CONFIG,
    MODELS_DIR,
    RAW_DATASET_PATH,
    RESULTS_DIR,
    SEED,
    SVM_CONFIG,
    TEST_SIZE,
    TFIDF_CONFIG,
)
from preprocessing import (
    aplicar_pipeline_pt,
    ensure_nltk_resources,
    limpar_texto_pt,
    remover_emoticons,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_models")


def carregar_e_auditar_dataset(caminho_csv: Path) -> pd.DataFrame:
    """Carrega o dataset e reproduz o tratamento da Célula 3 do notebook."""
    logger.info(f"Carregando dataset de {caminho_csv}...")
    if not caminho_csv.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado em {caminho_csv}. "
            "Certifique-se de que 'Train3Classes.csv' está na pasta 'data/'."
        )

    df_raw = pd.read_csv(
        caminho_csv,
        sep=";",
        encoding="utf-8",
        on_bad_lines="skip",
    )
    logger.info(f"Dataset bruto carregado: {len(df_raw):,} registros.")

    # 1. Remover nulos e vazios em tweet_text e sentiment
    df = df_raw.dropna(subset=["tweet_text", "sentiment"]).copy()
    df = df[df["tweet_text"].astype(str).str.strip() != ""].copy()

    # 2. Remover textos com conflito de rótulo
    rotulos_por_texto = df.groupby("tweet_text")["sentiment"].nunique()
    textos_conflitantes = set(rotulos_por_texto[rotulos_por_texto > 1].index)
    if textos_conflitantes:
        df = df[~df["tweet_text"].isin(textos_conflitantes)].copy()
        logger.info(
            f"Removidos por conflito de rótulo: {len(textos_conflitantes)} textos distintos."
        )

    # 3. Remover duplicatas exatas mantendo o primeiro
    antes_dedup = len(df)
    df = df.drop_duplicates(subset="tweet_text", keep="first").reset_index(drop=True)
    removidos_dedup = antes_dedup - len(df)
    logger.info(f"Duplicatas exatas removidas: {removidos_dedup:,}.")

    # Garantir codificação inteira do sentimento
    df["sentiment_enc"] = df["sentiment"].astype(int)
    logger.info(f"Corpus experimental pronto: {len(df):,} registros.")
    return df


def gerar_representacoes_textuais(df: pd.DataFrame) -> pd.DataFrame:
    """Gera as 4 representações textuais do TCC (Célula 7 do notebook)."""
    ensure_nltk_resources()
    logger.info("Gerando representação: Limpeza Básica (_base)...")
    df["_base"] = df["tweet_text"].astype(str).map(limpar_texto_pt)

    logger.info("Gerando representação: Pipeline Completo (text_limpo: stopwords + stemming)...")
    df["text_limpo"] = df["_base"].map(
        lambda b: aplicar_pipeline_pt(b, usar_stopwords=True, usar_stemming=True)
    )

    logger.info("Gerando representação: Sem Emoticons (texto_sem_emoticons)...")
    df["texto_sem_emoticons"] = df["tweet_text"].astype(str).map(remover_emoticons)

    logger.info("✓ Todas as representações textuais foram geradas.")
    return df


def treinar_e_avaliar_condicao(
    nome_modelo: str,
    nome_condicao: str,
    coluna_texto: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    """Cria e treina um Pipeline scikit-learn (TF-IDF + Classificador) para uma condição."""
    logger.info(f"--- Treinando {nome_modelo} | Condição: {nome_condicao} ---")

    X_train = df_train[coluna_texto].fillna("").astype(str)
    y_train = df_train["sentiment_enc"]
    X_test = df_test[coluna_texto].fillna("").astype(str)
    y_test = df_test["sentiment_enc"]

    # Instancia o vetorizador TF-IDF com parâmetros idênticos ao notebook
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_CONFIG["max_features"],
        ngram_range=TFIDF_CONFIG["ngram_range"],
        min_df=TFIDF_CONFIG["min_df"],
        sublinear_tf=TFIDF_CONFIG["sublinear_tf"],
    )

    # Instancia o classificador
    if nome_modelo == "Regressão Logística":
        clf = LogisticRegression(
            max_iter=LOGISTIC_REGRESSION_CONFIG["max_iter"],
            random_state=LOGISTIC_REGRESSION_CONFIG["random_state"],
            n_jobs=LOGISTIC_REGRESSION_CONFIG["n_jobs"],
        )
    elif nome_modelo == "SVM":
        clf = LinearSVC(
            random_state=SVM_CONFIG["random_state"]
        )
    else:
        raise ValueError(f"Modelo não suportado: {nome_modelo}")

    # Cria o Pipeline unificado
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])

    inicio = time.time()
    pipeline.fit(X_train, y_train)
    tempo_treino = time.time() - inicio

    y_pred = pipeline.predict(X_test)

    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    acc = float(accuracy_score(y_test, y_pred))

    logger.info(
        f"✓ {nome_modelo} [{nome_condicao}] concluído em {tempo_treino:.2f}s | "
        f"F1-Macro: {f1_macro:.4f} | Acurácia: {acc:.4f}"
    )

    metricas = {
        "modelo": nome_modelo,
        "condicao": nome_condicao,
        "f1_macro": round(f1_macro, 6),
        "accuracy": round(acc, 6),
        "tempo_segundos": round(tempo_treino, 2),
    }

    return pipeline, metricas


def executar_treinamento_completo(treinar_svm: bool = True):
    """Executa todo o pipeline de treinamento e persiste os modelos e métricas."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carrega dados e aplica auditoria
    df = carregar_e_auditar_dataset(RAW_DATASET_PATH)

    # 2. Split experimental 80/20 estratificado
    indices = np.arange(len(df))
    idx_tr, idx_te = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["sentiment_enc"],
    )
    logger.info(f"Split estratificado: Treino={len(idx_tr):,} | Teste={len(idx_te):,}")

    # 3. Pré-processamento
    df = gerar_representacoes_textuais(df)

    df_train = df.iloc[idx_tr].copy()
    df_test = df.iloc[idx_te].copy()

    # Validação idêntica ao notebook
    assert len(set(idx_tr) & set(idx_te)) == 0, "Índices sobrepostos!"
    assert (
        df_train["tweet_text"].isin(df_test["tweet_text"]).sum() == 0
    ), "Tweets duplicados entre treino e teste!"

    # Lista de modelos a treinar
    modelos_para_treinar = ["Regressão Logística"]
    if treinar_svm:
        modelos_para_treinar.append("SVM")

    slug_condicoes = {
        "Texto Bruto": "texto_bruto",
        "Limpeza Básica": "limpeza_basica",
        "Pipeline Completo": "pipeline_completo",
        "Sem Emoticons": "sem_emoticons",
    }

    slug_modelos = {
        "Regressão Logística": "lr",
        "SVM": "svm",
    }

    resumo_metricas = []
    metadata = {
        "data_treinamento": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "split_treino": len(df_train),
        "split_teste": len(df_test),
        "classes": CLASS_ID_TO_LABEL,
        "condicoes": CONDICOES,
        "modelos": {},
    }

    for modelo_nome in modelos_para_treinar:
        m_slug = slug_modelos[modelo_nome]
        for condicao in CONDICOES:
            coluna = CONDICAO_COLUNAS[condicao]
            c_slug = slug_condicoes[condicao]

            pipeline, metricas = treinar_e_avaliar_condicao(
                modelo_nome,
                condicao,
                coluna,
                df_train,
                df_test,
            )

            # Salva o pipeline joblib
            nome_arquivo = f"{m_slug}_{c_slug}.joblib"
            caminho_modelo = MODELS_DIR / nome_arquivo
            joblib.dump(pipeline, caminho_modelo)
            logger.info(f"Pipeline salvo em: {caminho_modelo}")

            resumo_metricas.append({
                "Modelo": modelo_nome,
                "Condição": condicao,
                "F1 Macro": metricas["f1_macro"],
                "Accuracy": metricas["accuracy"],
                "Tempo (s)": metricas["tempo_segundos"],
                "Arquivo": nome_arquivo,
            })

            if modelo_nome not in metadata["modelos"]:
                metadata["modelos"][modelo_nome] = {}
            metadata["modelos"][modelo_nome][condicao] = metricas

    # Salva resumo em CSV
    df_metricas = pd.DataFrame(resumo_metricas)
    caminho_csv_metricas = RESULTS_DIR / "metricas_treinamento_local.csv"
    df_metricas.to_csv(caminho_csv_metricas, index=False)
    logger.info(f"Métricas salvas em: {caminho_csv_metricas}")

    # Salva metadados em JSON
    caminho_json_meta = MODELS_DIR / "metadata.json"
    with open(caminho_json_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadados salvos em: {caminho_json_meta}")

    print("\n" + "=" * 70)
    print("RESUMO DO TREINAMENTO LOCAL (TCC II)")
    print("=" * 70)
    print(df_metricas.to_string(index=False))
    print("=" * 70)
    logger.info("✓ Todos os modelos foram treinados e exportados com sucesso!")


if __name__ == "__main__":
    executar_treinamento_completo(treinar_svm=True)
