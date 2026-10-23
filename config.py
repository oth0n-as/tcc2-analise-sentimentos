"""Configurações centrais do projeto TCC II — Análise de Sentimentos.

Centraliza hiperparâmetros, caminhos, mapeamento de classes e metadados
para garantir fidelidade metodológica entre o notebook e o protótipo.
"""

from pathlib import Path

# ============================================================
# DIRETÓRIOS DO PROJETO
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RAW_DATASET_PATH = DATA_DIR / "Train3Classes.csv"

# ============================================================
# REPRODUTIBILIDADE E SPLIT EXPERIMENTAL
# ============================================================
SEED = 42
TEST_SIZE = 0.20

# ============================================================
# CLASSES E CODIFICAÇÕES (Conforme Célula 3 e 5 do Notebook)
# ============================================================
# 0: Negativo (tweets coletados com query ':(')
# 1: Positivo (tweets coletados com query ':)')
# 2: Neutro   (tweets informativos de veículos jornalísticos)
CLASS_ID_TO_LABEL = {
    0: "Negativo",
    1: "Positivo",
    2: "Neutro",
}

CLASS_LABEL_TO_ID = {
    "Negativo": 0,
    "Positivo": 1,
    "Neutro": 2,
}

CLASS_DISPLAY_NAMES = ["Negativo", "Positivo", "Neutro"]

SENTIMENT_COLORS = {
    "Positivo": "#10b981",  # Verde esmeralda moderno
    "Negativo": "#f43f5e",  # Rosa/vermelho vibrante
    "Neutro": "#8b5cf6",    # Roxo suave moderno
}

SENTIMENT_ICONS = {
    "Positivo": "",
    "Negativo": "",
    "Neutro": "",
}

# ============================================================
# CONDIÇÕES EXPERIMENTAIS DO TCC II
# ============================================================
CONDICOES = [
    "Texto Bruto",
    "Limpeza Básica",
    "Pipeline Completo",
    "Sem Emoticons",
]

CONDICAO_COLUNAS = {
    "Texto Bruto": "tweet_text",
    "Limpeza Básica": "_base",
    "Pipeline Completo": "text_limpo",
    "Sem Emoticons": "texto_sem_emoticons",
}

CONDICAO_DESCRICOES = {
    "Texto Bruto": "Texto original, sem aplicação de qualquer etapa de pré-processamento textual.",
    "Limpeza Básica": "Conversão para minúsculas, remoção de URLs e menções (@), preservação do conteúdo de hashtags, redução de alongamentos e remoção de caracteres não alfanuméricos.",
    "Pipeline Completo": "Limpeza básica combinada com remoção de stopwords (preservando negações) e stemming (RSLPStemmer).",
    "Sem Emoticons": "Texto original após a remoção dos marcadores emocionais explícitos (:), :(, :D, etc.) via expressão regular diagnóstica.",
}

# ============================================================
# HIPERPARÂMETROS DOS MODELOS (Conforme Célula 11 do Notebook)
# ============================================================
MAX_FEATURES = 50000

TFIDF_CONFIG = {
    "max_features": MAX_FEATURES,
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
}

LOGISTIC_REGRESSION_CONFIG = {
    "max_iter": 1000,
    "random_state": SEED,
    "n_jobs": -1,
}

SVM_CONFIG = {
    "random_state": SEED,
}

# ============================================================
# EXEMPLOS DEMONSTRATIVOS PARA A INTERFACE
# ============================================================
DEMO_EXAMPLES = [
    {
        "titulo": "Mensagem claramente positiva",
        "texto": "Adorei o atendimento e o produto chegou super rápido, muito feliz! :)",
        "tipo": "Positivo",
    },
    {
        "titulo": "Mensagem claramente negativa",
        "texto": "Péssimo serviço, fiquei horas esperando e ninguém resolveu meu problema :(",
        "tipo": "Negativo",
    },
    {
        "titulo": "Mensagem neutra / jornalística",
        "texto": "A Folha de S.Paulo informa que a taxa básica de juros foi mantida em reunião nesta quarta-feira.",
        "tipo": "Neutro",
    },
    {
        "titulo": "Mensagem com emoticons marcantes",
        "texto": "Cheguei muitoooo cansado em casa mas o dia hoje valeu a pena :D vamos em frente!",
        "tipo": "Com Emoticon",
    },
    {
        "titulo": "Mensagem ambígua / irônica",
        "texto": "Nossa, que maravilha de transporte público... mais duas horas parado na chuva, simplesmente adoro :)",
        "tipo": "Ambíguo / Irônico",
    },
]
