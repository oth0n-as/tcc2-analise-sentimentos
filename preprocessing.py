"""Módulo de pré-processamento textual do TCC II.

Reproduz estritamente as funções, expressões regulares e regras de
tratamento textual estabelecidas no notebook original de experimentos.
"""

import html
import re
from typing import Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize

# ============================================================
# GARANTIA DE RECURSOS DO NLTK
# ============================================================
_NLTK_RESOURCES_DOWNLOADED = False


def ensure_nltk_resources() -> None:
    """Garante o download silencioso dos recursos do NLTK necessários."""
    global _NLTK_RESOURCES_DOWNLOADED
    if not _NLTK_RESOURCES_DOWNLOADED:
        for resource in ["stopwords", "punkt", "punkt_tab", "rslp"]:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
        _NLTK_RESOURCES_DOWNLOADED = True


# ============================================================
# REGRAS E STOPWORDS (Conforme Célula 7 do Notebook)
# ============================================================
ensure_nltk_resources()

NEGACOES_PT = {
    "não",
    "nao",
    "nunca",
    "jamais",
    "tampouco",
    "nem",
}

try:
    STOPWORDS_PT = set(stopwords.words("portuguese")) - NEGACOES_PT
except Exception:
    # Fallback seguro caso os recursos ainda estejam sendo baixados
    STOPWORDS_PT = set()

_stemmer_pt: Optional[RSLPStemmer] = None


def get_stemmer() -> RSLPStemmer:
    """Retorna a instância do stemmer RSLP em singleton."""
    global _stemmer_pt
    if _stemmer_pt is None:
        ensure_nltk_resources()
        _stemmer_pt = RSLPStemmer()
    return _stemmer_pt


# ============================================================
# EXPRESSÕES REGULARES DO NOTEBOOK
# ============================================================
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENCAO = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#(\w+)")
RE_REP = re.compile(r"(.)\1{2,}")
RE_NAO_ALFA = re.compile(r"[^a-zA-ZáàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ\s]")
RE_ESP = re.compile(r"\s+")

# Marcadores emocionais explícitos (Protegendo links http/https)
RE_EMOTICON = re.compile(
    r"(?<!http)(?<!https)"
    r"(?:[:;=][-^]?[)(/\\|DPdp]+|8[-^]?[)(\\|DPdp]+)"
)


# ============================================================
# FUNÇÕES DE PROCESSAMENTO
# ============================================================
def limpar_texto_pt(texto: str) -> str:
    """Aplica a Limpeza Básica definida no TCC II:
    - Desescapa entidades HTML e converte para minúsculas;
    - Remove URLs e menções (@usuario);
    - Preserva o conteúdo das hashtags (#fato -> fato);
    - Reduz alongamentos de caracteres (ex: 'muitooooo' -> 'muitoo');
    - Remove caracteres não alfanuméricos e pontuações;
    - Normaliza espaços múltiplos.
    """
    if not isinstance(texto, str):
        return ""

    texto = html.unescape(texto).lower()
    texto = RE_URL.sub(" ", texto)
    texto = RE_MENCAO.sub(" ", texto)
    texto = RE_HASHTAG.sub(r" \1 ", texto)
    texto = RE_REP.sub(r"\1\1", texto)
    texto = RE_NAO_ALFA.sub(" ", texto)
    return RE_ESP.sub(" ", texto).strip()


def remover_emoticons(texto: str) -> str:
    """Remove apenas os marcadores emocionais explícitos (emoticons).

    Condição diagnóstica do TCC para investigar a sensibilidade dos modelos
    à ausência de pistas emocionais diretas.
    """
    if not isinstance(texto, str):
        return ""

    texto = RE_EMOTICON.sub(" ", texto)
    return RE_ESP.sub(" ", texto).strip()


def aplicar_pipeline_pt(
    base_limpa: str,
    usar_stopwords: bool = True,
    usar_stemming: bool = True,
) -> str:
    """Aplica tokenização, remoção de stopwords (com tamanho > 2) e stemming.

    Recebe o texto já tratado pela limpeza básica.
    """
    ensure_nltk_resources()
    tokens = word_tokenize(base_limpa, language="portuguese")

    if usar_stopwords:
        tokens = [
            t for t in tokens
            if t not in STOPWORDS_PT and len(t) > 2
        ]

    if usar_stemming:
        stemmer = get_stemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


def preprocessar_por_condicao(texto: str, condicao: str) -> str:
    """Aplica a transformação textual correspondente a uma das quatro condições.

    Opções:
    - 'Texto Bruto': Retorna o texto original sem modificação.
    - 'Limpeza Básica': Normalização, remoção de URLs/menções e pontuações.
    - 'Pipeline Completo': Limpeza básica + remoção de stopwords + stemming.
    - 'Sem Emoticons': Texto original com emoticons removidos.
    """
    if not isinstance(texto, str):
        return ""

    if condicao == "Texto Bruto":
        return texto.strip()
    elif condicao == "Limpeza Básica":
        return limpar_texto_pt(texto)
    elif condicao == "Pipeline Completo":
        base = limpar_texto_pt(texto)
        return aplicar_pipeline_pt(base, usar_stopwords=True, usar_stemming=True)
    elif condicao == "Sem Emoticons":
        return remover_emoticons(texto)
    else:
        raise ValueError(f"Condição desconhecida: '{condicao}'")
