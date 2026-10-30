"""Aplicação Web Streamlit — Protótipo Acadêmico do TCC II.

Tema: Avaliação de Algoritmos e Técnicas de NLP para Análise de Sentimentos
      em Textos Informais no Contexto do Português Brasileiro.

Desenvolvido para demonstração à banca e orientador, permitindo inferência real,
diagnóstico didático da representação vetorial e comparação interativa entre as
quatro condições experimentais da pesquisa.
"""

import html
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    CLASS_DISPLAY_NAMES,
    CONDICAO_DESCRICOES,
    CONDICOES,
    DEMO_EXAMPLES,
    MODELS_DIR,
    RESULTS_DIR,
    SENTIMENT_COLORS,
)
from inference import (
    carregar_modelo,
    classificar_texto,
    comparar_todas_as_condicoes,
    verificar_modelos_disponiveis,
)

# ============================================================
# CONFIGURAÇÃO GERAL DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="TCC II — Análise de Sentimentos em Português Brasileiro",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS CUSTOMIZADO (PALETA: #0B0B12, #171426, #242235, #8B5CF6, #F5F3FF, #A9A5BC)
# ESTILO ACADÊMICO LIMPO E SÓBRIO (SEM EMOJIS)
# ============================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #0B0B12 !important;
        color: #F5F3FF !important;
        font-size: 15px !important;
    }

    /* ------------------------------------------------------------
       FAIXA SUPERIOR (INTEGRADA AO TEMA ESCURO)
       ------------------------------------------------------------ */
    header[data-testid="stHeader"], .stAppHeader {
        background-color: #0B0B12 !important;
        border-bottom: 1px solid #1E1B33 !important;
    }

    header[data-testid="stHeader"] svg,
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] span {
        color: #A9A5BC !important;
        fill: #A9A5BC !important;
    }

    /* ------------------------------------------------------------
       BARRA LATERAL (CONTRASTE, LARGURA CONFORTÁVEL E SEM QUEBRAS)
       ------------------------------------------------------------ */
    section[data-testid="stSidebar"] {
        background-color: #12101F !important;
        border-right: 1px solid #242235 !important;
        min-width: 305px !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #F5F3FF !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        margin-top: 10px !important;
        margin-bottom: 12px !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #A9A5BC !important;
        font-size: 0.92rem !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #242235 !important;
        margin: 16px 0 !important;
    }

    /* Oculta as bolinhas do radio button */
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    /* Botões/pills modernos de navegação */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin-bottom: 6px !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(139, 92, 246, 0.12) !important;
        border-color: rgba(139, 92, 246, 0.3) !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover p {
        color: #F5F3FF !important;
    }

    /* Item de navegação selecionado */
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: rgba(139, 92, 246, 0.22) !important;
        border: 1px solid rgba(139, 92, 246, 0.5) !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        color: #A9A5BC !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
        white-space: nowrap !important;
    }

    /* ------------------------------------------------------------
       BANNER INICIAL COMPACTO E ELEGANTE
       ------------------------------------------------------------ */
    .academic-banner {
        background: linear-gradient(135deg, rgba(23, 20, 38, 0.95) 0%, rgba(36, 34, 53, 0.85) 100%);
        border: 1px solid rgba(139, 92, 246, 0.32);
        border-radius: 10px;
        padding: 16px 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }

    .banner-tag {
        color: #A78BFA;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
        display: block;
    }

    .banner-title {
        color: #F5F3FF;
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.25;
        margin: 0 0 4px 0;
    }

    .banner-desc {
        color: #A9A5BC;
        font-size: 0.92rem;
        margin: 0;
        line-height: 1.4;
    }

    /* Título de seção */
    .section-title {
        color: #F5F3FF;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0 0 16px 0;
        letter-spacing: -0.01em;
    }

    /* ------------------------------------------------------------
       CAMPOS DE ENTRADA (#242235) COM FONTES LEGÍVEIS (15-16px)
       ------------------------------------------------------------ */
    textarea, input[type="text"], input {
        background-color: #242235 !important;
        color: #F5F3FF !important;
        border: 1px solid #36324D !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.96rem !important;
    }

    textarea:focus, input[type="text"]:focus, input:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 1px #8B5CF6 !important;
    }

    textarea::placeholder, input::placeholder {
        color: #A9A5BC !important;
        opacity: 0.75 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #242235 !important;
        border: 1px solid #36324D !important;
        border-radius: 8px !important;
        color: #F5F3FF !important;
        font-size: 0.95rem !important;
    }

    div[data-baseweb="select"] > div:focus-within {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 1px #8B5CF6 !important;
    }

    div[data-baseweb="select"] * {
        color: #F5F3FF !important;
        font-size: 0.95rem !important;
    }

    div[data-baseweb="popover"], ul[role="listbox"] {
        background-color: #171426 !important;
        border: 1px solid #36324D !important;
        border-radius: 8px !important;
    }

    li[role="option"] {
        background-color: #171426 !important;
        color: #F5F3FF !important;
        font-size: 0.92rem !important;
    }

    li[role="option"]:hover, li[aria-selected="true"] {
        background-color: rgba(139, 92, 246, 0.25) !important;
        color: #FFFFFF !important;
    }

    /* Rótulos dos campos */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextArea"] label {
        color: #F5F3FF !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 6px !important;
    }

    /* ------------------------------------------------------------
       BOTÃO PRINCIPAL (#8B5CF6)
       ------------------------------------------------------------ */
    div.stButton > button:first-child {
        background-color: #8B5CF6 !important;
        color: #FFFFFF !important;
        border: 1px solid #8B5CF6 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        padding: 11px 26px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3) !important;
    }

    div.stButton > button:first-child:hover {
        background-color: #7C3AED !important;
        border-color: #A78BFA !important;
        box-shadow: 0 4px 14px rgba(139, 92, 246, 0.45) !important;
    }

    /* ------------------------------------------------------------
       CARDS E CAIXAS DE TEXTO (#171426)
       ------------------------------------------------------------ */
    .metric-card {
        background-color: #171426;
        border: 1px solid #2D2A42;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }

    .metric-card:hover {
        border-color: rgba(139, 92, 246, 0.35);
    }

    .text-box {
        background-color: #242235;
        border: 1px solid #36324D;
        border-radius: 8px;
        padding: 12px 14px;
        font-family: monospace;
        font-size: 0.92rem;
        color: #F5F3FF;
        word-break: break-word;
        white-space: pre-wrap;
    }

    /* Badges de sentimento sem emojis */
    .badge-positivo {
        display: inline-flex;
        align-items: center;
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.02em;
    }

    .badge-negativo {
        display: inline-flex;
        align-items: center;
        background-color: rgba(244, 63, 94, 0.15);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.35);
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.02em;
    }

    .badge-neutro {
        display: inline-flex;
        align-items: center;
        background-color: rgba(139, 92, 246, 0.15);
        color: #C4B5FD;
        border: 1px solid rgba(139, 92, 246, 0.35);
        padding: 6px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.02em;
    }

    /* Alertas sóbrios sem emojis */
    .divergence-alert {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.12) 0%, rgba(161, 98, 7, 0.2) 100%);
        border: 1px solid rgba(234, 179, 8, 0.4);
        border-radius: 8px;
        padding: 14px 18px;
        color: #FDE047;
        font-size: 0.94rem;
        margin: 12px 0 20px 0;
        line-height: 1.5;
    }

    .convergence-alert {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(14, 116, 144, 0.15) 100%);
        border: 1px solid rgba(6, 182, 212, 0.35);
        border-radius: 8px;
        padding: 14px 18px;
        color: #67E8F9;
        font-size: 0.94rem;
        margin: 12px 0 20px 0;
        line-height: 1.5;
    }

    .info-condition-box {
        background-color: #171426;
        border: 1px solid #2D2A42;
        border-radius: 8px;
        padding: 12px 14px;
        color: #A9A5BC;
        font-size: 0.92rem;
        line-height: 1.5;
        margin-top: 6px;
    }

    [data-testid="stCaptionContainer"] p {
        font-size: 0.88rem !important;
        color: #A9A5BC !important;
    }

    div[data-testid="stDataFrame"] {
        background-color: #171426 !important;
        border: 1px solid #2D2A42 !important;
        border-radius: 8px !important;
    }

    /* Expander didático */
    div[data-testid="stExpander"] {
        background-color: #171426 !important;
        border: 1px solid #2D2A42 !important;
        border-radius: 8px !important;
        margin-top: 14px !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# CACHE DE DADOS E MODELOS
# ============================================================
@st.cache_resource
def carregar_modelo_cached(nome_modelo: str, condicao: str):
    """Carrega o pipeline scikit-learn via cache persistente do Streamlit."""
    return carregar_modelo(nome_modelo, condicao)


@st.cache_data
def carregar_resultados_consolidados():
    """Carrega os resultados experimentais consolidados oficiais do TCC II."""
    caminho_definitivo = RESULTS_DIR / "resultados_recuperados_TCC2" / "resultados_gerais_definitivos_tcc2.csv"
    if caminho_definitivo.exists():
        return pd.read_csv(caminho_definitivo)
    caminho_provisorio = RESULTS_DIR / "resultados_recuperados_TCC2" / "resultados_gerais_recuperados_provisorios.csv"
    if caminho_provisorio.exists():
        return pd.read_csv(caminho_provisorio)
    local_path = RESULTS_DIR / "metricas_treinamento_local.csv"
    if local_path.exists():
        return pd.read_csv(local_path)
    return None


@st.cache_data
def carregar_metricas_locais():
    """Carrega as métricas locais de treinamento dos modelos do protótipo interativo."""
    local_path = RESULTS_DIR / "metricas_treinamento_local.csv"
    if local_path.exists():
        return pd.read_csv(local_path)
    return None


@st.cache_data
def carregar_erros_bert():
    """Carrega o resumo de erros do BERTimbau."""
    caminho = RESULTS_DIR / "resultados_recuperados_TCC2" / "resumo_erros_bert_recuperados.csv"
    if caminho.exists():
        return pd.read_csv(caminho)
    return None


# ============================================================
# CABEÇALHO ACADÊMICO COMPACTO
# ============================================================
def render_header():
    st.markdown(
        """
        <div class="academic-banner">
            <span class="banner-tag">TCC II · CIÊNCIA DA COMPUTAÇÃO</span>
            <h1 class="banner-title">Análise de Sentimentos em Português Brasileiro</h1>
            <p class="banner-desc">
                Investigue como o pré-processamento influencia as previsões de modelos de classificação.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BARRA LATERAL (NAVEGAÇÃO SÓBRIA, SEM EMOJIS)
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("### Navegação")
        aba_selecionada = st.radio(
            "Módulos:",
            [
                "Análise Individual",
                "Comparar Condições",
                "Resultados do TCC II",
                "Sobre a Pesquisa",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### Configuração do Modelo")

        modelos_disp = verificar_modelos_disponiveis()
        opcoes_modelos = list(modelos_disp.keys()) if modelos_disp else ["Regressão Logística"]

        modelo_escolhido = st.selectbox(
            "Classificador:",
            options=opcoes_modelos,
            index=0,
            help="A Regressão Logística é o modelo de referência principal do MVP, conforme o notebook do TCC II.",
        )

        st.markdown("---")
        st.markdown("### Status dos Modelos")
        total_modelos = sum(len(conds) for conds in modelos_disp.values())
        if total_modelos >= 4:
            st.markdown(
                f"""
                <div class="metric-card" style="padding: 12px 14px; margin-bottom: 8px;">
                    <div style="font-size: 0.84rem; color: #A9A5BC; font-weight: 500; margin-bottom: 4px;">
                        Configurações disponíveis
                    </div>
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span style="font-size: 1.2rem; font-weight: 700; color: #F5F3FF;">
                            {total_modelos}/8
                        </span>
                        <span style="display: inline-flex; align-items: center; gap: 5px; font-size: 0.8rem; color: #34D399; font-weight: 600; background: rgba(16, 185, 129, 0.12); padding: 3px 8px; border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.25);">
                            <span style="display: inline-block; width: 6px; height: 6px; background-color: #10B981; border-radius: 50%;"></span>
                            Prontas
                        </span>
                    </div>
                    <div style="font-size: 0.75rem; color: #7E7A94; margin-top: 4px;">
                        2 classificadores × 4 condições
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Artefatos não encontrados em `models/`")

        st.markdown("---")
        st.caption("Protótipo acadêmico desenvolvido para demonstração ao orientador e à banca avaliadora.")

    return aba_selecionada, modelo_escolhido


# ============================================================
# RENDERIZADOR DE BADGE DE SENTIMENTO (SEM EMOJIS)
# ============================================================
def get_sentiment_badge_html(classe: str) -> str:
    classe_clean = classe.strip().capitalize()
    if classe_clean == "Positivo":
        return '<span class="badge-positivo">Positivo</span>'
    elif classe_clean == "Negativo":
        return '<span class="badge-negativo">Negativo</span>'
    elif classe_clean == "Neutro":
        return '<span class="badge-neutro">Neutro</span>'
    elif classe_clean == "Indeterminado":
        return '<span class="badge-neutro" style="background: rgba(148, 163, 184, 0.15); color: #CBD5E1; border: 1px solid rgba(148, 163, 184, 0.3);">Indeterminado</span>'
    return f"<span>{html.escape(classe)}</span>"


# ============================================================
# 1. PÁGINA: ANÁLISE INDIVIDUAL
# ============================================================
def pagina_analise_individual(modelo_escolhido: str):
    st.markdown('<div class="section-title">Análise individual</div>', unsafe_allow_html=True)

    # 1. Experimente um exemplo
    opcoes_ex = ["Selecionar uma frase de exemplo..."] + [f"{ex['titulo']} ({ex['tipo']})" for ex in DEMO_EXAMPLES]
    ex_selecionado = st.selectbox(
        "Experimente um exemplo",
        options=opcoes_ex,
        index=0,
    )

    texto_inicial = ""
    if ex_selecionado != "Selecionar uma frase de exemplo...":
        idx = opcoes_ex.index(ex_selecionado) - 1
        texto_inicial = DEMO_EXAMPLES[idx]["texto"]

    # 2. Formulário principal (Texto para análise + Condição experimental lado a lado)
    col_texto, col_cond = st.columns([2.2, 1])

    with col_texto:
        texto_usuario = st.text_area(
            "Texto para análise",
            value=texto_inicial,
            placeholder="Digite uma frase, tweet ou comentário...",
            height=125,
        )

    with col_cond:
        condicao_escolhida = st.selectbox(
            "Condição experimental",
            options=CONDICOES,
            index=0,
        )
        st.markdown(
            f'<div class="info-condition-box">{CONDICAO_DESCRICOES[condicao_escolhida]}</div>',
            unsafe_allow_html=True,
        )

    # 3. Botão de ação principal
    botao_analisar = st.button("Analisar sentimento", use_container_width=True)
    st.markdown(
        '<div style="color: #A9A5BC; font-size: 0.85rem; margin-top: 4px; margin-bottom: 16px;">'
        'A classificação é realizada por inferência real utilizando o pipeline treinado da condição selecionada.'
        '</div>',
        unsafe_allow_html=True,
    )

    if botao_analisar:
        if not texto_usuario.strip():
            st.warning("Por favor, digite uma mensagem para realizar a análise.")
            return

        with st.spinner("Realizando inferência no pipeline treinado..."):
            resultado = classificar_texto(
                texto=texto_usuario,
                condicao=condicao_escolhida,
                nome_modelo=modelo_escolhido,
            )

        if not resultado.get("sucesso"):
            st.error(f"Erro: {resultado.get('erro')}")
            return

        # Apresentação dos Resultados
        st.markdown("---")
        st.markdown('<div class="section-title">Resultado da Classificação</div>', unsafe_allow_html=True)

        col_res1, col_res2 = st.columns([1.2, 1.8])

        with col_res1:
            texto_orig_escaped = html.escape(resultado.get('texto_original', ''))
            texto_proc_val = resultado.get('texto_processado', '')
            texto_proc_escaped = html.escape(texto_proc_val) if texto_proc_val else ''
            modelo_escaped = html.escape(resultado.get('modelo', ''))
            condicao_escaped = html.escape(resultado.get('condicao', ''))
            classe_predita_val = resultado.get('classe_predita', 'Indeterminado')

            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="color: #A9A5BC; font-size: 0.85rem; text-transform: uppercase; font-weight: 700; margin-bottom: 8px;">
                        Sentimento Previsto
                    </div>
                    <div>
                        {get_sentiment_badge_html(classe_predita_val)}
                    </div>
                    <div style="margin-top: 14px; font-size: 0.9rem; color: #A9A5BC; line-height: 1.5;">
                        <b style="color: #F5F3FF;">Modelo:</b> {modelo_escaped}<br>
                        <b style="color: #F5F3FF;">Condição:</b> {condicao_escaped}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Caso o texto tenha ficado vazio após o pré-processamento
            if resultado.get("texto_vazio"):
                st.markdown(
                    """
                    <div class="convergence-alert" style="border-left: 4px solid #F59E0B; background: rgba(245, 158, 11, 0.08); margin-top: 10px;">
                        <b style="color: #FBBF24;">Atenção: Texto Vazio após Pré-processamento</b><br>
                        Todos os termos da entrada foram filtrados pelas regras da condição selecionada (ex.: remoção de stopwords ou emoticons).
                        Nenhum token restou para alimentar o modelo estatístico.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Probabilidades estimadas se disponíveis
            if resultado.get("probabilidades"):
                st.markdown("<b style='color: #F5F3FF; font-size: 0.92rem;'>Distribuição de Probabilidades</b>", unsafe_allow_html=True)
                probs = resultado["probabilidades"]
                df_probs = pd.DataFrame({
                    "Classe": list(probs.keys()),
                    "Probabilidade": list(probs.values()),
                })
                fig_prob = px.bar(
                    df_probs,
                    x="Probabilidade",
                    y="Classe",
                    orientation="h",
                    color="Classe",
                    color_discrete_map=SENTIMENT_COLORS,
                    range_x=[0, 1],
                    text=df_probs["Probabilidade"].apply(lambda p: f"{p*100:.1f}%"),
                )
                fig_prob.update_layout(
                    height=180,
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F5F3FF", size=13),
                    showlegend=False,
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False, tickfont=dict(color="#F5F3FF", size=13)),
                )
                st.plotly_chart(fig_prob, use_container_width=True)

        with col_res2:
            st.markdown("<b style='color: #F5F3FF; font-size: 0.92rem;'>Comparação Textual (Original vs. Pré-processado)</b>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="margin-bottom: 12px; margin-top: 6px;">
                    <span style="font-size: 0.82rem; color: #A9A5BC; font-weight: 700; letter-spacing: 0.05em;">TEXTO ORIGINAL (ENTRADA LIVRE):</span>
                    <div class="text-box">{texto_orig_escaped}</div>
                </div>
                <div>
                    <span style="font-size: 0.82rem; color: #A9A5BC; font-weight: 700; letter-spacing: 0.05em;">TEXTO APÓS PRÉ-PROCESSAMENTO ({condicao_escaped.upper()}):</span>
                    <div class="text-box">{texto_proc_escaped if texto_proc_escaped else '<i style="color: #A9A5BC;">[Texto vazio após a remoção de elementos]</i>'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            diff_chars = len(resultado.get('texto_original', '')) - len(resultado.get('texto_processado', ''))
            st.caption(f"Variação de caracteres: {len(resultado.get('texto_original', ''))} → {len(resultado.get('texto_processado', ''))} ({diff_chars} caracteres removidos ou transformados).")

        # ------------------------------------------------------------
        # ÁREA EXPANSÍVEL: ENTENDA ESTA PREVISÃO
        # ------------------------------------------------------------
        diagnostico = resultado.get("diagnostico")
        if diagnostico:
            with st.expander("Entenda esta previsão (Diagnóstico da Representação Textual)"):
                st.markdown(
                    """
                    <div style="font-size: 0.92rem; color: #A9A5BC; margin-bottom: 12px; line-height: 1.55;">
                        Este painel apresenta um <b>diagnóstico objetivo da representação vetorial (TF-IDF)</b>,
                        mostrando quais termos da frase foram capturados pelo vocabulário de treino, seus valores TF-IDF no texto,
                        e a contribuição efetiva para a pontuação da classe prevista.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if diagnostico.get("aviso"):
                    st.info(diagnostico["aviso"])

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown("**Tokens reconhecidos pelo modelo (no vocabulário):**")
                    rec = diagnostico.get("tokens_reconhecidos", [])
                    if rec:
                        st.markdown(
                            " ".join([
                                f"<code style='background: #242235; color: #34D399; padding: 4px 8px; border-radius: 4px; font-size: 0.9rem; margin: 2px;'>{html.escape(t)}</code>"
                                for t in rec
                            ]),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("Nenhum token da frase foi reconhecido no vocabulário treinado.")

                with col_d2:
                    st.markdown("**Tokens fora do vocabulário (Out-of-Vocabulary / ignorados):**")
                    oov = diagnostico.get("tokens_fora_vocabulario", [])
                    if oov:
                        st.markdown(
                            " ".join([
                                f"<code style='background: #242235; color: #FB7185; padding: 4px 8px; border-radius: 4px; font-size: 0.9rem; margin: 2px;'>{html.escape(t)}</code>"
                                for t in oov
                            ]),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("Todos os termos gerados foram encontrados no vocabulário.")

                pesos = diagnostico.get("pesos_features", [])
                if pesos:
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    st.markdown(f"**Contribuição efetiva dos termos para a classe prevista (*{html.escape(classe_predita_val)}*):**")
                    df_pesos = pd.DataFrame(pesos)
                    colunas_map = {
                        "token": "Termo / N-grama",
                        "tipo": "Tipo",
                        "tfidf": "Valor TF-IDF",
                        "coeficiente": "Coeficiente (W)",
                        "contribuicao": "Contribuição Efetiva (TF-IDF × W)",
                        "influencia": "Impacto na Decisão",
                    }
                    cols_presentes = [c for c in colunas_map.keys() if c in df_pesos.columns]
                    df_pesos_view = df_pesos[cols_presentes].rename(columns=colunas_map)

                    format_dict = {}
                    if "Valor TF-IDF" in df_pesos_view.columns:
                        format_dict["Valor TF-IDF"] = "{:.4f}"
                    if "Coeficiente (W)" in df_pesos_view.columns:
                        format_dict["Coeficiente (W)"] = "{:.4f}"
                    if "Contribuição Efetiva (TF-IDF × W)" in df_pesos_view.columns:
                        format_dict["Contribuição Efetiva (TF-IDF × W)"] = "{:+.4f}"

                    st.dataframe(
                        df_pesos_view.style.format(format_dict),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.markdown(
                    """
                    <div style="background-color: #171426; border-left: 3px solid #8B5CF6; padding: 10px 14px; border-radius: 6px; font-size: 0.86rem; color: #A9A5BC; margin-top: 12px; line-height: 1.5;">
                        <b>Nota Científica sobre a Contribuição Efetiva:</b> Em modelos lineares com TF-IDF, a contribuição real de uma característica
                        para o score da classe é calculada pela multiplicação do peso TF-IDF do termo na entrada pelo seu coeficiente linear (<i>Contribuição = TF-IDF × W</i>).
                        Coeficientes positivos elevam a probabilidade da classe; termos não presentes no texto têm TF-IDF zero e, portanto, não contribuem para a decisão.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:
        # Três cards informativos sóbrios exibidos no estado inicial
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown(
                """
                <div class="metric-card" style="padding: 16px 18px; margin-bottom: 0;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #8B5CF6; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 4px;">Escopo</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #F5F3FF; margin-bottom: 2px;">2 Modelos</div>
                    <div style="font-size: 0.85rem; color: #A9A5BC;">Regressão Logística e SVM</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_c2:
            st.markdown(
                """
                <div class="metric-card" style="padding: 16px 18px; margin-bottom: 0;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #8B5CF6; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 4px;">Metodologia</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #F5F3FF; margin-bottom: 2px;">4 Condições</div>
                    <div style="font-size: 0.85rem; color: #A9A5BC;">Pré-processamentos avaliados</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_c3:
            st.markdown(
                """
                <div class="metric-card" style="padding: 16px 18px; margin-bottom: 0;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #8B5CF6; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 4px;">Pesquisa</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #F5F3FF; margin-bottom: 2px;">TCC II</div>
                    <div style="font-size: 0.85rem; color: #A9A5BC;">Protótipo acadêmico</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# 2. PÁGINA: COMPARAR CONDIÇÕES
# ============================================================
def pagina_comparar_condicoes(modelo_escolhido: str):
    st.markdown('<div class="section-title">Comparação entre Condições Experimentais</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="color: #A9A5BC; font-size: 0.95rem; margin-bottom: 16px; line-height: 1.5;">
            Esta funcionalidade responde diretamente à <b>questão central do TCC II</b>:
            como a mesma mensagem é classificada quando submetida a diferentes intensidades de pré-processamento,
            cada qual processada por seu <b>respectivo pipeline treinado</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    opcoes_ex = ["Selecionar uma frase de exemplo..."] + [f"{ex['titulo']} ({ex['tipo']})" for ex in DEMO_EXAMPLES]
    ex_selecionado = st.selectbox(
        "Experimente um exemplo para comparação",
        options=opcoes_ex,
        index=0,
    )

    texto_inicial = ""
    if ex_selecionado != "Selecionar uma frase de exemplo...":
        idx = opcoes_ex.index(ex_selecionado) - 1
        texto_inicial = DEMO_EXAMPLES[idx]["texto"]

    texto_usuario = st.text_area(
        "Texto para teste comparativo simultâneo",
        value=texto_inicial,
        placeholder="Digite uma frase com emoticons, ironia ou termos informais...",
        height=110,
    )

    botao_comparar = st.button("Comparar as 4 condições simultaneamente", use_container_width=True)

    if botao_comparar:
        if not texto_usuario.strip():
            st.warning("Por favor, digite um texto para efetuar a comparação.")
            return

        with st.spinner("Executando inferência real nos quatro pipelines..."):
            comparacao = comparar_todas_as_condicoes(texto_usuario, nome_modelo=modelo_escolhido)

        st.markdown("---")

        # Alertas sóbrios sem emojis
        n_validas = len(comparacao.get("classes_preditas", []))

        if n_validas < 2:
            st.markdown(
                f"""
                <div class="convergence-alert" style="border-left: 4px solid #F59E0B; background: rgba(245, 158, 11, 0.08);">
                    <b>Comparação inconclusiva (número insuficiente de previsões válidas)</b><br>
                    Apenas {n_validas} de 4 condições produziu previsão válida (as demais condições não produziram previsões válidas, seja por termos esvaziados no pré-processamento ou por falha na inferência).
                    Não há previsões válidas suficientes para atestar concordância ou divergência entre representações.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif comparacao["divergencia_detectada"]:
            st.markdown(
                f"""
                <div class="divergence-alert">
                    <b>Divergência de classificação detectada ({n_validas} condições válidas avaliadas)</b><br>
                    O modelo alterou a previsão de sentimento dependendo da técnica de pré-processamento textual aplicada.
                    Isso evidencia experimentalmente a sensibilidade do classificador à remoção de marcadores linguísticos,
                    sem determinar que uma representação seja universalmente superior.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            primeira_classe = html.escape(comparacao["classes_preditas"][0]) if comparacao["classes_preditas"] else ""
            if n_validas == 4:
                st.markdown(
                    f"""
                    <div class="convergence-alert">
                        <b>Convergência entre as quatro condições ({primeira_classe})</b><br>
                        O modelo manteve a mesma previsão de sentimento sob todas as quatro representações testadas.
                        Isso reflete estabilidade em relação aos termos residuais para esta frase,
                        <b>embora não garanta, isoladamente, que a previsão seja semanticamente correta</b>
                        (especialmente em mensagens com ironia, ambiguidade ou termos raros).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="convergence-alert" style="border-left: 4px solid #8B5CF6;">
                        <b>Concordância parcial entre as condições avaliadas ({primeira_classe})</b><br>
                        O modelo manteve a mesma previsão de sentimento nas {n_validas} condições avaliadas com sucesso.
                        As demais condições não produziram previsões válidas.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Tabela Comparativa
        st.markdown("#### Tabela Comparativa de Pré-processamento e Sentimento")

        dados_tabela = []
        resumo_classes = []
        for r in comparacao["resultados"]:
            cond = r.get("condicao", "Desconhecida")
            if not r.get("sucesso"):
                dados_tabela.append({
                    "Condição Experimental": cond,
                    "Texto Processado": "(erro na inferência)",
                    "Sentimento Previsto": "Erro",
                    "Confiança": "N/A",
                })
            elif r.get("texto_vazio"):
                dados_tabela.append({
                    "Condição Experimental": cond,
                    "Texto Processado": "(vazio após filtros)",
                    "Sentimento Previsto": "Indeterminado",
                    "Confiança": "N/A",
                })
            else:
                classe_p = r.get("classe_predita", "Indeterminado")
                resumo_classes.append((cond, classe_p))
                probs = r.get("probabilidades") or {}
                conf_str = f"{probs.get(classe_p, 0)*100:.1f}%" if probs and classe_p in probs else "N/A"
                dados_tabela.append({
                    "Condição Experimental": cond,
                    "Texto Processado": r.get("texto_processado", ""),
                    "Sentimento Previsto": classe_p,
                    "Confiança": conf_str,
                })

        df_comp = pd.DataFrame(dados_tabela)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # Detalhamento em cards lado a lado
        st.markdown("#### Detalhamento por Condição")
        cols = st.columns(4)

        for i, r in enumerate(comparacao["resultados"]):
            with cols[i]:
                cond_escaped = html.escape(r.get('condicao', ''))
                txt_proc = r.get('texto_processado', '')
                txt_proc_escaped = html.escape(txt_proc) if txt_proc else ''

                if not r.get("sucesso"):
                    badge_html = '<span class="badge-neutro" style="background: rgba(239, 68, 68, 0.2); color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.4);">Erro</span>'
                    erro_msg = html.escape(r.get("erro", "Falha na inferência"))
                    status_content = f'<div class="text-box" style="font-size: 0.85rem; min-height: 70px; color: #F87171;">{erro_msg}</div>'
                elif r.get("texto_vazio"):
                    badge_html = '<span class="badge-neutro" style="background: rgba(148, 163, 184, 0.15); color: #CBD5E1; border: 1px solid rgba(148, 163, 184, 0.3);">Indeterminado</span>'
                    status_content = '<div class="text-box" style="font-size: 0.85rem; min-height: 70px; color: #A9A5BC;"><i>[Texto vazio após a remoção de elementos]</i></div>'
                else:
                    badge_html = get_sentiment_badge_html(r.get('classe_predita', ''))
                    status_content = f'<div class="text-box" style="font-size: 0.85rem; min-height: 70px;">{txt_proc_escaped if txt_proc_escaped else "<span style=\'color:#A9A5BC;\'>[vazio]</span>"}</div>'

                st.markdown(
                    f"""
                    <div class="metric-card" style="min-height: 235px;">
                        <div style="font-size: 0.9rem; font-weight: 700; color: #8B5CF6; margin-bottom: 6px;">
                            {cond_escaped}
                        </div>
                        <div style="margin-bottom: 12px;">
                            {badge_html}
                        </div>
                        <div style="font-size: 0.8rem; color: #A9A5BC; margin-bottom: 4px; font-weight: 700; letter-spacing: 0.04em;">
                            TEXTO APÓS PIPELINE:
                        </div>
                        {status_content}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Demonstração guiada sem emojis
        if n_validas < 2:
            texto_interpretacao = (
                f"A sentença fornecida produziu menos de duas previsões válidas ({n_validas} de 4). "
                "Portanto, a comparação entre representações é inconclusiva para esta entrada específica."
            )
        elif comparacao["divergencia_detectada"]:
            detalhe_resumo = " · ".join([f"{html.escape(c)}: <b>{html.escape(p)}</b>" for c, p in resumo_classes])
            texto_interpretacao = (
                f"Nesta demonstração, a classificação variou entre as representações com texto válido ({detalhe_resumo}). "
                "Isso evidencia empiricamente a sensibilidade do modelo à perda de elementos textuais "
                "(como marcadores expressivos ou desinências gramaticais), demonstrando que o pré-processamento "
                "altera a representação matemática do texto, sem determinar, isoladamente, qual previsão é a mais correta."
            )
        elif n_validas < 4:
            classe_unica = resumo_classes[0][1] if resumo_classes else "Indeterminada"
            texto_interpretacao = (
                f"Para esta sentença, as {n_validas} representações avaliadas com sucesso concordaram na classe <b>{html.escape(classe_unica)}</b>. "
                "As demais condições não produziram previsões válidas."
            )
        else:
            classe_unica = resumo_classes[0][1] if resumo_classes else "Indeterminada"
            texto_interpretacao = (
                f"Para esta sentença, todas as quatro representações textuais convergiram na classe <b>{html.escape(classe_unica)}</b>. "
                "Isso indica que os termos determinantes para a decisão foram mantidos ao longo dos filtros, "
                "demonstrando estabilidade nesta construção sintática específica."
            )

        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid #8B5CF6; margin-top: 20px;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #F5F3FF; margin-bottom: 6px;">
                    O que essa comparação demonstra?
                </div>
                <div style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.55;">
                    <b>Interpretação:</b> {texto_interpretacao}
                </div>
                <hr style="border-color: #2D2A42; margin: 12px 0;">
                <div style="font-size: 0.86rem; color: #A9A5BC; line-height: 1.55;">
                    <b>Guia de Leitura Metodológica:</b><br>
                    • <b>Texto Bruto vs. Sem Emoticons:</b> Avalia a sensibilidade à remoção dos marcadores emocionais explícitos (<code>:)</code>, <code>:(</code>, etc.).<br>
                    &nbsp;&nbsp;<i>Nota metodológica crucial:</i> Tanto nos modelos lineares com <code>TfidfVectorizer</code> quanto na rede <b>LSTM</b> (que utilizou o <code>Tokenizer</code> padrão do Keras com filtros de pontuação padrão), os símbolos de pontuação e emoticons são descartados por padrão na etapa de tokenização, não formando tokens isolados. Portanto, variações nesses modelos decorrem de ajustes contextuais no vocabulário e na convergência de treinamento, e não da presença literal de tokens de emoticons. Em contraste, no <b>BERTimbau</b> (cujo tokenizer WordPiece fragmenta e processa esses caracteres dentro de seu espaço representacional pré-treinado), a remoção de emoticons produziu um salto drástico na taxa de erro (de 0,3% para 15,7%), fornecendo indícios de dependência de marcadores associados à coleta por <i>distant supervision</i>.<br>
                    • <b>Limpeza Básica:</b> Filtra ruídos externos (links, menções, repetições excessivas) preservando o léxico e a morfologia das palavras.<br>
                    • <b>Pipeline Completo:</b> Elimina stopwords e reduz palavras ao radical (stemming). Em todos os quatro modelos da pesquisa, essa remoção agressiva causou perda consistente de desempenho médio.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 3. PÁGINA: RESULTADOS EXPERIMENTAIS DO TCC II
# ============================================================
def pagina_resultados_experimentais():
    st.markdown('<div class="section-title">Resultados Consolidados dos Experimentos do TCC II</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="color: #A9A5BC; font-size: 0.95rem; margin-bottom: 16px; line-height: 1.5;">
            Esta seção reúne as métricas científicas consolidadas na pesquisa e no artigo do TCC II,
            cobrindo as 16 combinações avaliadas no notebook definitivo (4 modelos × 4 condições),
            além de apresentar separadamente as métricas dos modelos empacotados para esta demonstração interativa.
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_geral = carregar_resultados_consolidados()

    if df_geral is None:
        st.error("Não foram encontrados arquivos de resultados na pasta `results/`.")
        return

    # Destaque da Descoberta Científica Central
    st.markdown(
        """
        <div class="metric-card" style="border-left: 4px solid #EF4444; margin-bottom: 20px;">
            <div style="font-size: 1.05rem; font-weight: 700; color: #F5F3FF; margin-bottom: 6px;">
                Principal Descoberta Científica do TCC II
            </div>
            <div style="font-size: 0.94rem; color: #cbd5e1; line-height: 1.55;">
                Em <b>todos os quatro modelos avaliados</b> (Regressão Logística, SVM, LSTM e BERTimbau),
                o <b>Pipeline Completo</b> (remoção de stopwords e stemming) apresentou desempenho substancialmente inferior
                ao <b>Texto Bruto</b>.
                A redução das palavras aos seus radicais e a eliminação de termos frequentes removeram pistas morfológicas e afetivas vitais,
                evidenciando empiricamente que técnicas agressivas de pré-processamento tendem a prejudicar a classificação de sentimentos em textos informais em português.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cores_condicoes = {
        "Texto Bruto": "#8B5CF6",       # Roxo principal
        "Sem Emoticons": "#06B6D4",     # Ciano
        "Limpeza Básica": "#F59E0B",    # Âmbar
        "Pipeline Completo": "#EF4444", # Vermelho
    }

    # Estruturação em Abas para Máxima Clareza Acadêmica
    tab_oficial, tab_delta, tab_local = st.tabs([
        "Resultados Definitivos do TCC II (16 Experimentos)",
        "Variação do Desempenho (Δ F1 vs. Texto Bruto)",
        "Métricas do Protótipo Local (Modelos Demonstrativos)",
    ])

    # ------------------------------------------------------------
    # ABA 1: RESULTADOS DEFINITIVOS DO ARTIGO / TCC II
    # ------------------------------------------------------------
    with tab_oficial:
        st.markdown(
            """
            <div class="metric-card" style="border-left: 4px solid #8B5CF6; margin-bottom: 18px;">
                <b style="color: #F5F3FF; font-size: 0.92rem;">Nota Metodológica sobre o Escopo Experimental Oficial:</b>
                <div style="color: #A9A5BC; font-size: 0.9rem; line-height: 1.5; margin-top: 4px;">
                    Os classificadores de Regressão Logística, SVM e a rede neural LSTM foram treinados com 79.281 exemplos e avaliados em 19.821 tweets de teste (divisão estratificada 80/20, SEED=42).
                    O <b>BERTimbau</b> foi avaliado em uma partição estratificada de 5.000 exemplos de treino e 1.000 de teste devido à restrição computacional de memória GPU.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Desempenho por Modelo e Condição (F1-Macro)")

        fig = px.bar(
            df_geral,
            x="Modelo",
            y="F1 Macro",
            color="Condição",
            barmode="group",
            color_discrete_map=cores_condicoes,
            title="Desempenho Consolidado Oficial (F1-Score Macro)",
            text=df_geral["F1 Macro"].apply(lambda v: f"{v:.4f}"),
        )

        fig.update_layout(
            height=440,
            paper_bgcolor="#171426",
            plot_bgcolor="#171426",
            font=dict(color="#F5F3FF", size=13),
            title=dict(font=dict(size=16, color="#F5F3FF")),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#A9A5BC", size=13),
            ),
            yaxis=dict(range=[0.7, 1.03], gridcolor="#2D2A42", tickfont=dict(color="#A9A5BC", size=12)),
            xaxis=dict(gridcolor="#2D2A42", tickfont=dict(color="#F5F3FF", size=13)),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Tabela Completa dos 16 Experimentos Oficiais")
        col_filtro_mod, col_filtro_cond = st.columns(2)
        with col_filtro_mod:
            modelos_unicos = df_geral["Modelo"].unique().tolist()
            mod_sel = st.multiselect("Filtrar por modelo:", modelos_unicos, default=modelos_unicos)
        with col_filtro_cond:
            cond_unicas = df_geral["Condição"].unique().tolist()
            cond_sel = st.multiselect("Filtrar por condição:", cond_unicas, default=cond_unicas)

        df_filtrado = df_geral[df_geral["Modelo"].isin(mod_sel) & df_geral["Condição"].isin(cond_sel)].copy()
        st.dataframe(
            df_filtrado.style.format({
                "F1 Macro": "{:.4f}",
                "Accuracy": "{:.4f}",
                "Tempo (s)": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Diagnóstico de Erros do BERTimbau
        df_erros_bert = carregar_erros_bert()
        if df_erros_bert is not None:
            st.markdown("---")
            st.markdown("### Diagnóstico do BERTimbau: Impacto da Remoção de Emoticons")
            st.markdown(
                """
                <div style="color: #A9A5BC; font-size: 0.94rem; margin-bottom: 14px; line-height: 1.55;">
                    O experimento evidenciou um comportamento crítico no BERTimbau:
                    ao remover os marcadores emocionais explícitos do texto original, a taxa de erro saltou de <b>0,3% para 15,7%</b>.
                    Esse salto fornece indícios de dependência de marcadores associados à coleta do corpus (<i>distant supervision</i> com <code>:)</code> e <code>:(</code>),
                    sustentando a hipótese de que a altíssima precisão observada em Texto Bruto (99,7%) estava associada a esse viés.
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_b1, col_b2 = st.columns([1.5, 1])
            with col_b1:
                fig_err = px.bar(
                    df_erros_bert,
                    x="Condição",
                    y="Taxa de erro",
                    color="Condição",
                    color_discrete_map=cores_condicoes,
                    title="Taxa de Erro do BERTimbau (Amostra de 1.000 Tweets)",
                    text=df_erros_bert["Taxa de erro"].apply(lambda t: f"{t*100:.1f}%"),
                )
                fig_err.update_layout(
                    height=320,
                    paper_bgcolor="#171426",
                    plot_bgcolor="#171426",
                    font=dict(color="#F5F3FF", size=13),
                    title=dict(font=dict(size=15, color="#F5F3FF")),
                    yaxis=dict(range=[0, 0.3], gridcolor="#2D2A42", tickfont=dict(color="#A9A5BC", size=12)),
                    xaxis=dict(tickfont=dict(color="#F5F3FF", size=13)),
                    showlegend=False,
                )
                st.plotly_chart(fig_err, use_container_width=True)

            with col_b2:
                st.dataframe(
                    df_erros_bert.style.format({"Taxa de erro": "{:.1%}"}),
                    use_container_width=True,
                    hide_index=True,
                )

    # ------------------------------------------------------------
    # ABA 2: TABELA DE VARIAÇÃO (DELTA F1) DO NOTEBOOK
    # ------------------------------------------------------------
    with tab_delta:
        st.markdown(
            """
            <div style="color: #A9A5BC; font-size: 0.93rem; margin-bottom: 16px; line-height: 1.55;">
                Esta tabela reproduz estritamente a <b>Tabela 3 do notebook de experimentos</b>,
                calculando a diferença de F1-Macro de cada condição experimental em relação à linha de base (<b>Texto Bruto</b>):
                $$\\Delta F1 = F1_{\\text{condição}} - F1_{\\text{Texto Bruto}}$$
                Valores negativos indicam degradação do classificador ocasionada pelo pré-processamento.
            </div>
            """,
            unsafe_allow_html=True,
        )

        ordem_modelos = ['Regressão Logística', 'SVM', 'LSTM', 'BERTimbau']
        ordem_condicoes = ['Texto Bruto', 'Limpeza Básica', 'Pipeline Completo', 'Sem Emoticons']

        tabela_f1 = df_geral.pivot(
            index='Condição',
            columns='Modelo',
            values='F1 Macro'
        ).reindex(index=ordem_condicoes, columns=ordem_modelos)

        delta_f1 = tabela_f1.subtract(tabela_f1.loc['Texto Bruto'], axis='columns')

        st.markdown("#### Tabela Comparativa de Variação do F1-Score Macro")
        st.dataframe(
            delta_f1.style.format("{:+.4f}"),
            use_container_width=True,
        )

        # Gráfico comparativo de variação
        df_delta_melt = delta_f1.reset_index().melt(
            id_vars="Condição",
            var_name="Modelo",
            value_name="Delta F1"
        )
        df_delta_melt = df_delta_melt[df_delta_melt["Condição"] != "Texto Bruto"]

        fig_delta = px.bar(
            df_delta_melt,
            x="Modelo",
            y="Delta F1",
            color="Condição",
            barmode="group",
            color_discrete_map=cores_condicoes,
            title="Impacto do Pré-processamento no F1-Macro (em Relação ao Texto Bruto)",
            text=df_delta_melt["Delta F1"].apply(lambda v: f"{v:+.4f}"),
        )
        fig_delta.update_layout(
            height=400,
            paper_bgcolor="#171426",
            plot_bgcolor="#171426",
            font=dict(color="#F5F3FF", size=13),
            title=dict(font=dict(size=16, color="#F5F3FF")),
            yaxis=dict(gridcolor="#2D2A42", tickfont=dict(color="#A9A5BC", size=12)),
            xaxis=dict(gridcolor="#2D2A42", tickfont=dict(color="#F5F3FF", size=13)),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#A9A5BC", size=13),
            ),
        )
        st.plotly_chart(fig_delta, use_container_width=True)

        st.markdown(
            """
            <div class="metric-card" style="border-left: 4px solid #F59E0B; margin-top: 14px;">
                <b style="color: #F5F3FF; font-size: 0.92rem;">Destaques da Tabela de Variação (Diferença Absoluta de F1-Macro):</b>
                <div style="font-size: 0.88rem; color: #A9A5BC; line-height: 1.55; margin-top: 6px;">
                    • <b>Pipeline Completo:</b> Provocou quedas severas em todas as famílias de algoritmos:
                      <b>-0,0316</b> na Regressão Logística, <b>-0,0354</b> no SVM, <b>-0,0525</b> na LSTM e <b>-0,2524</b> no BERTimbau.<br>
                    • <b>Sem Emoticons:</b> Teve variação residual nos modelos lineares com TF-IDF tradicional (-0,0007 e +0,0001) e moderada na LSTM (-0,0147),
                      pois suas tokenizações descartam pontuações por padrão; já no BERTimbau, a perda foi expressiva (<b>-0,1560</b>), fornecendo indícios de dependência de marcadores associados à coleta do corpus.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------
    # ABA 3: MODELOS DEMONSTRATIVOS DO PROTÓTIPO LOCAL
    # ------------------------------------------------------------
    with tab_local:
        st.markdown(
            """
            <div style="color: #A9A5BC; font-size: 0.93rem; margin-bottom: 16px; line-height: 1.55;">
                Esta aba apresenta as métricas obtidas no treinamento dos <b>modelos empacotados localmente</b> em <code>models/</code>
                (arquivos <code>.joblib</code> utilizados para inferência em tempo real nas abas <i>Análise Individual</i> e <i>Comparar Condições</i>).
                Foram treinados com o mesmo corpus <code>data/Train3Classes.csv</code> sob split estratificado 80/20.
            </div>
            """,
            unsafe_allow_html=True,
        )

        df_local = carregar_metricas_locais()
        if df_local is not None:
            st.dataframe(
                df_local.style.format({
                    "F1 Macro": "{:.4f}",
                    "Accuracy": "{:.4f}",
                    "Tempo (s)": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Nota de transparência metodológica: os artefatos locais espelham os pipelines tradicionais de Regressão Logística "
                "e SVM do notebook original, permitindo demonstrações interativas rápidas sem necessidade de aceleradores gráficos GPU."
            )
        else:
            st.info("Arquivo de métricas locais (`metricas_treinamento_local.csv`) não encontrado.")


# ============================================================
# 4. PÁGINA: SOBRE O PROJETO & METODOLOGIA
# ============================================================
def pagina_sobre_pesquisa():
    st.markdown('<div class="section-title">Sobre a Pesquisa e Metodologia Científica</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### 1. Contexto e Objetivo do TCC II
        Este trabalho investiga como diferentes técnicas de pré-processamento textual influenciam o desempenho
        de algoritmos de Processamento de Linguagem Natural (NLP) aplicados à análise de sentimentos em textos informais
        no contexto do Português Brasileiro.

        **Questão central:**  
        > *Técnicas tradicionais de pré-processamento textual (como remoção de stopwords, pontuação e stemming)
        auxiliam ou prejudicam a classificação automática de sentimentos em tweets?*

        ---

        ### 2. Condições Experimentais Avaliadas
        No experimento, cada algoritmo foi testado e avaliado em quatro representações textuais independentes:

        1. **Texto Bruto:** Texto original exatamente como publicado, mantendo pontuações, emoticons, hashtags e menções.
        2. **Limpeza Básica:** Remoção de URLs, menções `@usuario`, conversão para minúsculas, preservação do conteúdo textual de hashtags e redução de alongamentos (ex: *muitoooo* $\\rightarrow$ *muitoo*).
        3. **Pipeline Completo:** Limpeza básica somada à remoção de stopwords (com preservação de termos de negação como *não*, *nunca*, *jamais*) e redução ao radical via `RSLPStemmer`.
        4. **Sem Emoticons:** Condição diagnóstica que remove os marcadores emocionais explícitos (`:)`, `:D`, `:(` etc.) via expressão regular para mensurar o impacto direto desses marcadores expressivos.

        ---

        ### 3. Modelos Investigados no TCC
        - **Regressão Logística (TF-IDF):** Unigramas e bigramas, até 50.000 features, representação sublinear. Modelo leve, altamente interpretável e base do presente protótipo.
        - **Support Vector Machine (LinearSVC):** Hiperplano ótimo de separação no espaço vetorial TF-IDF.
        - **Rede Neural LSTM:** Modelo recorrente capaz de capturar dependências de ordem temporal entre os tokens sequenciados.
        - **BERTimbau (`bert-base-portuguese-cased`):** Transformer pré-treinado no corpus pt-BR, adaptado via fine-tuning.

        ---

        ### 4. Principais Descobertas Científicas
        - **Degradação pelo Pipeline Completo:** Em classificadores estatísticos (Regressão Logística e SVM), a aplicação do *Pipeline Completo* reduziu o F1-Score Macro em mais de 0,03 em termos absolutos (~0,83 $\\rightarrow$ ~0,80), demonstrando que a perda de desinências verbais e palavras estruturais afeta a interpretabilidade afetiva.
        - **Viés de Coleta por Emoticons:** O corpus utilizado (`Train3Classes.csv`) possui associação entre a estratégia de coleta por consulta (`:)` e `:(`) e os rótulos de sentimento. Modelos de alta capacidade como o BERTimbau aprendem rapidamente a explorar esses marcadores quando presentes.

        ---

        ### 5. Ressalva Metodológica sobre Emoticons e Tokenização (TF-IDF, LSTM e BERTimbau)
        - **Filtros de Pontuação (TF-IDF e LSTM):** Tanto o extrator linear `TfidfVectorizer` (com regex alfanumérica padrão `(?u)\\b\\w\\w+\\b`) quanto o `Tokenizer` do Keras utilizado na rede LSTM (com os filtros de pontuação padrão) descartam símbolos de pontuação e caracteres não alfanuméricos isolados. Portanto, emoticons como `:)` e `:(` não se transformam em tokens diretos no vocabulário dessas abordagens. Variações de desempenho entre *Texto Bruto* e *Sem Emoticons* nesses modelos decorrem de ajustes contextuais e estocásticos no treinamento, e não da leitura literal desses símbolos.
        - **Efeito no BERTimbau:** Em forte contraste, o transformer (**BERTimbau**), baseado em subpalavras (WordPiece), processa esses caracteres dentro de seu espaço representacional pré-treinado. Ao remover os marcadores explícitos, a taxa de erro saltou de **0,3% para 15,7%**, fornecendo indícios de dependência de marcadores associados à coleta por rótulos fracos (*distant supervision* baseada nas queries `:)` e `:(`).

        ---

        ### 6. Limitações Metodológicas e Rigor Acadêmico
        - **Protótipo Demonstrativo vs. Artigo Científico:** A aplicação web apresenta os resultados oficiais definitivos do TCC II obtidos nos 16 experimentos do notebook, e mantém separadas as métricas dos modelos treinados localmente em `models/` para execução interativa.
        - **Ironia e Contexto:** Frases com ambiguidade, ironia ou duplo sentido representam um limite intrínseco para classificadores supervisionados desacoplados de histórico discursivo ou contexto pragmático.
        - **Vocabulário Fixo (OOV):** Modelos lineares não conseguem ponderar termos inexistentes no vocabulário de treino (adjetivos raros, neologismos da internet), recorrendo ao viés marginal das classes.
        """
    )


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================
def main():
    render_header()
    aba, modelo = render_sidebar()

    if aba == "Análise Individual":
        pagina_analise_individual(modelo)
    elif aba == "Comparar Condições":
        pagina_comparar_condicoes(modelo)
    elif aba == "Resultados do TCC II":
        pagina_resultados_experimentais()
    elif aba == "Sobre a Pesquisa":
        pagina_sobre_pesquisa()


if __name__ == "__main__":
    main()
