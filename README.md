# TCC II — Análise de Sentimentos em Textos Informais no Português Brasileiro

> **Trabalho de Conclusão de Curso II em Ciência da Computação**  
> **Tema:** Avaliação de Algoritmos e Técnicas de NLP para Análise de Sentimentos em Textos Informais no Contexto do Português Brasileiro  
> **Protótipo Web:** Interface Interativa em Streamlit para Demonstração Acadêmica à Banca Avaliadora

---

## 📌 Visão Geral do Projeto

Este repositório contém o código-fonte do protótipo desenvolvido para o **TCC II**, cujo objetivo científico é investigar como diferentes técnicas de pré-processamento textual influenciam o desempenho de modelos de classificação de sentimentos aplicados a textos informais (tweets em português brasileiro).

### A Questão Central da Pesquisa
> *Técnicas tradicionais e intensas de pré-processamento textual (remoção de stopwords, remoção de pontuações e stemming) auxiliam ou prejudicam a classificação automática de sentimentos em textos informais em português?*

A aplicação web permite:
1. **Inferência Real Individual:** Classificar sentenças em três classes (**Positivo**, **Negativo**, **Neutro**) sob quatro condições experimentais.
2. **Diagnóstico da Representação ("Entenda esta previsão"):** Exibe exatamente quais termos da frase foram reconhecidos pelo TF-IDF e quais ficaram fora do vocabulário (OOV), com seus pesos numéricos e ressalva metodológica.
3. **Comparação Simultânea de Condições:** Testar o mesmo texto em tempo real sob as quatro condições, alertando visualmente quando o pré-processamento altera a previsão final do classificador.
4. **Demonstração Didática Guiada:** Bloco interpretativo que explica de forma acessível e objetiva o que a convergência ou divergência entre condições demonstra experimentalmente.
5. **Exibição dos Resultados Consolidados do TCC:** Tabela completa das 16 combinações e gráficos interativos com as métricas oficiais (F1-Macro, Acurácia) obtidas nos experimentos com Regressão Logística, SVM, LSTM e BERTimbau.
6. **Fundamentação Metodológica & Limitações:** Contextualização acadêmica detalhada para a banca avaliadora.

---

## 🔬 Condições Experimentais

Cada modelo foi treinado e avaliado em quatro representações textuais independentes:

| Condição | Descrição da Transformação |
|---|---|
| **Texto Bruto** | Texto original sem qualquer alteração prévia. Preserva pontuações, emoticons, hashtags e menções. |
| **Limpeza Básica** | Conversão para minúsculas, remoção de URLs e menções (`@usuario`), preservação do texto das hashtags e redução de repetições de caracteres (alongamentos). |
| **Pipeline Completo** | Limpeza básica + remoção de stopwords (com preservação estrita de negações como *não*, *nunca*, *jamais*) + radicalização via `RSLPStemmer`. |
| **Sem Emoticons** | Condição diagnóstica que remove apenas os marcadores emocionais explícitos (`:)`, `:D`, `:(`, etc.) via regex. |

---

## 🏗️ Estrutura do Repositório

```text
tcc2-streamlit/
│
├── app.py                      # Interface gráfica principal em Streamlit
├── train_models.py             # Script offline de treinamento e exportação dos modelos
├── preprocessing.py           # Funções e expressões regulares de pré-processamento (idênticas ao notebook)
├── inference.py                # Módulo desacoplado de inferência e predição
├── config.py                   # Hiperparâmetros, constantes e configurações centrais
├── requirements.txt            # Dependências Python do projeto
├── README.md                   # Documentação completa do projeto
├── .gitignore                  # Arquivos ignorados pelo versionamento Git
│
├── data/
│   └── Train3Classes.csv       # Corpus de tweets original (100.000 registros brutos)
│
├── models/
│   ├── lr_texto_bruto.joblib           # Pipeline TF-IDF + Regressão Logística (Texto Bruto)
│   ├── lr_limpeza_basica.joblib        # Pipeline TF-IDF + Regressão Logística (Limpeza Básica)
│   ├── lr_pipeline_completo.joblib     # Pipeline TF-IDF + Regressão Logística (Pipeline Completo)
│   ├── lr_sem_emoticons.joblib         # Pipeline TF-IDF + Regressão Logística (Sem Emoticons)
│   ├── svm_*.joblib                    # Pipelines treinados para SVM (LinearSVC)
│   └── metadata.json                   # Metadados do treinamento e métricas locais
│
├── results/
│   ├── metricas_treinamento_local.csv  # Métricas obtidas no treinamento local
│   └── resultados_recuperados_TCC2/    # CSVs consolidados dos experimentos originais do TCC
│
└── tests/
    ├── test_preprocessing.py   # Testes unitários do pipeline textual e regex
    └── test_inference.py       # Testes unitários do motor de predição e inferência
```

---

## 🚀 Instalação e Execução Local

### 1. Clonar o repositório e criar ambiente virtual

```bash
# Criar ambiente virtual Python (recomendado Python 3.10+)
python -m venv venv

# Ativar no Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Ativar no Linux / macOS
source venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar os testes automatizados

```bash
python -m pytest tests/
```

### 4. Treinar e exportar os modelos (opcional, já incluídos)

Caso queira reproduzir o treinamento do zero a partir do `data/Train3Classes.csv`:

```bash
python train_models.py
```
*O script aplicará a auditoria dos dados, divisão estratificada 80/20 (SEED=42), gerará as 4 representações e salvará os artefatos `.joblib` em `models/`.*

### 5. Iniciar a aplicação Streamlit

```bash
streamlit run app.py
```

A aplicação abrirá automaticamente no navegador em `http://localhost:8501`.

---

## ☁️ Publicação no Streamlit Community Cloud

O projeto foi arquitetado especificamente para ser facilmente publicado no **Streamlit Community Cloud**:

1. Suba o projeto para um repositório no seu **GitHub** (`git push origin main`);
2. Acesse [share.streamlit.io](https://share.streamlit.io/) e faça login com sua conta do GitHub;
3. Clique em **"New app"**;
4. Selecione o repositório, branch `main` e defina o arquivo principal como `app.py`;
5. Clique em **"Deploy"**.

Como os modelos scikit-learn treinados pesam poucos megabytes e o Streamlit apenas carrega os artefatos via `joblib` com `st.cache_resource`, a inicialização na nuvem é rápida e gratuita.

---

## 📊 Principais Resultados Experimentais

Métricas consolidadas a partir do conjunto de teste (19.821 tweets para baselines e 1.000 para BERTimbau):

| Modelo | Texto Bruto (F1) | Sem Emoticons (F1) | Limpeza Básica (F1) | Pipeline Completo (F1) |
|---|---|---|---|---|
| **Regressão Logística** | 0,8342 | 0,8332 | 0,8278 | **0,8025** (-3,2%) |
| **SVM (LinearSVC)** | 0,8301 | 0,8305 | 0,8219 | **0,7947** (-3,5%) |
| **LSTM** | 0,8528 | 0,8406 | 0,8396 | **0,8003** (-5,3%) |
| **BERTimbau** | 0,9970 | **0,8410** (-15,6%) | 0,8287 | **0,7446** (-25,2%) |

### Conclusões Científicas
1. **O Pré-processamento Intenso Prejudica:** Em todos os 4 modelos avaliados, o **Pipeline Completo** (remoção de stopwords e stemming) apresentou o pior desempenho, evidenciando que termos eliminados e modificações nos radicais removem nuances semânticas cruciais em textos informais.
2. **Dependência de Emoticons:** O BERTimbau alcançou 99,7% em Texto Bruto devido à associação quase perfeita entre consultas de coleta (`:)`, `:(`) e os rótulos de sentimento. Ao remover os emoticons, a taxa de erro subiu consideravelmente, revelando o papel determinante dessas pistas gráficas na linguagem informal das redes sociais.

---

## ⚖️ Limitações Científicas

- **Supervisão Ruidosa (Distant Supervision):** O corpus foi coletado a partir de buscas por emoticons, o que gera correlação direta entre certas construções e as classes de sentimento.
- **Ironia e Figuras de Linguagem:** Modelos estatísticos baseados em n-gramas possuem limitações estruturais para detectar sarcasmo e inversão de sentido.
- **Finalidade:** Protótipo demonstrativo para fins de pesquisa acadêmica, não devendo ser utilizado como ferramenta de diagnóstico psicológico ou emocional.

---

## 👨‍💻 Autor

- **Discente:** Graduando em Ciência da Computação
- **Trabalho:** Trabalho de Conclusão de Curso II (TCC II)
