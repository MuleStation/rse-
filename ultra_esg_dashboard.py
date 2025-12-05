# ultimate_esg_dashboard.py

import pandas as pd
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import plotly.express as px
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


# -------------------------------
# FONCTIONS DE BASE
# -------------------------------

# 1. Extraire texte PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# 2. Récupérer news ESG
def get_esg_news(company, max_articles=10):
    url = f"https://news.google.com/search?q={company}+ESG&hl=en-GB&gl=US&ceid=US:en"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    headlines = [a.text for a in soup.find_all('a') if a.text.strip()]
    return headlines[:max_articles]


# 3. Scoring ESG via NLP (FinBERT)
def esg_score(texts):
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-esg")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-esg")
    nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)
    results = []
    for t in texts:
        if len(t.strip()) > 50:
            res = nlp(t)[0]
            results.append(res)
    if results:
        df = pd.DataFrame(results)
        return df.groupby('label')['score'].mean().reset_index()
    else:
        return pd.DataFrame(columns=['label', 'score'])


# 4. Benchmark sectoriel simple (exemple fictif)
def benchmark_sector(scores, sector="Tech"):
    # Exemple fictif : moyenne des scores ESG du secteur
    sector_avg = {'positive': 0.6, 'neutral': 0.25, 'negative': 0.15}
    df_sector = pd.DataFrame(list(sector_avg.items()), columns=['label', 'score'])
    df_sector['type'] = 'Sector Avg'
    scores['type'] = 'Company'
    df_combined = pd.concat([scores, df_sector])
    return df_combined


# -------------------------------
# DASHBOARD STREAMLIT
# -------------------------------

st.title("Ultimate ESG & RSE Dashboard")

# Input entreprise
company = st.text_input("Nom de l'entreprise:", "Apple")
pdf_file = st.file_uploader("Rapport RSE / Annuel (PDF)", type=["pdf"])

if company:
    st.subheader(f"News ESG pour {company}")
    news = get_esg_news(company)
    for i, n in enumerate(news):
        st.markdown(f"{i + 1}. {n}")

if pdf_file:
    st.subheader(f"Analyse PDF RSE pour {company}")
    text = extract_text_from_pdf(pdf_file)
    st.text("Extraction terminée, analyse en cours...")

    df_scores = esg_score(text.split("\n\n"))

    if not df_scores.empty:
        # Benchmark sectoriel
        df_bench = benchmark_sector(df_scores)

        st.subheader("Scores ESG avec Benchmark Sectoriel")
        st.write(df_bench)

        # Radar chart
        fig = px.line_polar(df_bench, r='score', theta='label', color='type', line_close=True,
                            title=f"Radar ESG {company} vs Secteur")
        st.plotly_chart(fig)

        # Heatmap des scores
        pivot = df_bench.pivot(index='type', columns='label', values='score')
        fig2, ax = plt.subplots()
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig2)
    else:
        st.write("Aucun score calculé. Le PDF est peut-être trop court ou non analysable.")
