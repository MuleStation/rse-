# ultra_esg_dashboard.py
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

# -------------------------------
# Fonctions
# -------------------------------

def extract_text_from_pdf(pdf_file):
    """Extraire le texte d'un PDF uploadé via Streamlit"""
    file_bytes = pdf_file.read()  # lire le PDF en mémoire
    pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")  # ouvrir depuis le flux
    text = ""
    for page in pdf_doc:
        text += page.get_text()
    return text

def get_esg_news(company, max_articles=5):
    """Récupérer les dernières news ESG via Google News"""
    url = f"https://news.google.com/search?q={company}+ESG&hl=en-GB&gl=US&ceid=US:en"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    headlines = [a.text for a in soup.find_all('a') if a.text.strip()]
    return headlines[:max_articles]

def esg_score(texts):
    """Calculer le score ESG via FinBERT"""
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-esg")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-esg")
    nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)
    results = []
    for t in texts:
        if len(t.strip()) > 50:
            results.append(nlp(t)[0])
    if results:
        df = pd.DataFrame(results)
        return df.groupby('label')['score'].mean().reset_index()
    else:
        return pd.DataFrame(columns=['label','score'])

def benchmark_sector(scores, sector="Tech"):
    """Ajoute un benchmark sectoriel fictif pour comparaison"""
    sector_avg = {'positive':0.6, 'neutral':0.25, 'negative':0.15}
    df_sector = pd.DataFrame(list(sector_avg.items()), columns=['label','score'])
    df_sector['type'] = 'Secteur'
    scores['type'] = 'Entreprise'
    df_combined = pd.concat([scores, df_sector])
    return df_combined

# -------------------------------
# Dashboard Streamlit
# -------------------------------

st.set_page_config(page_title="Ultimate ESG Dashboard", layout="wide")
st.title("Ultimate ESG & RSE Dashboard")

# Input entreprise
company = st.text_input("Nom de l'entreprise", "Apple")
pdf_file = st.file_uploader("Téléchargez le rapport RSE / Annuel (PDF)", type=["pdf"])

# Affichage news ESG
if company:
    st.subheader(f"Dernières news ESG pour {company}")
    news = get_esg_news(company)
    if news:
        for i, n in enumerate(news):
            st.markdown(f"{i+1}. {n}")
    else:
        st.write("Aucune news ESG trouvée.")

# Analyse PDF
if pdf_file is not None:
    st.subheader(f"Analyse PDF RSE pour {company}")
    text = extract_text_from_pdf(pdf_file)
    st.text("PDF chargé. Analyse en cours...")
    
    # Scoring ESG
    df_scores = esg_score(text.split("\n\n"))
    
    if not df_scores.empty:
        # Benchmark sectoriel
        df_bench = benchmark_sector(df_scores)
        
        st.subheader("Scores ESG et Benchmark sectoriel")
        st.write(df_bench)
        
        # Radar chart interactif
        fig = px.line_polar(df_bench, r='score', theta='label', color='type', line_close=True,
                            title=f"Radar ESG: {company} vs Secteur")
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap des scores
        pivot = df_bench.pivot(index='type', columns='label', values='score')
        fig2, ax = plt.subplots()
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig2)
    else:
        st.write("Aucun score calculé. Le PDF est peut-être trop court ou non analysable.")
