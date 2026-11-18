import streamlit as st
import yfinance as yf
import pandas as pd
import feedparser
import urllib.parse
import requests
from google import genai

# Configuration de la page
st.set_page_config(page_title="Boussole des Marchés IA - Pro", layout="wide")
st.title("📊 Boussole des Marchés Boursiers & Analyse IA (Gemini)")

# Barre latérale pour la clé API Google Gemini
st.sidebar.header("Configuration")
gemini_api_key = st.sidebar.text_input("Clé API Google Gemini", type="password")
st.sidebar.markdown("[Obtenir une clé gratuite sur Google AI Studio](https://google.com)")

# 1. Définition des 4 actifs principaux
indices = {
    "S&P 500 (USA)": "^GSPC",
    "Euro Stoxx 600 NR (Europe)": "EXSA.DE",
    "MSCI Asia Pacific (Asie)": "EPP",
    "EUR / USD": "EURUSD=X"
}

# 2. Définition des poids lourds qui composent les fonds
leaders = {
    "USA (S&P 500)": {"Nvidia": "NVDA", "Apple": "AAPL", "Microsoft": "MSFT"},
    "Europe (Stoxx 600)": {"ASML": "ASML.AS", "LVMH": "MC.PA", "SAP": "SAP.DE"},
    "Asie (MSCI Asia Pac)": {"TSMC": "TSM", "Toyota": "TM", "Sony": "SONY"}
}

# --- SECTION 1 : VIX (Indice de la peur) ---
vix_data = yf.Ticker("^VIX").history(period="5d")
vix_actuel = vix_data['Close'].iloc[-1]

st.subheader(f"1. Climat de marché (VIX) : {vix_actuel:.2f}")
if vix_actuel > 25:
    st.error("⚠️ Forte nervosité (VIX > 25). Risque élevé de baisse ou de mouvements brusques.")
elif vix_actuel > 18:
    st.warning("⚖️ Volatilité modérée. Les investisseurs surveillent les catalyseurs.")
else:
    st.success("✅ Marchés calmes (VIX bas). Environnement propice à la hausse.")

st.divider()

# --- SECTION 2 : TENDANCE TECHNIQUE DES INDICES ---
st.subheader("2. Direction Technique des Indices (Prix vs Moyenne Mobile 50j)")
cols = st.columns(4)
resume_indices = []

for i, (nom, ticker) in enumerate(indices.items()):
    df = yf.Ticker(ticker).history(period="3mo")
    prix_actuel = df['Close'].iloc[-1]
    prix_hier = df['Close'].iloc[-2]
    variation = ((prix_actuel - prix_hier) / prix_hier) * 100
    mm50 = df['Close'].rolling(window=50).mean().iloc[-1]
    
    tendance = "HAUSSIER 🟢" if prix_actuel > mm50 else "BAISSIER 🔴"
    resume_indices.append(f"{nom}: {tendance} (Var: {variation:.2f}%)")
    
    with cols[i]:
        st.metric(label=nom, value=f"{prix_actuel:.2f}", delta=f"{variation:.2f}%")
        st.write(f"**Signal :** {tendance}")

st.divider()

# --- SECTION 3 : POIDS LOURDS & RÉSULTATS TRIMESTRIELS ---
st.subheader("3. Santé des Entreprises Leaders & Prochains Résultats")
cols_leaders = st.columns(3)
resume_entreprises = []

for idx, (zone, entreprises) in enumerate(leaders.items()):
    with cols_leaders[idx]:
        st.markdown(f"**{zone}**")
        for nom_ent, tick_ent in entreprises.items():
            stock = yf.Ticker(tick_ent)
            hist = stock.history(period="5d")
            var_ent = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
            
            date_res = "Non communiquée"
            try:
                cal = stock.calendar
                if cal and 'Earnings Date' in cal and len(cal['Earnings Date']) > 0:
                    date_res = cal['Earnings Date'].strftime('%d/%m/%Y')
            except Exception:
                pass
            
            icone = "📈" if var_ent >= 0 else "📉"
            st.write(f"{icone} **{nom_ent}** ({var_ent:+.2f}%) | Résultats : `{date_res}`")
            resume_entreprises.append(f"{nom_ent} ({zone}): {var_ent:+.2f}%, prochains résultats le {date_res}")

st.divider()

# --- SECTION 4 : ACTUALITÉS MACRO, CONFLITS & RÉSULTATS ---
st.subheader("4. Analyse IA : Macroéconomie, Géopolitique & Entreprises")

def get_news():
    headers = {"User-Agent": "Mozilla/5.0"}
    urls = [
        "https://google.com",
        "https://google.com",
        "https://google.com"
    ]
    titres = []
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:3]:
                titres.append(entry.title)
        except Exception:
            pass
    return titres

news_list = get_news()

with st.expander("Voir les 9 dernières dépêches analysées"):
    if news_list:
        for n in news_list:
            st.write(f"- {n}")
    else:
        st.write("Aucune dépêche récupérée pour le moment.")

if gemini_api_key:
    if st.button("🚀 Générer le diagnostic complet avec Gemini"):
        with st.spinner("Gemini 3.8 croise les graphiques, les résultats et la géopolitique..."):
            try:
                client = genai.Client(api_key=gemini_api_key)
                prompt = f"""
                Tu es un stratège de marché senior. Analyse la situation actuelle en croisant toutes ces données :
                
                1. VIX (Volatilité) : {vix_actuel:.2f}
                2. Tendance technique des indices : {', '.join(resume_indices)}
                3. Performance des entreprises leaders et dates de résultats : {', '.join(resume_entreprises)}
                4. Dernières actualités (Macro, Conflits armés, Résultats) :
                {chr(10).join(news_list) if news_list else "Utilise le contexte économique actuel."}
                
                Donne-moi une analyse structurée et directe :
                - **Impact Macro & Géopolitique :** Les conflits ou annonces économiques freinent-ils le marché ?
                - **Dynamique des Entreprises :** Les poids lourds tirent-ils les indices vers le haut ou le bas ? Y a-t-il des publications de résultats imminentes à surveiller ?
                - **Direction par actif :** Un mot (Haussier / Neutre / Baissier) et une justification courte pour le S&P 500, l'Euro Stoxx 600, le MSCI Asia Pacific et l'EUR/USD.
                """
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Erreur avec l'API Gemini : {e}")
else:
    st.info("👈 Appuie sur la flèche >> en haut à gauche et colle ta clé API Google Gemini pour activer le diagnostic IA.")
