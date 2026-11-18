import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import requests
from google import genai

st.set_page_config(page_title="Terminal Pro Marchés - 10 Piliers", layout="wide")
st.title("🏛️ Terminal Pro d'Analyse des Marchés (10 Piliers)")

st.sidebar.header("⚙️ Paramètres")
gemini_api_key = st.sidebar.text_input("Clé API Google Gemini (Gratuite)", type="password")
st.sidebar.markdown("[Créer une clé sur Google AI Studio](https://google.com)")

st.sidebar.divider()
st.sidebar.subheader("🔍 Actif personnalisé (Onglets 2, 4 & 5)")
custom_ticker = st.sidebar.text_input("Ticker Yahoo Finance (ex: NVDA, MC.PA, BTC-USD)", value="NVDA").upper()

@st.cache_data(ttl=600)
def get_market_data(ticker, period="1y"):
    df = yf.Ticker(ticker).history(period=period)
    if df.empty or len(df) < 50:
        return None
    
    close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']

    df['MM20'] = close.rolling(20).mean()
    df['MM50'] = close.rolling(50).mean()
    df['MM100'] = close.rolling(100).mean()
    df['MM200'] = close.rolling(200).mean()
    df['EMA21'] = close.ewm(span=21, adjust=False).mean()

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    low14, high14 = low.rolling(14).min(), high.rolling(14).max()
    df['Stoch_K'] = 100 * ((close - low14) / (high14 - low14))
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    std20 = close.rolling(20).std()
    df['BB_Upper'] = df['MM20'] + (std20 * 2)
    df['BB_Lower'] = df['MM20'] - (std20 * 2)

    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    up_move, down_move = high - high.shift(), low.shift() - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14).mean() / df['ATR'])
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14).mean() / df['ATR'])
    df['ADX'] = (100 * abs(plus_di - minus_di) / (plus_di + minus_di)).ewm(alpha=1/14).mean()

    df['Tenkan'] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    df['Kijun'] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    df['OBV'] = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    df['Z_Score'] = (close - df['MM50']) / close.rolling(50).std()
    df['Vol_Realisee'] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    df['Drawdown'] = ((close - close.cummax()) / close.cummax()) * 100
    return df
@st.cache_data(ttl=900)
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

actifs_principaux = {
    "S&P 500 (USA)": ("^GSPC", "FOREXCOM:SPXUSD"),
    "Euro Stoxx 600 NR (Europe)": ("EXSA.DE", "XETR:EXSA"),
    "MSCI Asia Pacific (Asie)": ("EPP", "AMEX:EPP"),
    "EUR / USD": ("EURUSD=X", "FX:EURUSD")
}

leaders = {
    "USA (S&P 500)": {"Nvidia": "NVDA", "Apple": "AAPL", "Microsoft": "MSFT"},
    "Europe (Stoxx 600)": {"ASML": "ASML.AS", "LVMH": "MC.PA", "SAP": "SAP.DE"},
    "Asie (MSCI Asia Pac)": {"TSMC": "TSM", "Toyota": "TM", "Sony": "SONY"}
}

data_cache = {nom: get_market_data(yf_t) for nom, (yf_t, _) in actifs_principaux.items()}
vix_df = get_market_data("^VIX", period="3mo")
vix_val = vix_df['Close'].iloc[-1] if vix_df is not None else 20.0

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 1. Convergence & IA", "📈 2. Technique & TV", "🌍 3. Macro & Calendrier", "🏢 4. Fondamental", "🛡️ 5. Quant & Risque"
])

with tab1:
    st.subheader("1. Synthèse des 4 Marchés & Régime (Piliers 4, 9 & 10)")
    cols = st.columns(4)
    score_convergence = 5.0 + (1.5 if vix_val < 18 else (-2.0 if vix_val > 25 else 0))
    resume_tech_ia = []

    for i, (nom, _) in enumerate(actifs_principaux.items()):
        df = data_cache[nom]
        if df is not None:
            last, prev = df.iloc[-1], df.iloc[-2]
            var = ((last['Close'] - prev['Close']) / prev['Close']) * 100
            pts = sum([last['Close'] > last['MM50'], last['Close'] > last['MM200'], last['MACD'] > last['MACD_Signal'], 45 < last['RSI'] < 70])
            sig = "🟢 HAUSSIER" if pts >= 3 else ("🔴 BAISSIER" if pts <= 1 else "🟡 NEUTRE")
            score_convergence += 0.6 if pts >= 3 else (-0.6 if pts <= 1 else 0)
            resume_tech_ia.append(f"{nom}: {sig} (Prix: {last['Close']:.2f}, RSI: {last['RSI']:.1f}, ADX: {last['ADX']:.1f})")
            with cols[i]:
                st.metric(label=nom, value=f"{last['Close']:.2f}", delta=f"{var:+.2f}%")
                st.write(f"**Direction :** {sig}")

    score_convergence = max(0.0, min(10.0, score_convergence))
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Score de Convergence Global", f"{score_convergence:.1f} / 10")
    regime_vol = "Calme (Risk-On) 🟢" if vix_val < 18 else ("Stressé (Risk-Off) 🔴" if vix_val > 25 else "Modéré 🟡")
    c2.metric("VIX (Indice de Peur)", f"{vix_val:.2f}", delta=regime_vol, delta_color="off")
    spx_adx = data_cache["S&P 500 (USA)"]['ADX'].iloc[-1] if data_cache["S&P 500 (USA)"] is not None else 20
    c3.metric("Régime Structurel (S&P 500)", "Tendance Forte (ADX > 25)" if spx_adx > 25 else "Range (ADX < 25)")

    st.divider()
    st.subheader("2. Santé des Entreprises Leaders & Prochains Résultats")
    cols_leaders = st.columns(3)
    resume_entreprises = []
    for idx, (zone, entreprises) in enumerate(leaders.items()):
        with cols_leaders[idx]:
            st.markdown(f"**{zone}**")
            for nom_ent, tick_ent in entreprises.items():
                try:
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
                    st.write(f"{'📈' if var_ent >= 0 else '📉'} **{nom_ent}** ({var_ent:+.2f}%) | Résultats : `{date_res}`")
                    resume_entreprises.append(f"{nom_ent} ({zone}): {var_ent:+.2f}%, résultats le {date_res}")
                except Exception:
                    st.write(f"• **{nom_ent}** : Indisponible")
    st.divider()
    st.subheader("3. Diagnostic Stratégique Complet (Macro + Entreprises + Géopolitique)")
    news_list = get_news()
    with st.expander("📰 Voir les 9 dernières dépêches Macro, Conflits & Entreprises"):
        for n in news_list: st.write(f"- {n}")

    if gemini_api_key:
        if st.button("🚀 Lancer l'analyse institutionnelle en 10 piliers (Gemini)"):
            with st.spinner("Analyse croisée des indices, des leaders mondiaux, du VIX et de la géopolitique..."):
                client = genai.Client(api_key=gemini_api_key)
                prompt = f"""Tu es un gérant de fonds senior. Analyse selon 10 piliers :
                - Score convergence : {score_convergence:.1f}/10 | VIX : {vix_val:.2f} ({regime_vol})
                - Indices : {', '.join(resume_tech_ia)}
                - Poids lourds & résultats : {', '.join(resume_entreprises)}
                - Actualités : {chr(10).join(news_list) if news_list else 'Standard'}
                Donne : 1. Impact Macro & Géopolitique, 2. Dynamique Entreprises, 3. Verdict (Haussier/Neutre/Baissier) par actif (S&P 500, Euro Stoxx 600, MSCI Asia Pac, EUR/USD), 4. Gestion du risque."""
                modeles = [m.name.replace("models/", "") for m in client.models.list() if "gemini" in m.name and "flash" in m.name] or ["gemini-3.8-flash", "gemini-3.8-flash-lite"]
                succes = False
                for mod in modeles:
                    try:
                        rep = client.models.generate_content(model=mod, contents=prompt)
                        st.success(f"Rapport généré via {mod}")
                        st.markdown(rep.text)
                        succes = True
                        break
                    except Exception:
                        continue
                if not succes:
                    st.warning("⚠️ Serveurs Google saturés. Synthèse immédiate :")
                    for item in resume_tech_ia: st.write(f"- {item}")
    else:
        st.info("👈 Ouvre le menu (flèche >> en haut à gauche) pour coller ta clé API Google Gemini gratuite.")

with tab2:
    st.subheader("Analyse Technique & Graphiques Temps Réel TradingView")
    choix_actif = st.selectbox("Choisir un marché :", list(actifs_principaux.keys()) + [f"Personnalisé ({custom_ticker})"])
    yf_sel, tv_sel = actifs_principaux[choix_actif] if choix_actif in actifs_principaux else (custom_ticker, custom_ticker.replace("-USD", "USD").replace(".PA", ""))
    df_tech = data_cache[choix_actif] if choix_actif in actifs_principaux else get_market_data(yf_sel)

    components.html(f"""<div class="tradingview-widget-container"><div id="tv_c"></div><script src="https://tradingview.com"></script><script>new TradingView.widget({{"width":"100%","height":420,"symbol":"{tv_sel}","interval":"D","timezone":"Europe/Paris","theme":"dark","style":"1","locale":"fr","studies":["MASimple@tv-basicstudies","RSI@tv-basicstudies","MACD@tv-basicstudies"],"container_id":"tv_c"}});</script></div>""", height=430)
    components.html(f"""<div class="tradingview-widget-container"><script src="https://tradingview.com" async>{{"interval":"1D","width":"100%","height":380,"symbol":"{tv_sel}","showIntervalTabs":true,"displayMode":"single","locale":"fr","colorTheme":"dark"}}</script></div>""", height=390)

    if df_tech is not None:
        last, prev = df_tech.iloc[-1], df_tech.iloc[-2]
        pivot = (prev['High'] + prev['Low'] + prev['Close']) / 3
        c_t1, c_t2, c_t3 = st.columns(3)
        with c_t1:
            st.markdown("**Tendance & Ichimoku**")
            st.write(f"- MM20/50 : `{last['MM20']:.2f}` / `{last['MM50']:.2f}`")
            st.write(f"- MM100/200 : `{last['MM100']:.2f}` / `{last['MM200']:.2f}`")
            st.write(f"- Tenkan/Kijun : `{last['Tenkan']:.2f}` / `{last['Kijun']:.2f}` | ADX : `{last['ADX']:.1f}`")
        with c_t2:
            st.markdown("**Momentum**")
            st.write(f"- RSI (14) : `{last['RSI']:.1f}` | Stoch : `{last['Stoch_K']:.1f}`")
            st.write(f"- MACD : `{last['MACD']:.2f}` (Signal : `{last['MACD_Signal']:.2f}`)")
        with c_t3:
            st.markdown("**Volatilité & Pivots**")
            st.write(f"- Bollinger : `{last['BB_Lower']:.2f}` — `{last['BB_Upper']:.2f}`")
            st.write(f"- ATR (14j) : `{last['ATR']:.2f}` | Pivot : `{pivot:.2f}`")
            with tab3:
    st.subheader("Macroéconomie, Inter-Marchés & Calendrier Investing.com")
    macro_tickers = {"Taux US 10Y": "^TNX", "DXY (Dollar)": "DX-Y.NYB", "Or": "GC=F", "Pétrole WTI": "CL=F", "Cuivre": "HG=F", "Crédit HYG": "HYG", "SPX Equal-Weight": "RSP", "Small Caps IWM": "IWM"}
    m_cols = st.columns(4)
    m_prices = {}
    for idx, (m_nom, m_tick) in enumerate(macro_tickers.items()):
        m_df = get_market_data(m_tick, period="3mo")
        if m_df is not None:
            p_now, p_prev = m_df['Close'].iloc[-1], m_df['Close'].iloc[-2]
            m_prices[m_nom] = m_df['Close']
            m_cols[idx % 4].metric(m_nom, f"{p_now:.2f}", f"{((p_now - p_prev)/p_prev)*100:+.2f}%")

    st.divider()
    r1, r2 = st.columns(2)
    spx_df = data_cache["S&P 500 (USA)"]
    if spx_df is not None and "Or" in m_prices:
        r1.metric("Ratio S&P 500 / Or", f"{(spx_df['Close'].iloc[-1] / m_prices['Or'].iloc[-1]):.2f}")
    if "SPX Equal-Weight" in m_prices and spx_df is not None:
        diff_b = ((m_prices["SPX Equal-Weight"].iloc[-1]/m_prices["SPX Equal-Weight"].iloc[-20]-1) - (spx_df['Close'].iloc[-1]/spx_df['Close'].iloc[-20]-1))*100
        r2.metric("Breadth (RSP vs SPX 1 mois)", f"{diff_b:+.2f}%")

    st.markdown("### 📅 Calendrier Économique en Direct (Investing.com)")
    components.html("""<iframe src="https://investing.com" width="100%" height="450" frameborder="0" allowtransparency="true"></iframe>""", height=460)

with tab4:
    st.subheader(f"Analyse Fondamentale & Valorisation : {custom_ticker}")
    try:
        info = yf.Ticker(custom_ticker).info
        if info and 'shortName' in info:
            st.markdown(f"### 🏢 {info.get('shortName')} ({info.get('sector', 'N/A')})")
            f1, f2, f3 = st.columns(3)
            with f1:
                st.write(f"- P/E : `{info.get('trailingPE', 'N/A')}` | Forward P/E : `{info.get('forwardPE', 'N/A')}`")
                st.write(f"- PEG : `{info.get('pegRatio', 'N/A')}` | EV/EBITDA : `{info.get('enterpriseToEbitda', 'N/A')}`")
            with f2:
                roe, pm = info.get('returnOnEquity'), info.get('profitMargins')
                st.write(f"- ROE : `{f'{roe*100:.1f}%' if roe else 'N/A'}` | Marge Nette : `{f'{pm*100:.1f}%' if pm else 'N/A'}`")
                st.write(f"- Dette/Capitaux : `{info.get('debtToEquity', 'N/A')}`")
            with f3:
                rg, fcf = info.get('revenueGrowth'), info.get('freeCashflow')
                st.write(f"- Croissance CA : `{f'{rg*100:+.1f}%' if rg else 'N/A'}`")
                st.write(f"- Free Cash Flow : `{f'{fcf/1e9:.2f} Md$' if fcf else 'N/A'}`")
                st.write(f"- Consensus : **{info.get('recommendationKey', 'N/A').upper()}** (Cible : `{info.get('targetMeanPrice', 'N/A')}`)")
        else:
            st.warning("Données fondamentales disponibles uniquement pour les actions (ex: NVDA, AAPL, MC.PA).")
    except Exception as e:
        st.error(f"Erreur : {e}")

with tab5:
    st.subheader(f"Quantitatif & Gestion du Risque ({custom_ticker})")
    df_q = get_market_data(custom_ticker)
    if df_q is not None:
        lq = df_q.iloc[-1]
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Z-Score (MM50)", f"{lq['Z_Score']:+.2f} σ")
        q2.metric("Volatilité (20j)", f"{lq['Vol_Realisee']:.1f} %")
        q3.metric("Drawdown Max (1an)", f"{df_q['Drawdown'].min():.1f} %")
        q4.metric("ATR (14j)", f"{lq['ATR']:.2f}")

        st.divider()
        rc1, rc2, rc3 = st.columns(3)
        capital = rc1.number_input("Capital (€/$)", min_value=100, value=10000, step=500)
        risque_pct = rc2.slider("Risque max/trade (%)", 0.5, 5.0, 1.0, 0.5)
        mult_atr = rc3.selectbox("Stop-Loss (x ATR)", [1.5, 2.0, 2.5, 3.0], index=1)

        entree = lq['Close']
        dist = lq['ATR'] * mult_atr
        risque_eur = capital * (risque_pct / 100)
        unites = risque_eur / dist if dist > 0 else 0
        st.info(f"**Plan {custom_ticker} :** Entrée `{entree:.2f}` | Stop-Loss `{(entree-dist):.2f}` | TP1 (1:2) `{(entree+dist*2):.2f}` | TP2 (1:3) `{(entree+dist*3):.2f}` | Taille : `{unites:.2f}` titres (`{(unites*entree):.2f} €/$`)")


