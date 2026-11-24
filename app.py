import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import requests
from groq import Groq
from google import genai

st.set_page_config(page_title="Terminal Pro Marchés - Institutionnel", layout="wide")
st.title("🏛️ Terminal Pro d'Analyse des Marchés (Multi-Piliers)")

# ==========================================
# BARRE LATÉRALE : API, ACTIF & SENTIMENT
# ==========================================
st.sidebar.header("⚙️ Trio IA 100 % Gratuit")
groq_api_key = st.sidebar.text_input("1. Clé Groq (Prioritaire - gsk_...)", type="password")
gemini_api_key = st.sidebar.text_input("2. Clé Gemini (Renfort 1 - AIza...)", type="password")
openrouter_api_key = st.sidebar.text_input("3. Clé OpenRouter (Renfort 2 - sk-or-...)", type="password")
st.sidebar.markdown("[Groq](https://groq.com) | [Gemini](https://ai.google.dev) | [OpenRouter](https://openrouter.ai)")

st.sidebar.divider()
st.sidebar.subheader("🔍 Actif personnalisé (Onglets 2, 4 & 5)")
custom_ticker = st.sidebar.text_input("Ticker Yahoo Finance (ex: NVDA, MC.PA, BTC-USD)", value="NVDA").strip().upper()

st.sidebar.divider()
st.sidebar.subheader("🧭 Données Sentiment & Valo (Injectées à l'IA)")
pmi_global = st.sidebar.slider("PMI Manufacturier Mondial", 40.0, 60.0, 50.2, 0.1)
cape_shiller = st.sidebar.number_input("CAPE de Shiller (S&P 500)", value=35.0, step=0.5)
buffett_ind = st.sidebar.number_input("Indicateur de Buffett (%)", value=195, step=5)
put_call = st.sidebar.slider("Ratio Put/Call (CBOE)", 0.40, 1.50, 0.85, 0.05)
aaii_sentiment = st.sidebar.selectbox("Enquête AAII / Sentix", ["Neutre", "Euphorie extrême (>50% Bulls)", "Panique extrême (>50% Bears)"])
capitulation_ms = st.sidebar.checkbox("Signal de Capitulation extrême actif")

# ==========================================
# FONCTIONS DE DONNÉES & CALCULS TECHNIQUES
# ==========================================
@st.cache_data(ttl=600)
def get_market_data(ticker, period="3y"):
    if not ticker:
        return None
    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        return None
    df = df.dropna(subset=['Close'])
    if len(df) < 60:
        return None

    close, high, low = df['Close'], df['High'], df['Low']

    # Moyennes Mobiles
    df['MM20'] = close.rolling(20).mean()
    df['MM50'] = close.rolling(50).mean()
    df['MM100'] = close.rolling(100).mean()
    df['MM200'] = close.rolling(200).mean()
    df['EMA21'] = close.ewm(span=21, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Stochastique
    low14, high14 = low.rolling(14).min(), high.rolling(14).max()
    df['Stoch_K'] = 100 * ((close - low14) / (high14 - low14).replace(0, np.nan))
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # Bandes de Bollinger & Squeeze
    std20 = close.rolling(20).std()
    df['BB_Upper'] = df['MM20'] + (std20 * 2)
    df['BB_Lower'] = df['MM20'] - (std20 * 2)
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['MM20'].replace(0, np.nan)) * 100

    # ATR & ADX
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    up_move, down_move = high - high.shift(), low.shift() - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14).mean() / df['ATR'].replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14).mean() / df['ATR'].replace(0, np.nan))
    df['ADX'] = (100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)).ewm(alpha=1/14).mean()

    # Ichimoku complet (Tenkan, Kijun, Nuage)
    df['Tenkan'] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    df['Kijun'] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['Senkou_B'] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    # Momentum & Courbe de Coppock
    df['Momentum_10'] = close - close.shift(10)
    monthly_close = close.resample('ME').last()
    roc14 = monthly_close.pct_change(14) * 100
    roc11 = monthly_close.pct_change(11) * 100
    weights = np.arange(1, 11)
    coppock_m = (roc14 + roc11).rolling(10).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    df['Coppock'] = coppock_m.reindex(df.index, method='ffill')

    # Quantitatif & Risque
    df['Z_Score'] = (close - df['MM50']) / close.rolling(50).std().replace(0, np.nan)
    daily_ret = close.pct_change()
    df['Vol_Realisee'] = daily_ret.rolling(20).std() * np.sqrt(252) * 100
    df['Drawdown'] = ((close - close.cummax()) / close.cummax().replace(0, np.nan)) * 100
    df['Sharpe_1Y'] = ((daily_ret.rolling(252).mean() * 252) - 0.04) / (daily_ret.rolling(252).std().replace(0, np.nan) * np.sqrt(252))
    return df


def format_earnings_date(stock):
    try:
        cal = stock.calendar
        if cal is None:
            return "Non communiquée"
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed:
                val = ed[0] if isinstance(ed, (list, tuple, pd.Series, pd.Index)) else ed
                return pd.to_datetime(val).strftime("%d/%m/%Y")
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                row = cal.loc["Earnings Date"].dropna()
                if not row.empty:
                    return pd.to_datetime(row.iloc[0]).strftime("%d/%m/%Y")
            if "Earnings Date" in cal.columns:
                col = cal["Earnings Date"].dropna()
                if not col.empty:
                    return pd.to_datetime(col.iloc[0]).strftime("%d/%m/%Y")
    except Exception:
        pass
    return "Non communiquée"


@st.cache_data(ttl=900)
def get_news():
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MarketTerminalBot/1.0)"}
    urls = [
        "https://news.google.com/rss/search?q=stock+market+economy+OR+macro&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=bourse+economie+marches+financiers&hl=fr&gl=FR&ceid=FR:fr",
        "https://news.google.com/rss/search?q=earnings+federal+reserve+oil+geopolitics&hl=en-US&gl=US&ceid=US:en"
    ]
    titres = []
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:3]:
                if getattr(entry, "title", None) and entry.title not in titres:
                    titres.append(entry.title)
        except Exception:
            pass
    return titres[:9]


# ==========================================
# CHARGEMENT DES DONNÉES MARCHÉS & MACRO
# ==========================================
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
vix_df = get_market_data("^VIX", period="6mo")
vix_val = float(vix_df['Close'].iloc[-1]) if vix_df is not None else 20.0

tnx_df = get_market_data("^TNX", period="6mo")
tnx_val = float(tnx_df['Close'].iloc[-1]) if tnx_df is not None else 4.2

wti_df = get_market_data("CL=F", period="6mo")
wti_val = float(wti_df['Close'].iloc[-1]) if wti_df is not None else 70.0

rsp_df = get_market_data("RSP", period="6mo")
breadth_ratio = "Indisponible"
if rsp_df is not None and data_cache["S&P 500 (USA)"] is not None and len(rsp_df) > 21 and len(data_cache["S&P 500 (USA)"]) > 21:
    perf_rsp = (rsp_df['Close'].iloc[-1] / rsp_df['Close'].iloc[-21] - 1) * 100
    perf_spx = (data_cache["S&P 500 (USA)"]['Close'].iloc[-1] / data_cache["S&P 500 (USA)"]['Close'].iloc[-21] - 1) * 100
    diff_breadth = perf_rsp - perf_spx
    breadth_ratio = f"{diff_breadth:+.2f}% ({'Large/Sain' if diff_breadth > 0 else 'Concentré'})"

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 1. Convergence & IA", "📈 2. Technique & TV", "🌍 3. Macro & Structure", "🏢 4. Valorisation & Fondamental", "🛡️ 5. Quant & Risque"
])

# ==========================================
# ONGLET 1 : CONVERGENCE & ANALYSE IA
# ==========================================
with tab1:
    st.subheader("1. Synthèse Directionnelle des 4 Marchés")
    cols = st.columns(4)
    score_convergence = 5.0 + (1.5 if vix_val < 18 else (-2.0 if vix_val > 25 else 0))
    resume_tech_ia = []

    for i, (nom, _) in enumerate(actifs_principaux.items()):
        df = data_cache[nom]
        with cols[i]:
            if df is not None and len(df) >= 2:
                last, prev = df.iloc[-1], df.iloc[-2]
                var = ((last['Close'] - prev['Close']) / prev['Close']) * 100
                above_cloud = last['Close'] > max(last['Senkou_A'], last['Senkou_B']) if pd.notna(last['Senkou_A']) and pd.notna(last['Senkou_B']) else True
                pts = sum([
                    bool(last['Close'] > last['MM50']) if pd.notna(last['MM50']) else False,
                    bool(last['Close'] > last['MM200']) if pd.notna(last['MM200']) else False,
                    bool(last['MACD'] > last['MACD_Signal']) if pd.notna(last['MACD']) and pd.notna(last['MACD_Signal']) else False,
                    bool(45 < last['RSI'] < 70) if pd.notna(last['RSI']) else False,
                    bool(above_cloud)
                ])
                sig = "🟢 HAUSSIER" if pts >= 4 else ("🔴 BAISSIER" if pts <= 1 else "🟡 NEUTRE")
                score_convergence += 0.6 if pts >= 4 else (-0.6 if pts <= 1 else 0)
                resume_tech_ia.append(
                    f"{nom}: {sig} (Prix: {last['Close']:.2f}, RSI: {last['RSI']:.1f}, ADX: {last['ADX']:.1f}, Coppock: {last['Coppock']:.1f}, Nuage Ichimoku: {'Au-dessus' if above_cloud else 'Sous/Dans'})"
                )
                st.metric(label=nom, value=f"{last['Close']:.2f}", delta=f"{var:+.2f}%")
                st.write(f"**Direction :** {sig}")
            else:
                st.warning(f"{nom} : Données indisponibles")

    score_convergence = max(0.0, min(10.0, score_convergence))
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score Convergence Global", f"{score_convergence:.1f} / 10")
    regime_vol = "Calme (Risk-On) 🟢" if vix_val < 18 else ("Stressé (Risk-Off) 🔴" if vix_val > 25 else "Modéré 🟡")
    c2.metric("VIX (Indice de Peur)", f"{vix_val:.2f}", delta=regime_vol, delta_color="off")
    c3.metric("Taux 10 ans US (^TNX)", f"{tnx_val:.2f} %")
    c4.metric("Breadth S&P (RSP vs SPX 1M)", breadth_ratio)

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
                    hist = stock.history(period="5d").dropna(subset=['Close'])
                    if hist is not None and len(hist) >= 2:
                        var_ent = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
                        date_res = format_earnings_date(stock)
                        st.write(f"{'📈' if var_ent >= 0 else '📉'} **{nom_ent}** ({var_ent:+.2f}%) | Résultats : `{date_res}`")
                        resume_entreprises.append(f"{nom_ent} ({zone}): {var_ent:+.2f}%, résultats le {date_res}")
                    else:
                        st.write(f"• **{nom_ent}** : Historique insuffisant")
                except Exception:
                    st.write(f"• **{nom_ent}** : Indisponible")

    st.divider()
    st.subheader("3. Diagnostic Stratégique Triple IA (Groq • Gemini • OpenRouter)")
    news_list = get_news()
    with st.expander("📰 Voir les dernières dépêches Macro, Conflits & Entreprises"):
        if news_list:
            for n in news_list:
                st.write(f"- {n}")
        else:
            st.write("Aucune dépêche récupérée.")

    if groq_api_key or gemini_api_key or openrouter_api_key:
        if st.button("⚡ Lancer l'analyse institutionnelle complète"):
            with st.spinner("Interrogation du trio IA (Groq ➔ Gemini ➔ OpenRouter)..."):
                prompt = f"""
                Agis comme un stratège macro et quantitatif institutionnel. Analyse l'ensemble des indicateurs suivants :

                1. TECHNIQUE & DIRECTION DES 4 MARCHÉS :
                {chr(10).join(resume_tech_ia)}
                Score de convergence : {score_convergence:.1f}/10

                2. MACRO, LIQUIDITÉ & STRUCTURE :
                - VIX : {vix_val:.2f} | Taux 10 ans US : {tnx_val:.2f}% | Pétrole WTI : {wti_val:.2f}$
                - Breadth S&P 500 (RSP vs SPX sur 1 mois) : {breadth_ratio}
                - PMI Manufacturier Mondial : {pmi_global}

                3. SENTIMENT & VALORISATION LONG TERME :
                - CAPE de Shiller : {cape_shiller} | Indicateur de Buffett : {buffett_ind}%
                - Ratio Put/Call CBOE : {put_call} | Enquête AAII/Sentix : {aaii_sentiment} | Signal Capitulation : {'Oui' if capitulation_ms else 'Non'}

                4. LEADERS & ACTUALITÉS :
                {chr(10).join(resume_entreprises)}
                Dépêches : {chr(10).join(news_list) if news_list else "Aucune"}

                Fournis une synthèse structurée en 4 blocs clairs :
                1. **Direction & Momentum Technique** (Tendances MM50/200, Ichimoku, Coppock, ADX).
                2. **Régime Macro & Liquidité** (Impact des taux 10 ans, pétrole, PMI et largeur de marché Breadth).
                3. **Sentiment & Valorisation** (Excès contrariens AAII/Put-Call vs cherté structurelle Buffett/CAPE).
                4. **Biais Directionnel Global par Zone** (USA, Europe, Asie, EUR/USD — à titre purement informatif, sans conseil d'investissement).
                """

                reponse_ia = None
                source_utilisee = ""

                # 1. GROQ : sélection dynamique d'un modèle actif
                if groq_api_key and not reponse_ia:
                    try:
                        client_groq = Groq(api_key=groq_api_key.strip())
                        models_groq = client_groq.models.list()
                        groq_ids = [m.id for m in getattr(models_groq, "data", []) if getattr(m, "id", None)]

                        preferred_groq = [
                            "llama-3.3-70b-versatile",
                            "llama-3.1-8b-instant",
                            "llama4-scout-17b-16e-instruct",
                        ]
                        chosen_groq = next((m for m in preferred_groq if m in groq_ids), None)
                        if not chosen_groq and groq_ids:
                            chosen_groq = next(
                                (m for m in groq_ids if not any(k in m.lower() for k in ["whisper", "guard", "tts", "embed"])),
                                groq_ids[0]
                            )

                        if not chosen_groq:
                            raise RuntimeError("Aucun modèle Groq disponible sur ce compte.")

                        chat = client_groq.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model=chosen_groq,
                        )
                        reponse_ia = chat.choices[0].message.content
                        source_utilisee = f"Groq ({chosen_groq})"
                    except Exception as e:
                        st.warning(f"Groq indisponible ({e}), bascule vers Gemini...")

                # 2. GEMINI : utilisation de gemini-3.8-flash
                if gemini_api_key and not reponse_ia:
                    try:
                        client_gemini = genai.Client(api_key=gemini_api_key.strip())
                        res = client_gemini.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=prompt
                        )
                        reponse_ia = res.text
                        source_utilisee = "Google Gemini 3.8 Flash"
                    except Exception as e:
                        st.warning(f"Gemini indisponible ({e}), bascule vers OpenRouter...")

                # 3. OPENROUTER : sélection dynamique d'un modèle gratuit (:free) disponible
                if openrouter_api_key and not reponse_ia:
                    try:
                        or_key = openrouter_api_key.strip()
                        headers_or = {
                            "Authorization": f"Bearer {or_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://streamlit.io",
                            "X-Title": "Terminal Pro Marches",
                        }
                        models_resp = requests.get(
                            "https://openrouter.ai/api/v1/models",
                            headers={"Authorization": f"Bearer {or_key}"},
                            timeout=15
                        )
                        models_resp.raise_for_status()
                        or_models = models_resp.json().get("data", [])
                        free_models = [m.get("id") for m in or_models if str(m.get("id", "")).endswith(":free")]

                        chosen_or = next(
                            (m for m in free_models if any(k in m.lower() for k in ["llama", "qwen", "gemma", "mistral"])),
                            free_models[0] if free_models else None
                        )
                        if not chosen_or:
                            raise RuntimeError("Aucun modèle ':free' trouvé sur OpenRouter.")

                        res_or = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers=headers_or,
                            json={
                                "model": chosen_or,
                                "messages": [{"role": "user", "content": prompt}]
                            },
                            timeout=30
                        )
                        res_or.raise_for_status()
                        reponse_ia = res_or.json()["choices"][0]["message"]["content"]
                        source_utilisee = f"OpenRouter ({chosen_or})"
                    except Exception as e:
                        st.error(f"Échec OpenRouter : {e}")

                if reponse_ia:
                    st.success(f"Analyse générée via **{source_utilisee}**")
                    st.markdown(reponse_ia)
    else:
        st.info("💡 Renseigne au moins une clé API dans le menu de gauche pour activer le diagnostic IA.")

# ==========================================
# ONGLET 2 : TECHNIQUE & TRADINGVIEW
# ==========================================
with tab2:
    st.subheader("📈 Indicateurs Techniques Complets & Graphique Temps Réel")
    choix_actif = st.selectbox(
        "Choisir un indice principal ou ton actif personnalisé :",
        list(actifs_principaux.keys()) + [f"Personnalisé ({custom_ticker})"]
    )

    if "Personnalisé" in choix_actif:
        yf_sym, tv_sym = custom_ticker, custom_ticker
        df_tech = get_market_data(yf_sym)
    else:
        yf_sym, tv_sym = actifs_principaux[choix_actif]
        df_tech = data_cache[choix_actif]

    tv_html = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width": "100%", "height": 480, "symbol": "{tv_sym}",
        "interval": "D", "timezone": "Europe/Paris", "theme": "dark",
        "style": "1", "locale": "fr", "toolbar_bg": "#f1f3f6",
        "enable_publishing": false, "allow_symbol_change": true,
        "studies": ["MASimple@tv-basicstudies", "IchimokuCloud@tv-basicstudies", "RSI@tv-basicstudies"],
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(tv_html, height=500)

    if df_tech is not None:
        last = df_tech.iloc[-1]
        st.markdown("#### Tableau de bord des 9 Indicateurs Techniques")
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("MM50 / MM200", f"{last['MM50']:.1f} / {last['MM200']:.1f}", "Golden Cross" if last['MM50'] > last['MM200'] else "Death Cross")
        t2.metric("RSI (14)", f"{last['RSI']:.1f}", "Surachat" if last['RSI'] > 70 else ("Survente" if last['RSI'] < 30 else "Neutre"))
        t3.metric("ADX (Force)", f"{last['ADX']:.1f}", "Tendance Forte" if last['ADX'] > 25 else "Range / Faible")
        t4.metric("MACD Hist.", f"{last['MACD_Hist']:+.2f}", "Haussier" if last['MACD_Hist'] > 0 else "Baissier")
        t5.metric("Courbe de Coppock", f"{last['Coppock']:+.2f}", "Retournement +" if last['Coppock'] > 0 else "Négatif")

        t6, t7, t8, t9 = st.columns(4)
        t6.metric("Stochastique %K", f"{last['Stoch_K']:.1f}")
        t7.metric("Momentum (10j)", f"{last['Momentum_10']:+.2f}")
        t8.metric("Largeur Bollinger (Squeeze)", f"{last['BB_Width']:.2f} %")
        nuage_statut = "Au-dessus du Nuage" if pd.notna(last['Senkou_A']) and pd.notna(last['Senkou_B']) and last['Close'] > max(last['Senkou_A'], last['Senkou_B']) else "Sous/Dans le Nuage"
        t9.metric("Ichimoku (Tenkan/Kijun)", f"{last['Tenkan']:.1f} / {last['Kijun']:.1f}", nuage_statut)
    else:
        st.info("Aucune donnée technique disponible pour cet actif.")

# ==========================================
# ONGLET 3 : MACRO, LIQUIDITÉ & STRUCTURE
# ==========================================
with tab3:
    st.subheader("🌍 Indicateurs Macro, Liquidité & Structure de Marché")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Taux 10 ans US (^TNX)", f"{tnx_val:.2f} %")
    m2.metric("Pétrole WTI (CL=F)", f"{wti_val:.2f} $")
    m3.metric("PMI Manufacturier (Saisi)", f"{pmi_global}", "Expansion" if pmi_global >= 50 else "Contraction")
    m4.metric("Breadth S&P (RSP vs SPX)", breadth_ratio)

    st.divider()
    st.markdown("#### Calendrier Économique Temps Réel")
    cal_html = """
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
      {
        "colorTheme": "dark", "isTransparent": false, "width": "100%", "height": "500",
        "locale": "fr", "importanceFilter": "0,1", "currencyFilter": "USD,EUR,JPY,CNY"
      }
      </script>
    </div>
    """
    components.html(cal_html, height=520)

# ==========================================
# ONGLET 4 : VALORISATION & FONDAMENTAL
# ==========================================
with tab4:
    st.subheader("🧮 Valorisation Long Terme & Fondamentaux")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Indicateur de Buffett (Saisi)", f"{buffett_ind} %", "Zone chère >150%" if buffett_ind > 150 else "Modéré", delta_color="inverse")
    v2.metric("CAPE de Shiller (Saisi)", f"{cape_shiller}", "Élevé >30" if cape_shiller > 30 else "Historique", delta_color="inverse")
    v3.metric("Ratio Put/Call CBOE", f"{put_call}")
    v4.metric("Sentiment AAII / Sentix", aaii_sentiment)

    st.divider()
    st.markdown(f"#### Ratios de Valorisation de l'actif sélectionné : `{custom_ticker}`")
    try:
        info = yf.Ticker(custom_ticker).info if custom_ticker else {}
        if info and ('shortName' in info or 'longName' in info):
            nom_affiche = info.get('shortName') or info.get('longName')
            st.write(f"**{nom_affiche}** — Secteur : {info.get('sector', 'N/A')}")
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("PE (Trailing)", f"{info.get('trailingPE'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else "N/A")
            f2.metric("PEG Ratio", f"{info.get('pegRatio'):.2f}" if isinstance(info.get('pegRatio'), (int, float)) else "N/A")
            f3.metric("Price / Book (PB)", f"{info.get('priceToBook'):.2f}" if isinstance(info.get('priceToBook'), (int, float)) else "N/A")
            f4.metric("Price / Sales (PS)", f"{info.get('priceToSalesTrailing12Months'):.2f}" if isinstance(info.get('priceToSalesTrailing12Months'), (int, float)) else "N/A")
            st.write(info.get("longBusinessSummary", "Aucune description disponible."))
        else:
            st.info("Les ratios PE/PEG/PB/PS individuels s'affichent lorsque tu saisis une action ou un ETF compatible dans la barre latérale (ex: NVDA, SPY, MC.PA).")
    except Exception as e:
        st.error(f"Erreur de récupération fondamentale : {e}")

# ==========================================
# ONGLET 5 : QUANTITATIF & RISQUE
# ==========================================
with tab5:
    st.subheader("🛡️ Métriques Quantitatives (Sharpe, Drawdown, Volatilité)")
    matrice_risque = []
    actifs_a_analyser = dict(data_cache)

    custom_df = get_market_data(custom_ticker)
    if custom_df is not None:
        actifs_a_analyser[f"Perso ({custom_ticker})"] = custom_df

    for nom, df in actifs_a_analyser.items():
        if df is not None:
            last = df.iloc[-1]
            matrice_risque.append({
                "Actif": nom,
                "Ratio de Sharpe (1 an)": f"{last['Sharpe_1Y']:.2f}" if pd.notna(last['Sharpe_1Y']) else "N/A",
                "Maximum Drawdown (3 ans)": f"{df['Drawdown'].min():.2f} %",
                "Drawdown Actuel": f"{last['Drawdown']:.2f} %",
                "Volatilité Réalisée (20j)": f"{last['Vol_Realisee']:.2f} %",
                "Z-Score (vs MM50)": f"{last['Z_Score']:+.2f} σ"
            })
    if matrice_risque:
        st.dataframe(pd.DataFrame(matrice_risque), use_container_width=True)
    else:
        st.info("Aucune métrique quantitative disponible pour le moment.")
