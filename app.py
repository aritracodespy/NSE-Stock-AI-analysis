import yfinance as yf
from tradingview_ta import TA_Handler
from ddgs import DDGS
import json
from datetime import datetime
from openai import OpenAI
import logging
import pandas as pd
import mplfinance as mpf
from fpdf import FPDF
import streamlit as st
import concurrent.futures

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_pdf(ticker, name, report_text):
    pdf = FPDF()
    pdf.add_page()

    # Register Unicode font
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", "DejaVuSans.ttf", uni=True)

    # Title
    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(0, 10, f"Equity Research Report: {name} ({ticker})", ln=True, align="C")

    pdf.set_font("DejaVu", "", 10)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%d-%m-%Y')}", ln=True, align="C")
    pdf.ln(6)

    # Body
    pdf.set_font("DejaVu", "", 11)
    pdf.multi_cell(0, 8, report_text)

    return bytes(pdf.output(dest="s"))
# --- 1. REFINED TECHNICAL FUNCTIONS ---

def create_stock_chart(symbol):
    symbol = symbol.strip().upper()

    if symbol.endswith(".NS"):
        symbol = symbol.replace(".NS", "")

    full_symbol = f"{symbol}.NS"
    try:
        # Fetch data
        df = yf.Ticker(full_symbol).history(period="6mo", interval="1d")
        if df.empty:
            logger.warning(f"No data found for symbol: {symbol}")
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Calculate EMAs
        df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

        # Calculate RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        period = 14
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Prepare last 150 candles
        plot_df = df.iloc[-100:].copy()
        plot_df.index.name = "Date"

        # Styling
        mc = mpf.make_marketcolors(
            up='green',
            down='red',
            edge='inherit',
            wick='#fff',
            volume='in',
            ohlc='i'
        )
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            facecolor="#121212",
            gridcolor="#333",
            marketcolors=mc
        )

        # Addplots
        apds = []
        for col in ["EMA10", "EMA20", "EMA50"]:
            if col in plot_df.columns:
                apds.append(mpf.make_addplot(plot_df[col], panel=0, width=1))

        if "RSI" in plot_df.columns:
            apds.extend([
                mpf.make_addplot(plot_df["RSI"], panel=1, color="orange", ylabel="RSI"),
                mpf.make_addplot([60] * len(plot_df), panel=1, color="gray", linestyle="--"),
                mpf.make_addplot([40] * len(plot_df), panel=1, color="gray", linestyle="--"),
            ])

        # Generate plot
        fig, _ = mpf.plot(
            plot_df,
            type="candle",
            style=style,
            addplot=apds,
            volume=False,
            panel_ratios=(3, 1),
            figsize=(14, 8),
            returnfig=True
        )
        return fig

    except Exception as e:
        return None
# --- 2. DATA RETRIEVAL (PARALLELIZED) ---

def fetch_fundamentals(symbol: str, full_symbol: str):
    try:
        ticker = yf.Ticker(full_symbol)
        info = ticker.info
        if not info or len(info) < 5: return None

        # RESTORED: Your exact handpicked fundamental keys
        keys = [
            "marketCap", "bookValue", "trailingPE", "forwardPE",
            "dividendYield", "fiveYearAvgDividendYield",
            "debtToEquity", "currentRatio",
            "returnOnEquity", "returnOnAssets",
            "freeCashflow", "revenueGrowth", "earningsGrowth",
            "profitMargins", "operatingMargins",
            "enterpriseToEbitda", "payoutRatio", "currency"
        ]

        extracted = {k: info.get(k) for k in keys if info.get(k) is not None}
        return {
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sectorKey") or info.get("sector"),
            "industry": info.get("industryKey") or info.get("industry"),
            "summary": info.get("longBusinessSummary"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "metrics": extracted
        }
    except Exception as e:
        logger.error(f"Fundamental fetch error: {e}")
        return None

def fetch_technicals(symbol: str):
    try:
        handler = TA_Handler(symbol=symbol, exchange="NSE", screener="india", interval="1d", timeout=10)
        indicators = handler.get_analysis().indicators

        # RESTORED: Your exact handpicked technical keys
        tech_keys = [
            "RSI", "ADX", "ADX+DI", "ADX-DI", "Mom", "Stoch.K", "Stoch.D",
            "MACD.macd", "MACD.signal", "EMA10", "EMA20", "EMA50", "EMA100", "EMA200",
            "Pivot.M.Classic.S3", "Pivot.M.Classic.S2", "Pivot.M.Classic.S1",
            "Pivot.M.Classic.Middle", "Pivot.M.Classic.R1", "Pivot.M.Classic.R2", "Pivot.M.Classic.R3"
        ]
        return {k: round(float(indicators.get(k)), 2) for k in tech_keys if indicators.get(k) is not None}
    except Exception:
        return {}

def fetch_news(symbol: str, name: str):
    try:
        query = f"{symbol} {name}" if name else symbol
        results = DDGS().news(query, region="in-en", max_results=10)
        return {r.get("date"): r.get("title") for r in results if r.get("date") and r.get("title")}
    except Exception:
        return {}

def fetch_weekly_history(full_symbol: str):
    try:
        df = yf.Ticker(full_symbol).history(period="6mo", interval="1wk")
        return {str(idx.date()): round(float(val), 2) for idx, val in df["Close"].items()}
    except Exception:
        return {}

def get_stock_snapshot(symbol: str):
    clean_symbol = symbol.strip().upper().replace(".NS", "")
    full_symbol = f"{clean_symbol}.NS"

    # PARALLEL EXECUTION
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Step 1: Core Fundamentals (Needed for name in News)
        fund_future = executor.submit(fetch_fundamentals, clean_symbol, full_symbol)
        fund_res = fund_future.result()

        if not fund_res: return None

        # Step 2: Parallelize the rest
        tech_future = executor.submit(fetch_technicals, clean_symbol)
        news_future = executor.submit(fetch_news, clean_symbol, fund_res["name"])
        week_future = executor.submit(fetch_weekly_history, full_symbol)

        return {
            "symbol": clean_symbol,
            "name": fund_res["name"],
            "sector": fund_res["sector"],
            "industry": fund_res["industry"],
            "businessSummary": fund_res["summary"],
            "price": fund_res["price"],
            "fundamental": fund_res["metrics"],
            "technical": tech_future.result(),
            "news": news_future.result(),
            "weekly_close_6mo": week_future.result()
        }

# --- 3. AI & PROMPT LOGIC ---

def openai_ai_api(model, base_url, api_key, prompt):
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {str(e)}"

def format_prompt(stock_data: dict) -> str:
    core_data = {
        "price": stock_data.get("price"),
        "fundamental": stock_data.get("fundamental", {}),
        "technical": stock_data.get("technical", {}),
        "weekly_close_6mo": stock_data.get("weekly_close_6mo", {}),
        "recent_news": stock_data.get("news", {})
    }
    symbol = stock_data.get("symbol")
    name = stock_data.get("name")
    sector = stock_data.get("sector")
    industry = stock_data.get("industry")
    business = stock_data.get("businessSummary")

    prompt = f"""
Act as a SEBI Registered Senior Equity Research Analyst specializing in the Indian Stock Market (NSE).

Analyze {symbol} ({name}), operating in the {sector} sector and {industry} industry.

BUSINESS OVERVIEW:
{business}

========================
STOCK DATA (RAW INPUT)
========================
{json.dumps(core_data, indent=4)}

Analysis Date: {datetime.now().strftime("%d-%m-%Y")}

========================
ANALYSIS INSTRUCTIONS
========================

1. COMPANY SNAPSHOT(under 1-2 sentences)
- Business overview with sector and industry.
- current price and some crucial metrics.

2. FUNDAMENTAL ANALYSIS (India & Sector Context)
- Evaluate valuation metrics (P/E, PEG, EV/EBITDA) relative to typical Indian peers in the {sector} sector.
- Assess balance sheet strength using Debt-to-Equity and Current Ratio.
- Comment on profitability and efficiency using ROE, ROA, Operating & Profit Margins.
- Analyze growth sustainability using Revenue Growth, Earnings Growth, and Free Cash Flow (if available).
- one line summary of fundamental analysis.

3. TECHNICAL ANALYSIS (Multi-Timeframe Logic)
- Determine the primary trend using EMA alignment (10/20/50 vs 100/200).
- Evaluate momentum and exhaustion using RSI, MACD, and ADX.
- Identify important support and resistance zones using Pivot levels.
- Highlight any trend continuation or reversal signals.
- one line summary of technical analysis.

4. PRICE TREND ANALYSIS (6-Month Weekly Closes)
- Analyze the 6-month weekly close price data to determine:
  - Medium-term trend direction (uptrend, downtrend, range-bound)
  - Presence of higher highs / higher lows or distribution patterns
- Comment on whether the current price aligns or conflicts with the medium-term trend.

5. NEWS & EVENT RISK ANALYSIS
- Summarize the overall sentiment from recent news headlines (positive, neutral, negative).
- Identify whether news flow suggests:
  - Fundamental improvement
  - Regulatory / sector risk
  - Short-term volatility triggers
- Do NOT assume news impact on price unless clearly justified.

6. ACTIONABLE TRADING & INVESTMENT VIEW
- Recommendation: Clearly state one of: Buy / Hold / Sell / Avoid / Watch.
- Entry Zone: Provide a realistic price range based on technical structure.
- Stop Loss: Define a strict invalidation level.
- Targets:
  - Target 1: Conservative
  - Target 2: Stretch
- Risk-to-Reward: Explain whether the setup is favorable or not.

7. FINAL VERDICT (Bottom Line)
- Provide a concise summary combining:
  - Fundamental strength or weakness
  - Technical trend quality
  - Medium-term price structure
  - News-driven risk
- Clearly state for whom the stock is suitable (short-term trader / swing trader / long-term investor).

========================
CONSTRAINTS
========================
- Use plain text with markdown bullet points only.
- Do NOT use tables.
- Maintain a professional, cautious, research-oriented tone.
- If any data is missing or inconclusive, explicitly acknowledge it.
- Do NOT provide financial advice or guarantees.
- Always add a short disclaimer for not finantial advice ai generated.
"""

    return prompt

# --- 4. STREAMLIT UI ---

st.set_page_config(page_title="NSE Stock AI", layout="wide")

defaults = {
    "model": "",
    "base_url": "",
    "api_key": ""
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# Sidebar
with st.sidebar:
    st.subheader("AI Configuration")

    st.text_input("Model", key="model", placeholder="Ai Model e.g. GPT-4.o", autocomplete="off")
    st.text_input(
        "Base URL",
        key="base_url",
        placeholder="OpenAI Compatible api base_url. ",
        autocomplete="off"
    )
    st.text_input("API Key", key="api_key", type="password", autocomplete="off")

    if st.button("Save Credentials"):
        if not all([
            st.session_state.model.strip(),
            st.session_state.base_url.strip(),
            st.session_state.api_key.strip()
        ]):
            st.error("All fields are required")
        else:
            st.success("Credentials saved in session")

st.title("🚀 NSE Stock AI Analysis")

with st.expander("About & Legal Disclaimer & Privacy"):
    st.info(
    """
    **About this App** This platform created by A NOOB self taught python developer with help of AI that
    bridges the gap between widely avalable raw market data and actionable insights for the NSE.
    By integrating real-time data from Yahoo Finance and TradingView and Duckduckgo-search with advanced AI, it
    simulates the analytical depth of a Senior Equity Research Analyst.

    **Disclaimer**
    This application is a personal hobby project developed for academic and portfolio purposes to demonstrate the integration of financial data APIs and LLMs avalable openly on puplic domain. The developer is not a SEBI-registered investment advisor, broker, or financial professional just a NOOB developer. This tool is provided "as is" for informational and educational purposes only.\n
    All financial decisions, trades, or investments made by the user are the sole responsibility of the user. The developer of this application expressly disclaim any and all liability for financial losses, damages, or consequences arising from the use of this tool. AI-generated insights are experimental and contain "hallucinations," inaccuracies, or biased data; they should never be the primary basis for investment decisions.\n
    Investing in the NSE/Stock Market involves significant risk. Past performance is not indicative of future results. Users are strongly encouraged to consult with a SEBI-registered financial advisor and perform independent due diligence before committing capital.

    **Privacy & Security** This is a serverless, client-side application. Your **API Key is never saved** on any database
    or server. It is stored only in your browser's temporary session state and is cleared
    once you close the tab.
    """
    )

if not all([
    st.session_state.model,
    st.session_state.base_url,
    st.session_state.api_key
]):
    st.warning("Configure AI credentials from the sidebar to continue")
    st.stop()


ticker_input = st.text_input("Enter NSE Stock Symbol", placeholder="e.g. RELIANCE, TCS", autocomplete="off").upper().strip()

@st.cache_data(show_spinner=False)
def get_data(symbol: str):
    df = yf.Ticker(symbol).history(period="1d")
    if df.empty:
        return None
    return df

if ticker_input:
    # 1. INPUT VALIDATION
    with st.spinner("Validating ticker..."):
        valid_ticker = ticker_input.replace(".NS", "")
        check = get_data(f"{valid_ticker}.NS")
        if check is None:
            st.error(f"Ticker '{ticker_input}' not found on NSE.")
            st.stop()

    # 2. PARALLEL FETCH
    with st.spinner("Analyzing data from multiple sources..."):
        data = get_stock_snapshot(valid_ticker)

    if data:
        # 3. METRICS CARDS
        st.subheader(f"📊 {data['name']} Overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"₹{data['price']}")
        m2.metric("P/E Ratio", f"{data['fundamental'].get('trailingPE', 'N/A')}")
        m3.metric("RSI (1D)", f"{data['technical'].get('RSI', 'N/A')}")
        m4.metric("Market Cap", f"₹{data['fundamental'].get('marketCap', 0)//10**7:,} Cr")

        # 4. CHART
        with st.expander("📈 Technical Chart", expanded=True):
            fig = create_stock_chart(valid_ticker)
            if fig: st.pyplot(fig)
            st.markdown(f"Goto TradingView Chart [TradingView Charts](https://in.tradingview.com/chart/?symbol=NSE%3A{valid_ticker})")
        # 5. AI ANALYSIS
        if st.session_state.api_key:
            with st.spinner("Generating SEBI-grade research report..."):
                prompt = format_prompt(data)
                report = openai_ai_api(st.session_state.model, st.session_state.base_url, st.session_state.api_key, prompt)
                st.markdown("---")
                st.markdown(report)

                pdf_bytes = generate_pdf(valid_ticker, data['name'], report)

                st.download_button(
                    label="📥 Download Research Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"{valid_ticker}_Research_Report.pdf",
                    mime="application/pdf"
                )

        else:
            st.info("Enter API Key in sidebar to generate AI Research Report.")
