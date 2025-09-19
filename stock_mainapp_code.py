import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from transformers import pipeline
import requests
from bs4 import BeautifulSoup



# 🏆 Set Streamlit Theme (Dark Mode)
st.set_page_config(
    page_title="AI Stock Market Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown(
    """
    <style>
    /* Background & Text Styling */
    body {
        background-color: #0B3D91;  /* Slightly lighter dark */
        color: #000000;  /* White text */
    }
    .stApp {
        background-color: #0B3D91; /* Softer dark */
        color: #000000;
    }
    
    /* Title & Headers */
    h1, h2, h3, h4 {
        color: #FAFAFA !important;  /* Bright white */
    }
    
    /* Fix Label Color */
    label {
        color: white !important;  /* Fix input label */
        font-weight: bold;
    }

    /* Input Field */
    .stTextInput>div>div>input {
        background-color: #ffffff !important;  /* Pure white */
        color: #000000 !important;  /* Black text */
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #bbb;
    }

    /* Buttons */
    .stButton>button {
        background-color: #0088CC !important;  /* Bright blue */
        color: white !important;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #0077AA !important;  /* Slightly darker blue */
    }

    /* Subheaders */
    .stMarkdown {
        font-size: 18px;
        font-weight: bold;
        color: #E0E0E0 !important;
    }

    /* Section Backgrounds */
    .stContainer {
        background-color: #2B313A; /* Softer dark */
        padding: 15px;
        border-radius: 10px;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)


# Load FinBERT Model for sentiment analysis
finbert = pipeline("text-classification", model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone")

# Hugging Face API Key (Replace with yours)
HUGGINGFACE_API_KEY = "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"

# Initialize session state to store stock data, news, summary, and forecast
if "stock_data" not in st.session_state:
    st.session_state.stock_data = None
if "news" not in st.session_state:
    st.session_state.news = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "forecast" not in st.session_state:
    st.session_state.forecast = None
if "df" not in st.session_state:
    st.session_state.df = None

# Function to fetch stock price data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")
    return hist


def fetch_stock_news_yahoo(ticker):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

    if response.status_code == 200:
        data = response.json()
        news_items = [
            (item["title"], item["link"]) for item in data.get("news", [])[:5]
        ]  # Extract titles and URLs
        return news_items  # Returning both title and link
    else:
        return [("Error fetching news.", "")]


# Function to summarize news using Hugging Face API
def summarize_news_with_huggingface(news_list):
    API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    
    prompt = f"""
    Summarize the following stock news into a **single paragraph** with **exactly 10 lines**. 
    Your response must be **only the final summary** and also ensure that your response is plain without any bold/italic/other formatting
    , it should be a response where even a 10th grade student can interpret and understand.

    Stock News:
    {'. '.join(news_list)}

    Provide the summary below:
    """
    
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        summary_text = response.json()[0]["generated_text"]
        return clean_summary(summary_text)
    else:
        return "Error summarizing news."

# Function to clean summary text
def clean_summary(text):
    text = text.split("Provide the summary below:")[-1].strip()
    # Extract text after `</think>` if it exists
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text.replace("\n", " ").replace("**", "").replace("_", "").strip()


# Function to analyze news sentiment using FinBERT
def analyze_news_sentiment(news_summary):
    sentiment_result = finbert(news_summary)[0]
    return sentiment_result["label"].lower()


# Revised AI recommendation (only sentiment-based)
def generate_recommendation(news_summary):
    sentiment_result = finbert(news_summary)[0]
    label = sentiment_result["label"].lower()
    score = sentiment_result["score"]

    if label == "positive":
        return f"✅ **BUY** – Sentiment is positive (Confidence: {score:.2f})"
    elif label == "negative":
        return f"❌ **SELL** – Sentiment is negative (Confidence: {score:.2f})"
    else:
        return f"⚖️ **HOLD** – Sentiment is neutral (Confidence: {score:.2f})"


def get_ticker_from_name(company_name):
    """
    Uses Hugging Face's DeepSeek LLM to determine the correct stock ticker.
    If an error occurs, returns the original input.
    """
    API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    prompt = f"What is the stock ticker symbol for the company '{company_name}'? Only return the ticker symbol in uppercase."

    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        ticker = response.json()[0]["generated_text"].strip()
        if len(ticker) <= 5 and ticker.isalpha():  # Basic validation for stock symbols
            return ticker.upper()
    
    return company_name  # Return original input if no valid ticker is found

def fetch_financial_statements(ticker):
    """
    Fetches Income Statement and Balance Sheet for the given stock ticker.
    Formats numbers to a readable financial format with dollar signs and commas.
    """
    stock = yf.Ticker(ticker)
    
    # Fetch financial statements
    income_statement = stock.financials
    balance_sheet = stock.balance_sheet

    def format_financials(df):
        def format_value(x):
            if isinstance(x, (int, float)):
                if abs(x) >= 1e9:
                    return f"${x/1e9:.2f} Billion"
                elif abs(x) >= 1e6:
                    return f"${x/1e6:.2f} Million"
                elif abs(x) >= 1e3:
                    return f"${x/1e3:.2f} Thousand"
                else:
                    return f"${x:.2f}"
            else:
                return x
        return df.applymap(format_value)
    
    return format_financials(income_statement), format_financials(balance_sheet) #, format_financials(cash_flow)

# Load a list of major stock tickers
POPULAR_TICKERS = [
    "AAPL - Apple", "TSLA - Tesla", "MSFT - Microsoft", "GOOGL - Alphabet (Google)",
    "AMZN - Amazon", "META - Meta (Facebook)", "NFLX - Netflix", "NVDA - Nvidia",
    "BRK.A - Berkshire Hathaway", "JPM - JPMorgan Chase", "V - Visa", "DIS - Disney",
    "PYPL - PayPal", "INTC - Intel", "AMD - AMD", "IBM - IBM","CSCO - Cisco", "JNPR - Juniper Networks"]

# Convert dropdown selection into ticker symbol
def extract_ticker(selection):
    return selection.split(" - ")[0]  # Extracts ticker from "AAPL - Apple"


def extract_article_content_bs4(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to extract the main article text based on common tags
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text() for p in paragraphs])

            # Optional: Limit to first 2000 characters (you can adjust)
            return article_text[:2000]
        else:
            return "Failed to fetch article."
    except:
        return "Error fetching content."


# Streamlit UI
st.title("📈 AI Stock Market Assistant")

selected_ticker = st.selectbox("Select a Stock:", POPULAR_TICKERS, index=0)  # Default to AAPL
ticker = extract_ticker(selected_ticker)  # Extract ticker symbol

if st.button("Analyze Stock"):
    st.subheader(f"📊 {ticker} Stock Analysis")
    
    # Fetch Stock Data
    st.session_state.stock_data = fetch_stock_data(ticker)
    
    # Display Stock Price Trend with High & Low Prices
    st.subheader("📊 Stock Price Trend (Last 6 Months)")
    fig = go.Figure()
    
    # Add Closing Price Line
    fig.add_trace(go.Scatter(x=st.session_state.stock_data.index, y=st.session_state.stock_data["Close"], 
                             mode='lines', name='Close Price', line=dict(color='blue')))
    
    # Add High Price Line
    fig.add_trace(go.Scatter(x=st.session_state.stock_data.index, y=st.session_state.stock_data["High"], 
                             mode='lines', name='High Price', line=dict(color='green', dash="dot")))
    
    # Add Low Price Line
    fig.add_trace(go.Scatter(x=st.session_state.stock_data.index, y=st.session_state.stock_data["Low"], 
                             mode='lines', name='Low Price', line=dict(color='red', dash="dot")))
    
    # Update Graph Layout
    fig.update_layout(title=f"{ticker} Stock Price Trend",
                      xaxis_title="Date", yaxis_title="Stock Price",
                      width=600, height=400)
    
    st.plotly_chart(fig)
    

    # Custom Link Styling
    st.markdown(
    	"""
    	<style>
    	a {
        	color: white !important;
        	text-decoration: none;
    	}
    	a:hover {
        	text-decoration: underline;
        	color: #ADD8E6;
    	}
    	</style>
    	""",
    	unsafe_allow_html=True
	)
	
    # Fetch News Data
    st.subheader("📰 Latest Stock News")
    st.session_state.news = fetch_stock_news_yahoo(ticker)
    for i, (headline, link) in enumerate(st.session_state.news, 1):
        st.markdown(f"{i}. [{headline}]({link})")  # Display clickable news links

  
    
    st.markdown(
    	"""
    	<style>
    	/* Styling for paragraph text */
   	 p {
        	color: white;
        	font-size: 16px; /* You can adjust to 16px if you want slightly smaller */
    		}
    	</style>
    	""",
    	unsafe_allow_html=True
	)

    # Summarize News
    st.subheader("🤖 AI News Summary")
    
    full_articles = []
    for headline, link in st.session_state.news:
        content = extract_article_content_bs4(link)
        full_articles.append(content)

    st.session_state.summary = summarize_news_with_huggingface(full_articles)
    st.write(st.session_state.summary)

    # Fetch and Display Financial Statements
    st.subheader("📊 Company Financial Statements")
    income_statement, balance_sheet = fetch_financial_statements(ticker)
    
    with st.expander("📜 Income Statement"):
        st.dataframe(income_statement.style.set_properties(**{"text-align": "left"}))
    
    with st.expander("📊 Balance Sheet"):
        st.dataframe(balance_sheet.style.set_properties(**{"text-align": "left"}))

    # AI Investment Recommendation
    st.subheader("💡 AI Investment Recommendation")
    recommendation = generate_recommendation(st.session_state.summary)
    st.markdown(f"### {recommendation}")