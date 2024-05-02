############################################################################################################
# S&PStock Bullishness Sentiment Analysis
###########################§################################################################################

import streamlit as st
import yfinance as yf
from gnews import GNews 
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.graph_objects as go




############################################################################################################
# Functions defnitons                                                                                      
############################################################################################################
@st.cache_data()
def get_stock_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        return round(data['Close'][0], 2)
    except:
        return "data not available"
@st.cache_data()
def get_stock_returns(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1mo")
        return round(data['Close'][-1] - data['Open'][0], 2)
    except:
        return "data not available"
@st.cache_data()
def get_stock_series(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1mo")
        return data['Close']
    except:
        return "data not available"

@st.cache_data()
def get_stock_news(com):
    google_news = GNews(period="7d", max_results=10)
    df = google_news.get_news(company)

    url_data = []
    for i in range(0,len(df)):
        url_data.append(df[i]["url"])
        
    title_data = []
    for i in range(0,len(df)):
        title_data.append(df[i]["title"])
        
    published_date_data = []
    for i in range(0,len(df)):
        date_string = df[i]["published date"]
        date_format = "%a, %d %b %Y %H:%M:%S %Z"
        date_object = datetime.strptime(date_string, date_format)
        new_date_format = "%d/%m/%Y"
        new_date = date_object.strftime(new_date_format)
        date_object = datetime.strptime(new_date, new_date_format)
        published_date_data.append(date_object)

    publisher_data = []
    for i in range(0,len(df)):
        publisher_data.append(df[i]["publisher"]["title"])

    df = pd.DataFrame({"Title": title_data, "URL" : url_data, "Date" : published_date_data, "Publisher" : publisher_data})

    global progress_bar
    progress_bar = st.progress(0)
    text_list = []
    for i in range(0, len(df)):
        article = Article(df["URL"][i])
        try:
            article.download()
            article.parse()
            text = article.text
            text_list.append(text)
            progress_bar.progress(i/len(df))    
        except:
            text_list.append("not_found")
            progress_bar.progress(i/len(df))

    progress_bar.empty()
    df["Text"] = text_list
    news = df[df["Text"] != "not_found"]
    return news


@st.cache_data()
def analyze_news_sentiment(news):
    # Load the pre-trained sentiment analysis model
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Define a function to preprocess the text
    def preprocess_text(text):
        # Tokenize the text
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Return the preprocessed inputs
        return inputs

    # Define a function to predict the sentiment
    def predict_sentiment(inputs):
        # Make predictions
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).detach().numpy()[0]

        # Get the predicted sentiment label
        labels = ["negative", "neutral", "positive"]
        predicted_label = labels[probabilities.argmax()]

        # Return the predicted sentiment label and probabilities
        return predicted_label, probabilities

    # Analyze the sentiment of each news article
    sentiment_data = []
    counter = 0
    for article in news["Text"]:
        # Preprocess the text
        inputs = preprocess_text(article)

        # Predict the sentiment
        sentiment_label, sentiment_probabilities = predict_sentiment(inputs)

        # Append the sentiment label and probabilities to the sentiment data
        sentiment_data.append((sentiment_label, sentiment_probabilities))

    # Add the sentiment data to the news dataframe
    news["Sentiment"] = sentiment_data


    # Return the news dataframe with sentiment analysis results
    return news

@st.cache_data()
def bullishness(news_analyzed):
    positive = 0
    negative = 0
    neutral = 0 
    for i in range(0, len(news_analyzed)):
        if news_analyzed["Sentiment"].iloc[i][0] == "positive":
            positive += 1
        elif news_analyzed["Sentiment"].iloc[i][0] == "negative":
            negative += 1
        else:
            neutral += 1

    bullishness = round((positive - negative) / (positive + negative + neutral) *100, 2)
    return bullishness


smi = pd.read_excel("SMI.xlsx") # Load the SMI Companies

############################################################################################################
# Streamlit App
###########################################################################################################
st.set_page_config(page_title="SMI Stocks Bullishness Index", page_icon="📈", layout="wide")

col1, col2 = st.columns([1,3])

with col1:
    st.image("SMI_Stocks_Bullishness.png", use_column_width=True)
with col2:
    st.title('SMI Stocks Bullishness Index')
    st.write("This app is a Stock Bullishness Sentiment Analysis tool that can be used to analyze and predict SMI stocks returns. The information provided in this app is for informational purposes only and should not be considered as financial advice.")

    options = smi["Company"].tolist()
    company = st.selectbox("Select the company to be inspected", options)

    ticker = smi[smi["Company"] == company]["Ticker"].values[0]
    st.write(f"The current price of {company} ({ticker}) is {get_stock_current_price(ticker)} CHF")

    returns = get_stock_returns(ticker)
    color = "green" if returns >= 0 else "red"
    st.markdown(f"The returns of {ticker} in the last 30 days is <span style='color: {color}; font-weight: bold;'>{returns}%</span>", unsafe_allow_html=True)

col1, col2 = st.columns([1,1])
with col1.container():
    data = get_stock_series(ticker)

    fig = go.Figure(data=go.Scatter(x=data.index, y=data.values))
    fig.update_layout(autosize=False, height=400, xaxis_title = "Date", yaxis_title = "USD", title="1 Month Stock Development")  # Change this to your desired height
    st.plotly_chart(fig, use_container_width=True)

with col2:
    news = get_stock_news(company)
    news_analyzed = analyze_news_sentiment(news)
    bullishness_sentiment = bullishness(news_analyzed)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bullishness_sentiment,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [-100, 100]}},
        ))
    fig.update_layout(title="Current Bullishness Index")
    st.plotly_chart(fig, use_container_width=True)

