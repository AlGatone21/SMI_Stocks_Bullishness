############################################################################################################
# S&PStock Bullishness Sentiment Analysis
###########################§################################################################################

import streamlit as st
import yfinance as yf
from gnews import GNews
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta
from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import requests
from lxml import etree
import feedparser
import pickle
from statsmodels.iolib.smpickle import load_pickle




############################################################################################################
# Functions defnitons                                                                                      #
############################################################################################################

def search_news(source, start_date, end_date, max_items=100):
    url = f"https://news.google.com/rss/search?q=source:{source}%20after:{start_date}%20before:{end_date}"
    response = requests.get(url)
    
    # Parse the XML response
    root = etree.fromstring(response.content)

    # Extract the relevant data
    data = []
    for i, item in enumerate(root.findall(".//item")):
        if i >= max_items:
            break
        data.append({
            "title": item.find("title").text,
            "link": item.find("link").text,
            "pubDate": item.find("pubDate").text,
            "source": item.find("source").text,
            "description": item.find("description").text
        })

    return data

# load the current stock price from Yahoo Finance
@st.cache_data()
def get_stock_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        return round(data['Close'][0], 2)
    except:
        return "data not available"

# load the stock returns from Yahoo Finance
@st.cache_data()
def get_stock_returns(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="7d")
        return round(data['Close'][-1] - data['Open'][0], 2)
    except:
        return "data not available"

# load the stock series from Yahoo Finance
@st.cache_data()
def get_stock_series(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="7d")
        return data['Close']
    except:
        return "data not available"

# load the stock volume from Yahoo Finance
@st.cache_data()
def get_stock_volume(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="7d")
        return sum(data['Volume'])
    except:
        return "data not available"

# load the stock volatility from Yahoo Finance
@st.cache_data()
def get_stock_volatility(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="7d")
        return round(data['High'].std(), 2)
    except:
        return "data not available"

# load the stock news from Google News
@st.cache_data()
def get_stock_news():

    # Get today's date
    today = datetime.today()
    # Calculate the date seven days ago
    seven_days_ago = today - timedelta(days=7)
    # Format the dates as strings
    today_str = today.strftime("%Y-%m-%d")
    seven_days_ago_str = seven_days_ago.strftime("%Y-%m-%d")

    # Get the news data from Google News
    df = []
    for source in ["Neue Zürcher Zeitung Wirtschaft", "Neue Zürcher Zeitung Finanz", "Inside Paradeplatz", "Finews.ch", "SRF Wirtschaft"]:
        response = search_news(source, seven_days_ago_str, today_str)
        for i in response:
            df.append(i)

    # get the url
    url_data = []
    for i in range(0,len(df)):
        url_data.append(df[i]["link"])

    # get the title   
    title_data = []
    for i in range(0,len(df)):
        title_data.append(df[i]["title"])

    # Convert the published date to a datetime object    
    published_date_data = []
    for i in range(0,len(df)):
        date_string = df[i]["pubDate"]
        date_format = "%a, %d %b %Y %H:%M:%S %Z"
        date_object = datetime.strptime(date_string, date_format)
        new_date_format = "%d/%m/%Y"
        new_date = date_object.strftime(new_date_format)
        date_object = datetime.strptime(new_date, new_date_format)
        published_date_data.append(date_object)

    # get the publisher
    publisher_data = []
    for i in range(0,len(df)):
        publisher_data.append(df[i]["source"])

    # get the description
    description_data = []
    for i in range(0,len(df)):
        description_data.append(df[i]["description"])

    # Create a dataframe with the news data
    df = pd.DataFrame({"Title": title_data, "URL" : url_data, "Date" : published_date_data, "Publisher" : publisher_data, "Description" : description_data})

    text_list = []
    for i in range(0, len(df)):
        article = Article(df["URL"][i])
        try:
            article.download()
            article.parse()
            text = article.text
            text_list.append(text)   
        except:
            text_list.append("not_found")

    df["Text"] = text_list
    news = df[df["Text"] != "not_found"]
    return news


@st.cache_data()
def analyze_news_sentiment(news):

    pipe = pipeline("text-classification", model="scherrmann/GermanFinBert_SC_Sentiment")

    # Analyze the sentiment of each news article
    sentiment_data = []
    for article in news["Text"]:
        try:
            sentiment = pipe(article[:512])
            sentiment_data.append(sentiment[0]["label"])
        except:
            sentiment_data.append("not_found")

    # Add the sentiment data to the news dataframe
    news["Sentiment"] = sentiment_data
    return news

@st.cache_data()
def bullishness(news_analyzed):
    positive = 0
    negative = 0
    neutral = 0 
    for i in range(0, len(news_analyzed)):
        if news_analyzed["Sentiment"].iloc[i] == "Positiv":
            positive += 1
        elif news_analyzed["Sentiment"].iloc[i] == "Negativ":
            negative += 1
        else:
            neutral += 1

    try:
        bullishness = round((positive - negative) / (positive + negative + neutral) *100, 2)
    except:
        bullishness = 0
    return bullishness

# Load the model
model = load_pickle('linear_model.pickle')

@st.cache_data()
def predict_stock_price(open, sentiment, volume, volatility):
    return model.predict([[1, open, sentiment, volume, volatility]])[0]

@st.cache_data()
def predict_lower_limit(open, sentiment, volume, volatility):
    return model.conf_int()[0]["const"] + model.conf_int()[0]["Open"] * open + model.conf_int()[0]["Sentiment_Score_t1"] * sentiment + model.conf_int()[0]["Volume_t1"] * volume + model.conf_int()[0]["Volatility_t1"] * volatility

@st.cache_data()
def predict_upper_limit(open, sentiment, volume, volatility):
    return model.conf_int()[1]["const"] + model.conf_int()[1]["Open"] * open + model.conf_int()[1]["Sentiment_Score_t1"] * sentiment + model.conf_int()[1]["Volume_t1"] * volume + model.conf_int()[1]["Volatility_t1"] * volatility

smi = pd.read_excel("SMI.xlsx") # Load the SMI Companies

############################################################################################################
# Streamlit App
###########################################################################################################

def app():
    if 'accepted_disclaimer' not in st.session_state:
        st.session_state['accepted_disclaimer'] = False

    if not st.session_state['accepted_disclaimer']:
        # Disclaimer page
        col1, col2 = st.columns([1,3])
        with col1:
            st.image("SMI_Stocks_Bullishness.png", use_column_width=True)
        with col2:
            st.title('Disclaimer')
            st.write("This app is a Stock Bullishness Sentiment Analysis tool that can be used to analyze and predict SMI stocks returns. The information provided in this app is for informational purposes only and should not be considered as financial advice.")
            if st.button('I Understand and Accept'):
                st.session_state['accepted_disclaimer'] = True
    else:
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
            rgb = mcolors.to_rgb(color)  # Convert the color name to RGB values
            red, green, blue = [int(255 * x) for x in rgb]  # Scale the RGB values to the range 0-255
            st.markdown(f"The return of {ticker} for the last week is <span style='color: {color}; font-weight: bold;'>{returns}%</span>", unsafe_allow_html=True)

        col1, col2 = st.columns([1,1])
        with col1.container():
            data = get_stock_series(ticker)

            # Create a plot of the stock development
            fig = go.Figure(data=go.Scatter(x=data.index, y=data.values, fill='tozeroy', fillcolor=f'rgba({red},{green},{blue},0.05)', line=dict(color=color)))
            fig.update_layout(autosize=False, height=400, xaxis_title = "Date", yaxis_title = "CHF", title="1 Week Stock Development")  # Change this to your desired height
            fig.update_yaxes(range=[min(data.values)*0.9, max(data.values)*1.1])  # Adjust y-axis to the values of the series
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # create a gauge plot with the bullishness sentiment
            news = get_stock_news()
            news_analyzed = analyze_news_sentiment(news)
            bullishness_sentiment = bullishness(news_analyzed)

            color2 = "green" if bullishness_sentiment >= 0 else "red"

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bullishness_sentiment,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [-100, 100]},
                        'bar': {'color': color2}}))
            fig.update_layout(title="Current SMI Bullishness Index")
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("## Stock Prediction")
        open_price = get_stock_current_price(ticker)
        volume = get_stock_volume(ticker)
        volatility = get_stock_volatility(ticker)
        sentiment = bullishness_sentiment/100

        prediction = predict_stock_price(open_price, sentiment, volume, volatility)
        return1w = round((prediction - open_price) / open_price * 100, 2)
        color3 = "green" if return1w >= 0 else "red"
        rgb3 = mcolors.to_rgb(color3)  # Convert the color name to RGB values
        red3, green3, blue3 = [int(255 * x) for x in rgb3]  # Scale the RGB values to the range 0-255        
        st.markdown(f"The predicted 1 week stock price for {ticker} is {round(prediction, 2)} CHF, which implies a predicted 1 week return of  <span style='color: {color3}; font-weight: bold;'>{return1w}%</span>", unsafe_allow_html=True)

        # Calculate the upper and lower confidence interval boundaries
        upper_limit = predict_upper_limit(open_price, sentiment, volume, volatility)
        lower_limit = predict_lower_limit(open_price, sentiment, volume, volatility)

        # Update the series with the 1 week target price
        target_date = datetime.today() + timedelta(days=7)
        data.loc[target_date] = prediction
        

        # Create a plot of the updated stock development
        fig = go.Figure()

        # Add the main line
        fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', name='Prediction',
                                line=dict(color=color3)))

        # Add the upper confidence interval boundary
        data_upper = data.copy()
        data_upper.loc[target_date] = upper_limit
        fig.add_trace(go.Scatter(x=data_upper.index, y=data_upper.values, mode='lines', name='Upper Limit',
                                line=dict(color='rgba(0,0,255,0.4)', dash='dash')))

        # Add the lower confidence interval boundary
        data_lower = data.copy()
        data_lower.loc[target_date] = lower_limit
        fig.add_trace(go.Scatter(x=data_lower.index, y=data_lower.values, mode='lines', name='Lower Limit',
                                line=dict(color='rgba(255,0,0,0.4)', dash='dash')))

        fig.update_layout(autosize=False, height=400, xaxis_title = "Date", yaxis_title = "CHF", title="1 Week Target Price")
        fig.update_yaxes(range=[min(data.values)*0.9, max(data.values)*1.1])
        st.plotly_chart(fig, use_container_width=True)
        
        if st.checkbox("Show News"):
            st.write("## Recent News")

            for index, row in news.iterrows():
                st.write(f"### {row['Title']}")
                st.write(f"Published on {row['Date']} by {row['Publisher']}")
                st.markdown(row['Description'], unsafe_allow_html=True)
                st.write("---")

if __name__ == '__main__':
    app()