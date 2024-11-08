############################################################################################################
# DISCLAIMER
# This app is a Stock Bullishness Sentiment Analysis tool that can be used to analyze and predict SMI 
# stocks returns. The information provided in this app is for informational purposes only and should 
# not be considered as financial advice.
############################################################################################################

############################################################################################################
# Github Copilot was used to assist develop the code below. The generated code was then modified to fit 
# the requirements. No prompt is available, since the autocompletion function was used to generate the code.
############################################################################################################

############################################################################################################
# SMI Stocks Bullishness Index
###########################§################################################################################

import streamlit as st
import yfinance as yf
from gnews import GNews
from tqdm import tqdm
import pandas as pd
import numpy as np
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

# search for news articles using Google News RSS feed
def search_news(source, start_date, end_date, max_items=100):
    url = f"https://news.google.com/rss/search?q=source:{source}%20after:{start_date}%20before:{end_date}" # Create the URL for the RSS feed
    response = requests.get(url)
    
    # Parse the XML response
    root = etree.fromstring(response.content)

    # Extract the relevant data form the response
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
        data = get_stock_series(ticker)
        returns = (data[-1] - data[0]) / data[0]
        return returns
    except:
        return "data not available"

# load the stock series from Yahoo Finance
@st.cache_data()
def get_stock_series(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d")
        return data['Close']
    except:
        return "data not available"

# load the stock volume from Yahoo Finance
@st.cache_data()
def get_stock_volume(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d")
        return sum(data['Volume'])
    except:
        return "data not available"

# load the stock volatility from Yahoo Finance
@st.cache_data()
def get_stock_volatility(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d")
        data["Return"] = data["Return"] = (data["Close"] - data["Open"]) / data["Open"] # Calculate the daily returns
        return data["Return"].std() # Calculate the standard deviation of the returns
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
    # Load the sentiment analysis model from transformers
    pipe = pipeline("text-classification", model="scherrmann/GermanFinBert_SC_Sentiment")

    # Analyze the sentiment of each news article
    sentiment_data = []
    for article in news["Text"]:
        try:
            sentiment = pipe(article[:512]) # Limit the text to 512 tokens (model requirement)
            sentiment_data.append(sentiment[0]["label"])
        except:
            sentiment_data.append("not_found") # If the sentiment cannot be analyzed, set it to "not_found"

    # Add the sentiment data to the news dataframe
    news["Sentiment"] = sentiment_data
    return news

# Calculate the bullishness sentiment of the news
@st.cache_data()
def bullishness(news_analyzed):
    positive = 0
    negative = 0
    neutral = 0 
    for i in range(0, len(news_analyzed)): # Count the number of positive, negative, and neutral news articles
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

# Load the linear model
model = load_pickle('linear_regr_model.pickle')

# Load the KNN model
knn_model = pickle.load(open("knn.sav", 'rb'))

# Load the Logistic Regression Model
log_reg = pickle.load(open("logistic_regression.sav", 'rb'))

# Load the SVC Model
svc = pickle.load(open("svc.sav", 'rb'))

# Load the Naive Bayes Model from the file
naive_bayes = pickle.load(open("naive_bayes.sav", 'rb'))

# Load the Decision Tree Model
dt = pickle.load(open("decision_tree.sav", 'rb'))

# predict the stock price using the linear model (open + return)
def predict_stock_price(open, sentiment, volume, volatility, returnt1):
    volume = np.log(volume)
    returns = model.predict([[1, sentiment, volume, volatility, returnt1]])[0]
    return open * (1 + returns)

# predict the lower limit of the stock price using the linear model (open + return)
def predict_lower_limit(open, sentiment, volume, volatility, returnt1, alpha = 0.05):
    volume = np.log(volume)
    returns =  model.conf_int(alpha)[0]["const"] + model.conf_int(alpha)[0]["Sentiment_Score_t1"] * sentiment + model.conf_int(alpha)[0]["log_Volume_t1"] * volume + model.conf_int(alpha)[0]["Volatility_t1"] * volatility + model.conf_int(alpha)[0]["Return_t1"] * returnt1
    return open * (1 + returns)

# predict the upper limit of the stock price using the linear model (open + return)
def predict_upper_limit(open, sentiment, volume, volatility, returnt1, alpha = 0.05):
    volume = np.log(volume)
    returns =  model.conf_int(alpha)[1]["const"] + model.conf_int(alpha)[1]["Sentiment_Score_t1"] * sentiment + model.conf_int(alpha)[1]["log_Volume_t1"] * volume + model.conf_int(alpha)[1]["Volatility_t1"] * volatility + model.conf_int(alpha)[1]["Return_t1"] * returnt1
    return open * (1 + returns)

smi = pd.read_excel("SMI.xlsx") # Load the SMI Companies

backtest_data = pd.read_excel("Knn_Backtest.xlsx") # Load the backtest data

# Compute the backtest for the stock
@st.cache_data()
def compute_backtest(df, ticker, start_date, end_date):
    data = df[df["Ticker"] == ticker].copy() # Filter the data for the selected stock
    data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)] # Filter the data for the selected date range
    data.reset_index(drop=True, inplace=True) # Reset the index of the data
    data["Holding_index"] = 0 # Initialize the holding index
    data["Strategy_index"] = 0 # Initialize the strategy index
    data.loc[0,"Holding_index"] = 100 # Set the initial holding index to 100
    data.loc[0,"Strategy_index"] = 100 # Set the initial strategy index to 100
    for i in range(1, len(data)): # Compute the holding and strategy index for each period (week)
        data.loc[i,"Holding_index"] = data.loc[i-1,"Holding_index"] * (1 + data.loc[i,"weekly.returns"])
        data.loc[i,"Strategy_index"] = data.loc[i-1,"Strategy_index"] * (1 + data.loc[i,"strategy_returns"])
    return data # Return the backtest data

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
            st.markdown(f"The return of {ticker} for the last week is <span style='color: {color}; font-weight: bold;'>{round(returns*100,2)}%</span>", unsafe_allow_html=True)

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

            color2 = "green" if bullishness_sentiment >= 0 else "red" # Set the color based on the sentiment

            # Create the gauge plot diplaing the bullishness sentiment of the SMI
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bullishness_sentiment,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [-100, 100]},
                        'bar': {'color': color2}}))
            fig.update_layout(title="Current SMI Bullishness Index")
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("## Stock Prediction")
        open_price = get_stock_current_price(ticker) # Get the current stock price
        volume = get_stock_volume(ticker) # Get the stock volume
        volatility = get_stock_volatility(ticker) # Get the stock volatility
        sentiment = bullishness_sentiment/100 #convert the sentiment to a scale between -1 and 1
        returnt1 = returns
    
        prediction = max(predict_stock_price(open_price, sentiment, volume, volatility, returnt1),0) # Predict the stock price, ensuring it is not negative
        return1w = (prediction - open_price) / open_price # Calculate the predicted return
        color3 = "green" if return1w >= 0 else "red"
        rgb3 = mcolors.to_rgb(color3)  # Convert the color name to RGB values
        red3, green3, blue3 = [int(255 * x) for x in rgb3]  # Scale the RGB values to the range 0-255        
        st.markdown(f"The predicted 1 week stock price for {ticker} is {round(prediction, 2)} CHF, which implies a predicted 1 week return of  <span style='color: {color3}; font-weight: bold;'>{round(return1w*100,2)}%</span>", unsafe_allow_html=True)

        # Calculate the upper and lower confidence interval boundaries
        confidence = st.select_slider("Select the confidence level (%)", options=range(1,100))
        alpha = 1 - confidence / 100
        upper_limit = max(predict_upper_limit(open_price, sentiment, volume, volatility, returnt1, alpha),0)
        lower_limit = max(predict_lower_limit(open_price, sentiment, volume, volatility, returnt1, alpha), 0)

        max_upper_limit = max(predict_upper_limit(open_price, sentiment, volume, volatility, returnt1, 0.01),0) # Calculate the 99% confidence interval
        min_lower_limit = max(predict_lower_limit(open_price, sentiment, volume, volatility, returnt1, 0.01),0) # Calculate the 99% confidence interval

        prediction_volatility = round(((max_upper_limit - min_lower_limit) / prediction) * 100, 2) # Calculate the prediction volatility as the range of the 99% confidence interval divided by the prediction
        
        st.write(f"The {confidence}% confidence interval for the 1 week stock price is between {round(lower_limit, 2)} and {round(upper_limit, 2)} CHF")
        if prediction_volatility > 30: # Display a warning if the prediction volatility is high
            st.markdown("<div style='color: red; font-weight: bold;'>Warning:</div>", unsafe_allow_html=True)
            st.write(f'The prediction volatility is high with {prediction_volatility}% of the predicted price. This indicates a high uncertainty in the prediction.')

        # Update the series with the 1 week target price
        target_date = datetime.today() + timedelta(days=7)
        data.loc[target_date] = prediction
        

        # Create a plot of the updated stock development
        fig = go.Figure()

        # Add the main line
        fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', name='Prediction',
                                line=dict(color=color3), fill='tozeroy', fillcolor=f'rgba({red3},{green3},{blue3},0.05)'))

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

        fig.update_layout(autosize=True, height=400, xaxis_title = "Date", yaxis_title = "CHF", title="1 Week Target Price")
        fig.update_yaxes(range=[min(data_lower.values)*0.9, max(data_upper.values)*1.1])
        st.plotly_chart(fig, use_container_width=True)

        st.write("## Machine Learning Classifiers Prediction")
        # Predict the stock price using the machine learning models and convert the labels to BUY, SELL, or HOLD
        prediction_knn = knn_model.predict([[sentiment, np.log(volume), volatility, returnt1]])[0]
        prediction_log_reg = log_reg.predict([[sentiment, np.log(volume), volatility, returnt1]])[0]
        prediction_svc = svc.predict([[sentiment, volume, volatility, returnt1]])[0]
        prediction_naive_bayes = naive_bayes.predict([[sentiment, np.log(volume), volatility, returnt1]])[0]
        prediction_dt = dt.predict([[sentiment, np.log(volume), volatility, returnt1]])[0]
        prediction_knn = "SELL" if prediction_knn == 2 else "BUY" if prediction_knn == 1 else "HOLD"
        prediction_log_reg = "SELL" if prediction_log_reg == 2 else "BUY" if prediction_log_reg == 1 else "HOLD"
        prediction_svc = "SELL" if prediction_svc == 2 else "BUY" if prediction_svc == 1 else "HOLD"
        prediction_naive_bayes = "SELL" if prediction_naive_bayes == 2 else "BUY" if prediction_naive_bayes == 1 else "HOLD"
        prediction_dt = "SELL" if prediction_dt == 2 else "BUY" if prediction_dt == 1 else "HOLD"
        
    
        models = ['KNN', 'Logistic Regression', 'SVC', 'Naive Bayes', 'Decision Tree']
        
        predictions = {
            'KNN': prediction_knn,
            'Logistic Regression': prediction_log_reg,
            'SVC': prediction_svc,
            'Naive Bayes': prediction_naive_bayes,
            'Decision Tree': prediction_dt
        }
        # accuracies of the models, see the model accuracies in the model training notebook
        accuracies = {
            'KNN': 43.4,
            'Logistic Regression': 37.5,
            'SVC': 37.2,
            'Naive Bayes': 36.2,
            'Decision Tree': 37.0
        }

        selected_model = st.selectbox('Select a model:', models) # Select the model to display the prediction
        
        # Display the prediction for the selected model
        st.markdown(f"""The {selected_model} model predicts a <span style='color: {"red" if predictions[selected_model] == "SELL" else "green" if predictions[selected_model] == "BUY" else "black"};font-weight: bold;;'>{predictions[selected_model]}</span> 
        reccomandation for the {ticker} stock, the accuracy of this classifier is {accuracies[selected_model]}%""", unsafe_allow_html=True)

        st.write("## Appendix")
        if st.checkbox("Show Linear Model Details"): # Display the model details
            st.write("## Model Summary")
            st.write(model.summary())

        
        if st.checkbox("Show News"):
            st.write("## Recent News")

            for index, row in news.iterrows(): # Display the news articles
                st.write(f"### {row['Title']}") # Display the title of the article
                st.write(f"Published on {row['Date']} by {row['Publisher']}") # Display the published date and the publisher
                st.markdown(row['Description'], unsafe_allow_html=True) # Display the description of the article (link to the article)
                st.write("---")
        
        if st.checkbox("Show Backtesting Results"): # Display the backtesting results
            st.write("## Backtesting Results with the KNN Model")
            start_date = pd.Timestamp(st.date_input("Start Date", datetime(2010, 1, 1))) # Select the start date, convert to a pandas timestamp
            end_date = pd.Timestamp(st.date_input("End Date", datetime.today())) # Select the end date, convert to a pandas timestamp
            data = compute_backtest(backtest_data, ticker, start_date, end_date) # Compute the backtest
            actual_start_date = data["Date"].iloc[0] # Get the actual start date
            actual_end_date = data["Date"].iloc[-1] # Get the actual end date
            holding_returns = data["Holding_index"].iloc[-1] # Get the holding returns
            strategy_returns = data["Strategy_index"].iloc[-1] # Get the strategy returns
            st.write(f"Holding Returns: {round(holding_returns-100,0)}%")
            st.write(f"Strategy Returns: {round(strategy_returns-100,0)}%")

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=data["Date"], y=data["Holding_index"], mode='lines', name='Holding Stock')) # Add the holding stock to the plot
            fig.add_trace(go.Scatter(x=data["Date"], y=data["Strategy_index"], mode='lines', name='Strategy')) # Add the strategy to the plot

            fig.update_layout(title=f'Returns of Strategy vs Holding Stock for {ticker}, between {actual_start_date} and {actual_end_date}',
                            xaxis_title='Date',
                            yaxis_title='Index')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__': # Run the app
    app()