import re
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import snscrape.modules.twitter as sntwitter
from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="https://www.freepnglogos.com/uploads/twitter-logo-png/twitter-logo-vector-png-clipart-1.png")
st.title("Welcome to Twitter Sentiment Analysis! 👋")
st.caption("Twitter Sentiment Analysis is a web-based application that can analyze the sentiment of tweets based on the VADER (Valence Aware Dictionary and Sentiment Reasoner) algorithm. This app allows users to enter a query (keyword, hashtag, etc.) to search for tweets and generates a sentiment score. This app was created by a [student](https://github.com/agungmahadana/) using Python and Streamlit.")

def get_data(amount, query):
    # Declare variables
    tweets = []
    url = []
    # Loop through tweets
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        # To prevent unecessary looping
        if i > amount:
            break
        tweets.append(tweet.rawContent)
        url.append(tweet.url)
    return pd.DataFrame(list(zip(tweets, url)), columns=['TWEETS', 'URL'])

def get_sentiment(data):
    # Declare variables
    sentiment = []
    polarity = []
    subjectivity = []
    pos = []
    neg = []
    neu = []
    compound = []
    # Loop through tweets
    for i in range(len(data)):
        # Translate tweet to english then put it in a list
        translated = GoogleTranslator(source='auto', target='en').translate(data['TWEETS'][i])
        score = SentimentIntensityAnalyzer().polarity_scores(translated)
        polarity.append(TextBlob(translated).sentiment.polarity)
        subjectivity.append(TextBlob(translated).sentiment.subjectivity)
        pos.append(score['pos'])
        neg.append(score['neg'])
        neu.append(score['neu'])
        compound.append(score['compound'])
        # Get sentiment
        if score['compound'] >= 0.05:
            sentiment.append('positive')
        elif score['compound'] <= - 0.05:
            sentiment.append('negative')
        else:
            sentiment.append('neutral')
    data['SENTIMENT'] = sentiment
    data['POLARITY'] = polarity
    data['SUBJECTIVITY'] = subjectivity
    data['POS'] = pos
    data['NEG'] = neg
    data['NEU'] = neu
    data['COMPOUND'] = compound
    return data

def percentage(part, whole):
    return round(100 * float(part) / float(whole), 1)

def get_percentage(data):
    # Calculating the number of positive, negative and neutral tweets
    no_of_tweets = len(data)
    no_of_pos = len(data.loc[data['SENTIMENT'] == 'positive'])
    no_of_neg = len(data.loc[data['SENTIMENT'] == 'negative'])
    no_of_neu = len(data.loc[data['SENTIMENT'] == 'neutral'])
    # Calculating the percentage of positive, negative and neutral tweets
    pos_percentage = percentage(no_of_pos, no_of_tweets)
    neg_percentage = percentage(no_of_neg, no_of_tweets)
    neu_percentage = percentage(no_of_neu, no_of_tweets)
    return pos_percentage, neg_percentage, neu_percentage, no_of_pos, no_of_neg, no_of_neu

def case_folding(data):
    corpus = []
    for i in range(len(data)):
        data[i] = re.sub(r'http\S+', '', data[i])
        data[i] = re.sub(r'#\w+', '', data[i])
        data[i] = re.sub(r'@\w+', '', data[i])
        data[i] = re.sub(r'[^\w\s]', ' ', data[i])
        data[i] = re.sub(r'\s+', ' ', data[i])
        data[i] = re.sub(r'[0-9]', ' ', data[i])
        data[i] = data[i].lower()
        data[i] = data[i].split()
        print(data[i])
        corpus.append(" ".join(data[i]))
    return corpus

def word_cloud(data):
    # Generate wordcloud
    word_data = case_folding(data['TWEETS'])
    text = " ".join(review for review in word_data)
    mask = np.array(Image.open("cloud.png"))
    wordcloud = WordCloud(background_color="white", mask=mask).generate(text)
    # Display the generated image:
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

def chart(data_percentage):
    # Define data and labels
    sizes = [data_percentage[0], data_percentage[1], data_percentage[2]]
    labels = [f"Positive ({data_percentage[0]}%)", f"Negative ({data_percentage[1]}%)", f"Neutral ({data_percentage[2]}%)"]
    counts = [data_percentage[3], data_percentage[4], data_percentage[5]]
    # Generate pie chart
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(sizes, labels=labels, autopct=lambda pct: counts.pop(0), startangle=90, colors=['#14B95F', '#F9596E', '#FFA803'], pctdistance=0.5)
    ax.axis('equal')
    # Draw circle to make donut chart
    circle = plt.Circle((0,0), 0.6, color='white')
    fig = plt.gcf()
    fig.gca().add_artist(circle)
    return fig

def download(final):
    result_str = ''
    # Loop through columns labls
    for i in range(len(final.axes[1])):
        if i != len(final.axes[1])-1:
            result_str += str(final.axes[1][i]) + ';'
        else:
            result_str += str(final.axes[1][i]) + '\n'
    # Loop through values
    for row in final.itertuples(index=False):
        for val in row:
            result_str += str(val) + ';'
        result_str += '\n'
    return result_str

# Frontend
query = st.text_input("Enter Queries", placeholder="Type here...")
amount = st.number_input("Enter the Number of Tweets", min_value=1) - 1
if st.button("Analyze") and query != '':
    data_pure = get_data(amount, query)
    data_sentiment = get_sentiment(data_pure)
    download = download(data_sentiment)
    st.write(data_sentiment)
    st.pyplot(word_cloud(data_pure))
    st.pyplot(chart(get_percentage(data_sentiment)))
    st.download_button(label='Download CSV', data=download, file_name='Twitter Sentiment Analysis.csv', mime='text/csv')
else :
    st.caption("Contoh Query: :a - iphone:a - @jokowi:a - #covid:a - from:911:a - pemilu since:2023-01-01 until:2023-01-02")