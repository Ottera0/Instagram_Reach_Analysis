import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sb 
import plotly.express as px

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

data = pd.read_csv("data/archive/Instagram_data.csv", encoding='latin1')
data = data.dropna()
print(data.head())

#analyzing reach from home

# plt.figure(figsize=(10,10))
# plt.style.use('fivethirtyeight')
# plt.title("Distribution of Impressions From Home")
# sb.distplot(data['From Home'])
# #plt.show()


#Analyzing reach from hashtags

# plt.figure(figsize=(10,10))
# plt.title("Distribution of Impressions From Hashtags")
# sb.distplot(data['From Hashtags'])
# #plt.show()

#Analyzing reach from Explore

# plt.figure(figsize=(10, 10))
# plt.title("Distribution of Impressions From Explore")
# sb.distplot(data['From Explore'])
# #plt.show()

#Percentage of impressions from various sources

# home = data["From Home"].sum()
# hashtags = data["From Hashtags"].sum()
# explore = data["From Explore"].sum()
# other = data["From Other"].sum()

# labels = ['From Home','From Hashtags','From Explore','Other']
# values = [home, hashtags, explore, other]

# fig = px.pie(data, values=values, names=labels, 
#              title='Impressions on Instagram Posts From Various Sources', hole=0.5)
# fig.show()

#caption wordcloud

# text = " ".join(i for i in data.Caption)
# stopwords = set(STOPWORDS)
# wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
# plt.style.use('classic')
# plt.figure( figsize=(12,10))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()

#hashtags wordcloud

# text = " ".join(i for i in data.Hashtags)
# stopwords = set(STOPWORDS)
# wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
# plt.figure( figsize=(12,10))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()

#analyzing relationship likes and impressions
# figure = px.scatter(data_frame = data, x="Impressions",
#                     y="Likes", size="Likes", trendline="ols", 
#                     title = "Relationship Between Likes and Impressions")
# figure.show()
#clear linear relationship

#analyzing relationship  number of comments and impressions
# figure = px.scatter(data_frame = data, x="Impressions",
#                     y="Comments", size="Comments", trendline="ols", 
#                     title = "Relationship Between Comments and Total Impressions")
# figure.show()
# #no relationship between the two

#