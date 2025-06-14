# Employee Sentiment Analysis Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from textblob import TextBlob
import os

# ------------------------ Task 1: Load & Sentiment Labeling ------------------------
df = pd.read_csv('test.csv')  # assume columns: ['EmployeeID', 'Date', 'Message']
df['Date'] = pd.to_datetime(df['Date'])

def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Message'].apply(get_sentiment)

# ------------------------ Task 2: Exploratory Data Analysis ------------------------
sns.countplot(x='Sentiment', data=df)
plt.title("Sentiment Distribution")
plt.savefig("visualization/sentiment_distribution.png")
plt.close()

# Sentiment over time
df['Month'] = df['Date'].dt.to_period('M')
monthly_sentiment = df.groupby(['Month', 'Sentiment']).size().unstack().fillna(0)
monthly_sentiment.plot(kind='bar', stacked=True)
plt.title("Monthly Sentiment Trend")
plt.savefig("visualization/monthly_sentiment_trend.png")
plt.close()

# ------------------------ Task 3: Employee Score Calculation ------------------------
sentiment_score = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['Score'] = df['Sentiment'].map(sentiment_score)
df['Month'] = df['Date'].dt.to_period('M')
monthly_score = df.groupby(['EmployeeID', 'Month'])['Score'].sum().reset_index()

# ------------------------ Task 4: Employee Ranking ------------------------
ranked = monthly_score.copy()
ranked['Month'] = ranked['Month'].astype(str)
ranked_top = ranked.groupby('Month').apply(
    lambda x: x.sort_values(by=['Score', 'EmployeeID'], ascending=[False, True]).head(3)
).reset_index(drop=True)
ranked_bottom = ranked.groupby('Month').apply(
    lambda x: x.sort_values(by=['Score', 'EmployeeID']).head(3)
).reset_index(drop=True)

# ------------------------ Task 5: Flight Risk Identification ------------------------
df['NegativeFlag'] = df['Sentiment'] == 'Negative'
rolling_neg = df[df['NegativeFlag']].groupby('EmployeeID').rolling('30D', on='Date').count()['NegativeFlag']
flight_risks = rolling_neg[rolling_neg >= 4].reset_index()['EmployeeID'].unique().tolist()

# ------------------------ Task 6: Predictive Modeling ------------------------
df['MessageLength'] = df['Message'].apply(lambda x: len(str(x)))
df['WordCount'] = df['Message'].apply(lambda x: len(str(x).split()))
features_df = df.groupby(['EmployeeID', 'Month']).agg({
    'Score': 'sum',
    'Message': 'count',
    'MessageLength': 'mean',
    'WordCount': 'mean'
}).reset_index()

features_df.rename(columns={
    'Message': 'MessageFrequency',
    'MessageLength': 'AvgMsgLength',
    'WordCount': 'AvgWordCount'
}, inplace=True)

X = features_df[['MessageFrequency', 'AvgMsgLength', 'AvgWordCount']]
y = features_df['Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Save performance plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sentiment Score")
plt.ylabel("Predicted Score")
plt.title(f"Linear Regression\nRMSE: {rmse:.2f}, R2: {r2:.2f}")
plt.savefig("visualization/regression_performance.png")
plt.close()

# ------------------------ Save Final Visualization for Monthly Score ------------------------
pivot = monthly_score.pivot(index='Month', columns='EmployeeID', values='Score').fillna(0)
pivot.plot(figsize=(12, 6))
plt.title("Monthly Sentiment Score per Employee")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualization/monthly_sentiment_score_visualization_fixed.png")
plt.close()

# ------------------------ Print Final Summary ------------------------
print("Top 3 Positive Employees per Month:\n", ranked_top)
print("Top 3 Negative Employees per Month:\n", ranked_bottom)
print("Flight Risk Employees:\n", flight_risks)
print(f"Model RMSE: {rmse:.2f}, R^2 Score: {r2:.2f}")
