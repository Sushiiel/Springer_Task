import pandas as pd
from datetime import timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model and tokenizer globally for efficiency
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
labels = ['Negative', 'Neutral', 'Positive']


def predict_sentiment(message: str) -> str:
    """
    Predict sentiment of a message using pretrained model.
    """
    inputs = tokenizer(message, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(scores).item()
    return labels[predicted_class]


def apply_sentiment_label(df: pd.DataFrame, text_col: str = 'message') -> pd.DataFrame:
    """
    Apply sentiment prediction to a DataFrame and add a new column.
    """
    df['sentiment'] = df[text_col].apply(predict_sentiment)
    return df


def preprocess_dates(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Convert date column to datetime and extract year-month.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df['month'] = df[date_col].dt.to_period('M').astype(str)
    return df


def calculate_sentiment_score(sentiment: str) -> int:
    """
    Assign score based on sentiment.
    """
    return {'Positive': 1, 'Negative': -1, 'Neutral': 0}[sentiment]


def calculate_monthly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly sentiment scores per employee.
    """
    df['score'] = df['sentiment'].apply(calculate_sentiment_score)
    monthly = df.groupby(['employee', 'month'])['score'].sum().reset_index()
    return monthly


def get_top_employees(df_scores: pd.DataFrame, top_n: int = 3, month: str = None):
    """
    Get top N positive and negative employees for a specific month.
    """
    if month:
        df_scores = df_scores[df_scores['month'] == month]

    top_pos = df_scores.sort_values(by=['month', 'score', 'employee'], ascending=[True, False, True])
    top_neg = df_scores.sort_values(by=['month', 'score', 'employee'], ascending=[True, True, True])

    top_pos_grouped = top_pos.groupby('month').head(top_n)
    top_neg_grouped = top_neg.groupby('month').head(top_n)

    return top_pos_grouped, top_neg_grouped


def detect_flight_risks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify employees who sent 4+ negative messages in any rolling 30-day window.
    """
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['sentiment'] == 'Negative'].sort_values(by=['employee', 'date'])

    flight_risks = set()

    for emp, group in df.groupby('employee'):
        dates = group['date'].tolist()
        for i in range(len(dates)):
            count = 1
            for j in range(i + 1, len(dates)):
                if (dates[j] - dates[i]).days <= 30:
                    count += 1
                    if count >= 4:
                        flight_risks.add(emp)
                        break
                else:
                    break
    return pd.DataFrame({'employee': list(flight_risks)})


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features for modeling sentiment scores.
    """
    df['message_length'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))

    df['score'] = df['sentiment'].apply(calculate_sentiment_score)
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M').astype(str)

    feature_df = df.groupby(['employee', 'month']).agg(
        message_count=('message', 'count'),
        avg_length=('message_length', 'mean'),
        avg_word_count=('word_count', 'mean'),
        sentiment_score=('score', 'sum')
    ).reset_index()

    return feature_df


def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize numeric features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)
