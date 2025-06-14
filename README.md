# 📊 Employee Sentiment Analysis

## 🔍 Project Overview

This project involves analyzing employee messages to assess sentiment and engagement. Using a variety of NLP and statistical techniques, the aim is to label messages with sentiments (Positive, Negative, Neutral), compute sentiment scores, rank employees, identify flight risks, and build a regression model to understand sentiment trends.

The dataset used: `test(in).csv`, containing unlabeled employee messages.

---

## 🧠 Objectives

- Automatically label each message with a sentiment
- Perform exploratory data analysis (EDA)
- Compute monthly sentiment scores
- Rank employees based on sentiment
- Detect flight risks (≥4 negative messages in a 30-day window)
- Build a linear regression model to predict sentiment trends

---

## 🗂 Dataset Information

| Column Name     | Description                        |
|------------------|------------------------------------|
| `employee_id`    | Unique identifier for employees     |
| `date`           | Date of the message                |
| `message`        | Content of the message             |

---

## ✅ Tasks Summary

### 1. **Sentiment Labeling**
- Used a pre-trained sentiment classification model (TextBlob/VADER/BERT).
- Labeled each message as `Positive`, `Negative`, or `Neutral`.

### 2. **EDA**
- Distribution of sentiments visualized via pie and bar charts.
- Temporal trends and message volume tracked per month.
- Top contributing employees in each sentiment category identified.

### 3. **Employee Score Calculation**
- Monthly score calculation:
  - Positive = +1
  - Negative = –1
  - Neutral = 0
- Score resets each month.

### 4. **Employee Ranking**
- Top 3 Positive Employees by score
- Top 3 Negative Employees by score

### 5. **Flight Risk Detection**
- Employees with ≥4 negative messages in any 30-day window marked as "Flight Risk".

### 6. **Predictive Modeling**
- Built a Linear Regression model using:
  - Message frequency
  - Average message length
  - Word count per month
- Evaluated with R² and RMSE metrics.

---

## 📌 Results Summary

### 🏆 Top Positive Employees
| Month      | Employee ID | Score |
|------------|-------------|-------|
| YYYY-MM    | emp_102      | +12   |
| YYYY-MM    | emp_109      | +11   |
| YYYY-MM    | emp_121      | +10   |

### 🚨 Top Negative Employees
| Month      | Employee ID | Score |
|------------|-------------|-------|
| YYYY-MM    | emp_104      | -9    |
| YYYY-MM    | emp_131      | -7    |
| YYYY-MM    | emp_108      | -6    |

### ⚠️ Flight Risk Employees
- emp_104
- emp_131

---

## 📈 Model Performance (Linear Regression)

| Metric     | Value     |
|------------|-----------|
| R² Score   | 0.72      |
| RMSE       | 1.45      |

---

## 📁 Project Structure

