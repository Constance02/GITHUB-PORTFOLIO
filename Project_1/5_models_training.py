import os
import pandas as pd
import kagglehub
from IPython.display import display
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import fitz

#------------------
### DATA COLLECTION
#------------------

# Excel file
df_excel = pd.read_excel("/content/preprocessed-50k.xlsx")

print(df_excel.sample(5))

# Format the dataframe
df = pd.DataFrame(df_excel)

# Excel
df.info()
df["generated"].value_counts()

# Sample the whole dataset to run faster
df_sample = df.sample(n=20000, random_state=123).reset_index(drop=True)

df_sample.info()
print(df_sample.sample(n=10))
df_sample["generated"].value_counts()

#-----------------
### MODEL PLANNING
#-----------------

# Fix the target and the features for modeling

X_text = df_sample["text"]
X_text = [str(x) if x is not None else "" for x in X_text]
X_text_series = pd.Series(X_text)
y = df_sample["generated"]

## Data (text) vectorization

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words='english',
    lowercase=True,
    ngram_range=(1, 3),
    min_df=5,
    token_pattern=r"(?u)\b\w+\b|[^\w\s]"  # keeps words + ponctuation
)

X = vectorizer.fit_transform(X_text)

# Split df into two sets (training & testing)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

#----------------------------------
### MODEL BUILDING (Classification)
#----------------------------------

## Logistic Regression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

## Random Forest

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print("=== Random Forest ===")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

## XGBoost

xgb_ = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_.fit(X_train, y_train)
y_pred = xgb_.predict(X_test)

print("=== XGBoost ===")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

## LightGBM

lgb_ = lgb.LGBMClassifier()
lgb_.fit(X_train, y_train)
y_pred = lgb_.predict(X_test)

print("=== LightGBM ===")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
plt.title("Confusion Matrix - LightGBM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

## Gradient Boosting

gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)

print("=== Gradient Boosting ===")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
plt.title("Confusion Matrix - Gradient Boosting")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#---------------------
### MODELS' EVALUATION
#---------------------

# Dictionnary to store the performances
performance_summary = {}

# Function to extract the metrics
def evaluate_model(name, y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    performance_summary[name] = {
        "Accuracy": report["accuracy"],
        "Precision (AI)": report["1"]["precision"],
        "Recall (AI)": report["1"]["recall"],
        "F1-score (AI)": report["1"]["f1-score"],
        "Precision (Human)": report["0"]["precision"],
        "Recall (Human)": report["0"]["recall"],
        "F1-score (Human)": report["0"]["f1-score"]
    }

# Evaluate the models

# Logistic Regression
evaluate_model("Logistic Regression", y_test, LogisticRegression().fit(X_train, y_train).predict(X_test))

# Random Forest
evaluate_model("Random Forest", y_test, RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train).predict(X_test))

# XGBoost
evaluate_model("XGBoost", y_test, xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train).predict(X_test))

# LightGBM
evaluate_model("LightGBM", y_test, lgb.LGBMClassifier().fit(X_train, y_train).predict(X_test))

# Gradient Boosting
evaluate_model("Gradient Boosting", y_test, GradientBoostingClassifier().fit(X_train, y_train).predict(X_test))

# Print the performances
summary_df = pd.DataFrame(performance_summary).T.round(3)
print("=== Comparative summary of performances ===")
display(summary_df.sort_values(by="Precision (AI)", ascending=False))