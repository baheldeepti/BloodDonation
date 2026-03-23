import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import plotly.express as px
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🩸 Blood Donation DSS", layout="wide")

# ---------------- OPENAI SAFE INIT ----------------
from openai import OpenAI

client = None
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_gpt_insight(prompt: str) -> str:
    if client is None:
        return "⚠️ OpenAI key not configured"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ OpenAI error: {e}"

# ---------------- DATA ----------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/baheldeepti/BloodDonation/main/Balanced_Blood_Donation_Dataset.csv"
    return pd.read_csv(url)

# ---------------- NAV ----------------
tabs = [
    "🏠 Overview",
    "📊 Exploratory Analysis",
    "🤖 Modeling & Recommendations",
    "📈 Budget Optimization"
]

selected = st.sidebar.radio("Go to", tabs)

# ---------------- OVERVIEW ----------------
if selected == "🏠 Overview":
    st.title("🩸 Blood Donation Decision Support System")
    st.image("https://news-notes.ufhealth.org/files/2020/02/foweb_blood_donation_art_pg26.jpg", width=700)

# ---------------- EDA ----------------
elif selected == "📊 Exploratory Analysis":
    st.title("📊 Exploratory Data Analysis")
    df = load_data().copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("% Repeat Donors", f"{df['Target'].mean()*100:.2f}%")
    col3.metric("Avg Donation Volume", f"{df['Monetary'].mean():.2f} ml")

    st.plotly_chart(px.pie(df, names='CampaignResponse'))

    st.subheader("💡 AI Insights")
    st.info(get_gpt_insight("Analyze blood donation dataset trends"))

# ---------------- MODELING ----------------
elif selected == "🤖 Modeling & Recommendations":
    st.title("🤖 Modeling")

    df = load_data().copy()
    df['Monetary_per_Freq'] = df['Monetary'] / (df['Frequency'] + 1)
    df['Intensity'] = df['Frequency'] / (df['Recency'] + 1)

    features = ['Recency', 'Frequency', 'Monetary', 'Time', 'Age', 'CampaignResponse', 'Monetary_per_Freq', 'Intensity']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'SVM': SVC(probability=True),
        'Naive Bayes': GaussianNB()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        results.append({
            "Model": name,
            "ROC AUC": roc_auc_score(y_test, y_prob)
        })

    st.dataframe(pd.DataFrame(results).sort_values("ROC AUC", ascending=False))

# ---------------- OPTIMIZATION ----------------
elif selected == "📈 Budget Optimization":
    st.title("📈 Optimization")

    df = load_data().copy()

    st.write("Using Logistic Regression for simplicity")

    X = df[['Recency', 'Frequency', 'Monetary']]
    y = df['Target']

    model = LogisticRegression()
    model.fit(X, y)

    probs = model.predict_proba(X)[:,1]

    v = st.number_input("Value per donation", 150)
    c = st.number_input("Cost per contact", 1)
    B = st.number_input("Budget", 100)

    n = len(df)

    prob = LpProblem("donor", LpMaximize)
    x = [LpVariable(f"x{i}", cat=LpBinary) for i in range(n)]

    prob += lpSum([x[i]*(probs[i]*v - c) for i in range(n)])
    prob += lpSum([x[i]*c for i in range(n)]) <= B

    prob.solve()

    df["Contact"] = [int(x[i].value()) for i in range(n)]

    st.dataframe(df)
