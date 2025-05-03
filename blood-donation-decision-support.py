# blood_donation_dss_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import plotly.express as px
import openai
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary


# Streamlit page configuration
st.set_page_config(page_title="🩸 Blood Donation DSS", layout="wide")

# Navigation menu using Streamlit's native radio buttons
st.sidebar.title("Navigation")
selected = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "📊 Exploratory Analysis", "🤖 Modeling & Recommendations", "📈 Budget Optimization"]
)


# OpenAI insight generator
@st.cache_data
def get_gpt_insight(prompt: str) -> str:
    openai.api_key = st.secrets.get("OPENAI_API_KEY", "")
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

# --- PAGE 1: INTRODUCTION ---
if selected == "🏠 Overview":
    st.title("🩸 Blood Donation Decision Support System")
    st.image(
        "https://news-notes.ufhealth.org/files/2020/02/foweb_blood_donation_art_pg26.jpg",
        caption="Donate Blood, Save Lives", width = 700
    )
    st.markdown(
        """
        Welcome to the Blood Donation DSS—an AI-powered application to:
        - Predict repeat donors
        - Forecast donation volume
        - Recommend personalized outreach
        - Maximize donation value
        """
    )
    st.subheader("📖 Data Dictionary")
    st.markdown(
        """
        | Column            | Description                                |
        |-------------------|--------------------------------------------|
        | Recency           | Months since last donation                 |
        | Frequency         | Total number of donations                  |
        | Monetary          | Total blood volume donated (in mL)         |
        | Time              | Months since first donation                |
        | Age               | Donor age (years)                          |
        | CampaignResponse  | 1 if donor responded to previous campaign  |
        | Target            | 1 if donor donated again, else 0           |
        """
    )

# --- PAGE 2: EXPLORATORY ANALYSIS ---
elif selected == "📊 Exploratory Analysis":
    st.title("📊 Exploratory Data Analysis")
    url = ("https://raw.githubusercontent.com/baheldeepti/BloodDonation/main/Balanced_Blood_Donation_Dataset.csv")
    df = pd.read_csv(url)



    # KPI summary
    st.subheader("📌 KPI Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("% Repeat Donors", f"{df['Target'].mean()*100:.2f}%")
    col3.metric("Avg Donation Volume", f"{df['Monetary'].mean():.2f} ml")

    # Categorical distributions
    st.subheader("📊 Categorical Distributions")
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.pie(df, names='CampaignResponse', title='Campaign Response Rate')
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        target_counts = df['Target'].value_counts().rename_axis('Target').reset_index(name='Count')
        fig2 = px.bar(target_counts, x='Target', y='Count', title='Repeat Donation Counts')
        st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap
    st.subheader("🔍 Correlation Heatmap")
    corr_mat = df.corr()
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    # Boxplots by target
    st.subheader("📊 Boxplots by Donation Outcome")
    b1, b2 = st.columns(2)
    with b1:
        fig4 = px.box(df, x='Target', y='Monetary', title='Monetary by Outcome')
        st.plotly_chart(fig4, use_container_width=True)
    with b2:
        fig5 = px.box(df, x='Target', y='Recency', title='Recency by Outcome')
        st.plotly_chart(fig5, use_container_width=True)

    # Scatter plot
    st.subheader("🔗 Frequency vs Monetary")
    fig6 = px.scatter(df, x='Frequency', y='Monetary', color=df['Target'].map({0:'No',1:'Yes'}),
                      title='Frequency vs Monetary by Repeat Donation')
    st.plotly_chart(fig6, use_container_width=True)



    # AI insights
    st.subheader("💡 AI-Generated Insights")
    prompt = (
        "Act as a Data Analyst and Generate key insights from a blood donation dataset with columns: Recency, Frequency, Monetary, "
        "Time, Age, CampaignResponse, Target, based on observed distributions and correlations."
    )
    ins = get_gpt_insight(prompt)
    st.info(ins)

# --- PAGE 3: MODELING & RECOMMENDATIONS ---
elif selected == "🤖 Modeling & Recommendations":
    st.title("🔍 Modeling & Recommendations")
    st.markdown("""
    1. Feature engineering & select top-7  
    2. Multicollinearity check  
    3. Train models & ensembles  
    4. ROC & confusion matrix  
    5. Upload CSV or add manual entries  
    6. Personalized outreach recommendations  
    7. AI insights & next steps  
    """)

    # 1) Load & engineer features
    url = "https://raw.githubusercontent.com/baheldeepti/BloodDonation/main/Balanced_Blood_Donation_Dataset.csv"
    df = pd.read_csv(url)
    df['Monetary_per_Freq'] = df['Monetary'] / (df['Frequency'] + 1)
    df['Intensity']         = df['Frequency'] / (df['Recency'] + 1)

    # 2) Select top-7 features by correlation
    corr_t = df.corr()['Target'].abs().sort_values(ascending=False)
    feats  = corr_t.index[1:8].tolist()
    st.subheader("🔑 Top 7 Features")
    st.table(pd.DataFrame({
        'Feature': feats,
        '|Corr| w/ Target': corr_t[feats].round(3).values
    }))

    # 3) Multicollinearity check
    fc = df[feats].corr().abs()
    high = [(fc.index[i], fc.columns[j], fc.iloc[i,j])
            for i in range(len(fc)) for j in range(i) if fc.iloc[i,j]>0.9]
    if high:
        st.warning("🚨 High collinearity detected:")
        for f1,f2,v in high:
            st.write(f"• {f1} ↔ {f2}: {v:.2f}")
    else:
        st.success("✅ No high collinearity.")

    # 4) Split & scale
    X, y = df[feats], df['Target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    X_trs, X_tes = scaler.transform(X_tr), scaler.transform(X_te)

    # 5) Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'CatBoost': CatBoostClassifier(verbose=0),
        'Support Vector Machine': SVC(probability=True),
        'Naive Bayes': GaussianNB()
    }

    # 6) Train & evaluate
    results = []
    trained_models = {}
    for name, mdl in models.items():
        mdl.fit(X_trs, y_tr)
        trained_models[name] = mdl

        y_p  = mdl.predict(X_tes)
        y_pr = mdl.predict_proba(X_tes)[:,1]

        results.append({
            'Model':     name,
            'Accuracy':  accuracy_score( y_te, y_p),
            'Precision': precision_score(y_te, y_p),
            'Recall':    recall_score(   y_te, y_p),
            'F1 Score':  f1_score(      y_te, y_p),
            'ROC AUC':   roc_auc_score( y_te, y_pr)
        })

    # 7) Plot ROC curves
    st.subheader("📉 ROC Curves")
    plt.figure(figsize=(8,6))
    for name, mdl in trained_models.items():
        y_pr = mdl.predict_proba(X_tes)[:,1]
        fpr, tpr, _ = roc_curve(y_te, y_pr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y_te,y_pr):.2f})")

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    st.pyplot(plt)

    # 8) Show comparison table
    st.subheader("📋 Model Comparison")
    
    df_res = (
        pd.DataFrame(results)
          .sort_values('ROC AUC', ascending=False)
          .reset_index(drop=True)
    )
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    # Color‐code only the metric columns:
    styled = (
        df_res.style
              .background_gradient(subset=metrics, cmap='Greens')
              .highlight_max(subset=metrics, color='lightgreen')
              .highlight_min(subset=metrics, color='salmon')
    )
    
    st.dataframe(styled)
    st.subheader("📋 Model Comparison")
    df_res = pd.DataFrame(results).sort_values('ROC AUC', ascending=False).reset_index(drop=True)
    st.dataframe(df_res)

    # 9) Persist for optimization page
    st.session_state['trained_models'] = trained_models
    st.session_state['scaler']         = scaler
    st.session_state['features']       = feats
    # AI insights & next steps
    st.subheader("🤖 AI Insights & Next Steps")
    prompt = (
        "Senior data scientist with 20 yrs exp. Given:\n\n"
        f"{df_res.to_csv(index=False)}\n\n"
        f"Confusion Matrix ({best_model}): {cm.tolist()}\n\n"
        "Recommend best model and business cases where which model would fit best and recommendations"
    )
    st.info(get_gpt_insight(prompt))

# --- PAGE 5: CAMPAIGN BUDGET OPTIMIZATION ---
elif selected == "📈 Budget Optimization":
    st.title("📈 Campaign Budget Optimization")
    st.markdown(
        """
        **What is this?**  
        This tool helps you decide *which* donors to contact given a fixed outreach budget, 
        so you maximize your *expected* donation return.

        **Where does donor data come from?**  
        You can either upload a CSV file here or manually enter one donor at a time below.
        """
    )

    # ➤ Data Input
    st.subheader("1. Donation Data Input")
    uploaded = st.file_uploader("Upload donor data CSV", type=["csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded)
        st.success("✅ CSV loaded!")
    else:
        # manual entry form
        if "manual_opt" not in st.session_state:
            st.session_state.manual_opt = []
        with st.form("opt_entry_form", clear_on_submit=True):
            r = st.number_input("Recency (months since last donation)", 0, 100, 10)
            f = st.number_input("Frequency (total donations)", 1, 50, 2)
            m = st.number_input("Monetary (total mL donated)", 0, 5000, 500)
            t = st.number_input("Time (months since first donation)", 1, 200, 12)
            a = st.number_input("Age (years)", 18, 100, 35)
            c = st.selectbox("Responded to last campaign?", ["Yes", "No"])
            if st.form_submit_button("Add donor"):
                st.session_state.manual_opt.append({
                    "Recency": r,
                    "Frequency": f,
                    "Monetary": m,
                    "Time": t,
                    "Age": a,
                    "CampaignResponse": 1 if c == "Yes" else 0
                })
                st.success("Donor added!")
        df_in = pd.DataFrame(st.session_state.manual_opt)

    if df_in.empty:
        st.info("Please upload a CSV or add at least one donor above.")
    else:
        st.subheader("2. Preview & Explain Data")
        st.dataframe(df_in)
        st.markdown(
            """
            **Glossary of terms:**  
            - **Recency:** Months since last donation.  
            - **Frequency:** How many times they’ve donated.  
            - **Monetary:** Total volume donated (in mL).  
            - **Time:** Months since first donation.  
            - **CampaignResponse:** Whether they engaged with the last outreach (1=Yes, 0=No).
            """
        )

        # ➤ Feature engineering (same as modeling page)
        df_in["Monetary_per_Freq"] = df_in["Monetary"] / (df_in["Frequency"] + 1)
        df_in["Intensity"] = df_in["Frequency"] / (df_in["Recency"] + 1)

        # ➤ Pick trained model
        models = st.session_state["trained_models"]
        scaler = st.session_state["scaler"]
        feats = st.session_state["features"]
        model_name = st.selectbox("Choose predictive model", list(models.keys()))
        model = models[model_name]

        # ➤ Predict probabilities
        X_opt = scaler.transform(df_in[feats])
        p_i = model.predict_proba(X_opt)[:, 1]
        df_opt = df_in.copy()
        df_opt["Predicted Probability"] = np.round(p_i, 3)
        st.subheader("3. Predicted Donation Probability")
        st.markdown(
            "This is the model’s best guess, on a scale from 0 to 1, of how likely each donor is to give again."
        )
        st.dataframe(df_opt)

        # ➤ Budget parameters
        st.subheader("4. Set Your Budget & Values")
        v = st.number_input("Value per successful donation (USD)", value=150)
        c = st.number_input("Cost per contact (USD)", value=1)
        B = st.number_input("Total budget (USD)", value=100)
        

        # ➤ AI-Generated Strategy Insights
        st.subheader("6. AI-Generated Strategy Insights")
        ai_prompt = (
            "You are a nonprofit fundraising strategist reviewing the donor optimization results below:\n\n"
            f"{df_opt.to_csv(index=False)}\n\n"
            "Please present your analysis in three clearly labeled sections:\n"
            "1. **Key Findings:** 2–3 bullet points summarizing the most important insights.\n"
            "2. **What This Means:** A brief, non-technical explanation of how these results impact our outreach strategy.\n"
            "3. **Recommendations:** 2–3 actionable, easy-to-understand steps we should take next.\n"
            "Write in a conversational style that anyone on the team can follow."
        )

        insight = get_gpt_insight(ai_prompt)
        st.info(insight)
