# blood_donation_dss_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import whisper
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import av
import tempfile
from prophet import Prophet
from st_aggrid import AgGrid
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

import soundfile as sf

# Streamlit page configuration
st.set_page_config(page_title="🩸 Blood Donation DSS", layout="wide")

# Navigation menu using Streamlit's native radio buttons
# Add tab to Streamlit sidebar navigation


# Define tab options once
tabs = [
    "🏠 Overview", "📊 Exploratory Analysis","🤖 Modeling & Recommendations", "📈 Budget Optimization", "🔁 What-If Scenario", 
     "💬 Conversational Chatbot",
    "🎙️ Voice Assistant"
]

# Safely get the index for the default tab
default_index = tabs.index(st.session_state.get("selected_tab", "🏠 Overview"))

# Render sidebar radio button
selected = st.sidebar.radio("Go to", tabs, index=default_index)

# Persist selected tab in session state
st.session_state.selected_tab = selected


# OpenAI insight generator

def get_gpt_insight(prompt: str) -> str:
    openai.api_key = st.secrets.get("OPENAI_API_KEY", "")
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()




# --- Added load_data function ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/baheldeepti/BloodDonation/main/Balanced_Blood_Donation_Dataset.csv"
    return pd.read_csv(url)

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
        
    #Donation volumne by campaign response
    st.subheader("📊 Donation Volume by Campaign Response")
    fig3 = px.box(df, x="CampaignResponse", y="Monetary", color="Target", title="Donation Volume")
    st.plotly_chart(fig3, use_container_width=True)

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
    5. AI insights & next steps  
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


    # 9) Persist for optimization page
    st.session_state["basic_models"]      = trained_models.copy()
    st.session_state["basic_features"]    = feats.copy()
    st.session_state['scaler']         = scaler


    # --- Advanced Hyperparameter Tuning & Ensembling ---

    # 4. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )
    
    # 5. Define models & hyperparameter grids
    model_params = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000),
            {'classifier__C': [0.1, 1, 10]}
        ),
        'Random Forest': (
            RandomForestClassifier(),
            {'classifier__n_estimators': [100], 'classifier__max_depth': [5, 15]}
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(),
            {'classifier__n_estimators': [100], 'classifier__learning_rate': [0.05, 0.1]}
        ),
        'AdaBoost': (
            AdaBoostClassifier(),
            {'classifier__n_estimators': [100, 200]}
        ),
        'XGBoost': (
            XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            {'classifier__n_estimators': [100], 'classifier__max_depth': [3, 5]}
        ),
        'SVM': (
            SVC(probability=True),
            {'classifier__C': [0.1, 1], 'classifier__kernel': ['rbf']}
        ),
        'Naive Bayes': (
            GaussianNB(),
            {}  # no hyperparameters
        )
    }
    
    results = []
    trained_models = {}
    
    # 6. Train, tune, CV-evaluate, and test-evaluate each model
    for name, (estimator, params) in model_params.items():
        # Build pipeline with scaler + estimator
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', estimator)
        ])
        
        if params:
            grid = GridSearchCV(
                pipe, params, cv=5, scoring='roc_auc', n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            cv_auc     = grid.best_score_
        else:
            pipe.fit(X_train, y_train)
            best_model = pipe
            cv_auc     = cross_val_score(
                pipe, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1
            ).mean()
        
        trained_models[name] = best_model
        
        # Test set evaluation
        y_pred = best_model.predict(X_test)
        if hasattr(best_model.named_steps['classifier'], "predict_proba"):
            y_proba = best_model.predict_proba(X_test)[:, 1]
        else:
            # fallback if no predict_proba
            y_proba = best_model.decision_function(X_test)
        
        results.append({
            'Model':          name,
            'CV ROC AUC':     round(cv_auc, 4),
            'Test Accuracy':  round(accuracy_score(y_test, y_pred), 4),
            'Test Precision': round(precision_score(y_test, y_pred), 4),
            'Test Recall':    round(recall_score(y_test, y_pred), 4),
            'Test F1 Score':  round(f1_score(y_test, y_pred), 4),
            'Test ROC AUC':   round(roc_auc_score(y_test, y_proba), 4)
        })
    
    # 7. Build a Voting Ensemble of top 3 by CV ROC AUC
    top3 = sorted(results, key=lambda x: x['CV ROC AUC'], reverse=True)[:3]
    ensemble = VotingClassifier(
        estimators=[(r['Model'], trained_models[r['Model']]) for r in top3],
        voting='soft'
    )
    ensemble_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', ensemble)
    ])
    ensemble_pipe.fit(X_train, y_train)
    
    # Ensemble test evaluation
    y_pred_e   = ensemble_pipe.predict(X_test)
    y_proba_e  = ensemble_pipe.predict_proba(X_test)[:, 1]
    ensemble_cv_auc = cross_val_score(
        ensemble_pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc', n_jobs=-1
    ).mean()
    
    results.append({
        'Model':          'Voting Ensemble',
        'CV ROC AUC':     round(ensemble_cv_auc,    4),
        'Test Accuracy':  round(accuracy_score(y_test, y_pred_e), 4),
        'Test Precision': round(precision_score(y_test, y_pred_e), 4),
        'Test Recall':    round(recall_score(y_test, y_pred_e),    4),
        'Test F1 Score':  round(f1_score(y_test, y_pred_e),        4),
        'Test ROC AUC':   round(roc_auc_score(y_test, y_proba_e),  4)
    })
    
    # 8. Display final results
    df_results = (
        pd.DataFrame(results)
          .sort_values(by='CV ROC AUC', ascending=False)
          .reset_index(drop=True)
    )
    
    st.subheader("📊 Advanced Tuning & Ensembling Results")
    df_results = (
    pd.DataFrame(results)
      .sort_values(by='CV ROC AUC', ascending=False)
      .reset_index(drop=True)
)

    
    # Define which columns to color‐code
    metrics = [
        'CV ROC AUC',
        'Test Accuracy',
        'Test Precision',
        'Test Recall',
        'Test F1 Score',
        'Test ROC AUC'
    ]
    
    # Apply a green gradient to the metric columns,
    # highlight the best in light green and worst in salmon
    styled = (
        df_results.style
                  .background_gradient(subset=metrics, cmap='Greens')
                  .highlight_max(subset=metrics, color='lightgreen')
                  .highlight_min(subset=metrics, color='salmon')
    )
    
    st.dataframe(styled)
    # 9. Plot ROC Curves for All Models
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, model in trained_models.items():
        if hasattr(model.named_steps['classifier'], "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")
    
    # Ensemble curve
    fpr_e, tpr_e, _ = roc_curve(y_test, y_proba_e)
    ax.plot(fpr_e, tpr_e, '--', lw=2, label=f"Voting Ensemble (AUC = {auc(fpr_e, tpr_e):.2f})")
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for All Models')
    ax.legend(loc='lower right')
    st.pyplot(fig)
    #save models
    st.session_state["advanced_models"]   = trained_models.copy()
    st.session_state["advanced_features"] = feats.copy()

    
    # --- AI-Generated Model Comparison & Recommendations ---
    st.subheader("🤖 AI Model Comparison & Recommendations")
    ai_prompt = (
    "You are a Senior Data Scientist with 20 years of experience. "
    "Below are two CSV tables: the original basic-model results (`df_res`) and the advanced "
    "hyperparameter tuning & ensemble results (`df_results`).\n\n"
    f"Original Models:\n{df_res.to_csv(index=False)}\n\n"
    f"Tuned & Ensemble Models:\n{df_results.to_csv(index=False)}\n\n"
    "1) **Combined Performance & Use Case Table**: Create a single markdown table with these columns:\n"
    "   - `Model`\n"
    "   - `Best Use Case` (which donor-outreach scenario it's ideal for)\n"
    "   - `Improvement?` (Yes/No, indicating if tuning improved over original)\n"
    "   - `Δ ROC AUC` (the change in ROC AUC, e.g. +0.03)\n\n"
    "Include the “Voting Ensemble” as its own row in this table.\n\n"
    "2) **Recommendations**: Below the table, list 2–3 bullet points. For each:\n"
    "   - **Recommendation:** The action to take.\n"
    "   - **Why:** The rationale.\n"
    "   - **Impact:** The expected benefit for our donor outreach strategy.\n\n"
    "Format everything in clear markdown."
)

    st.info(get_gpt_insight(ai_prompt))
    st.session_state["scaler"]         = scaler

# --- PAGE 4: BUDGET OPTIMIZATION ---
elif selected == "📈 Budget Optimization":
    st.title("📈 Campaign Budget Optimization")
    st.markdown("""
        **What is this?**  
        Decide which donors to contact under a fixed outreach budget for maximum expected return.

        **Where does donor data come from?**  
        Upload a CSV or enter donors manually below.
    """)

    # 1) Data input
    uploaded = st.file_uploader("Upload donor data CSV", type=["csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded)
        st.success("✅ CSV loaded!")
    else:
        if "manual_opt" not in st.session_state:
            st.session_state.manual_opt = []
        with st.form("opt_form", clear_on_submit=True):
            r = st.number_input("Recency (months since last donation)", 0, 100, 10)
            f = st.number_input("Frequency (total donations)", 1, 50, 2)
            m = st.number_input("Monetary (mL donated)", 0, 5000, 500)
            t = st.number_input("Time (months since first donation)", 1, 200, 12)
            a = st.number_input("Age (years)", 18, 100, 35)
            c = st.selectbox("Responded to last campaign?", ["Yes","No"])
            if st.form_submit_button("Add donor"):
                st.session_state.manual_opt.append({
                    "Recency": r,
                    "Frequency": f,
                    "Monetary": m,
                    "Time": t,
                    "Age": a,
                    "CampaignResponse": 1 if c=="Yes" else 0
                })
                st.success("Donor added!")
        df_in = pd.DataFrame(st.session_state.manual_opt)

    if df_in.empty:
        st.info("Please upload a CSV or add at least one donor.")
        st.stop()

    # 2) Preview & glossary
    st.subheader("Donation Data")
    st.dataframe(df_in)
    st.markdown("**Glossary:** Recency, Frequency, Monetary, Time, Age, CampaignResponse")

    # 3) Recompute only the two features models expect
    df_in["Monetary_per_Freq"] = df_in["Monetary"] / (df_in["Frequency"] + 1)
    df_in["Monetary_per_Donation"] = df_in["Monetary"] / (df_in["Frequency"] + 1)
    df_in["Intensity"]         = df_in["Frequency"] / (df_in["Recency"] + 1)
    df_in["Donation_Intensity"]         = df_in["Frequency"] / (df_in["Recency"] + 1)


    # 4) Model-set selection
    st.subheader("Choose Predictive Model Set")
    choice = st.radio("Which model family?", ["Original Models", "Advanced Models"])
    if choice=="Original Models":
        models = st.session_state["basic_models"]
        feats  = st.session_state["basic_features"]
    else:
        models = st.session_state["advanced_models"]
        feats  = st.session_state["advanced_features"]

    # 5) Validate features
    missing = [col for col in feats if col not in df_in.columns]
    if missing:
        st.error(f"Missing features: {missing}. Please include them and retry.")
        st.stop()

    # 6) Scale & predict
    scaler     = st.session_state["scaler"]
    model_name = st.selectbox("Pick a model", list(models.keys()))
    model      = models[model_name]
    X_opt      = scaler.transform(df_in[feats])
    p_i        = model.predict_proba(X_opt)[:,1]

    df_opt = df_in.copy()
    df_opt["Predicted Probability"] = np.round(p_i,3)
    st.subheader("Predicted Donation Probability")
    st.dataframe(df_opt)

    # 7) Budget inputs
    st.subheader("Budget & Values")
    v = st.number_input("Value per donation (USD)", 150)
    c = st.number_input("Cost per contact (USD)", 1)
    B = st.number_input("Total budget (USD)", 100)

    # 8) Optimization
    n       = len(df_opt)
    prob_lp = LpProblem("donor_alloc", LpMaximize)
    x_vars  = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]
    prob_lp += lpSum([x_vars[i]*(p_i[i]*v - c) for i in range(n)])
    prob_lp += lpSum([x_vars[i]*c for i in range(n)]) <= B
    prob_lp.solve()

    df_opt["Contact"]         = [int(x.value()) for x in x_vars]
    df_opt["Expected Return"] = np.round(df_opt["Predicted Probability"]*v,2)

    st.subheader("Optimization Results")
    st.markdown("**Contact = 1** → reach out to these donors")
    st.dataframe(df_opt)
    st.metric("Total Expected Return (USD)",
              f"{df_opt.loc[df_opt.Contact==1,'Expected Return'].sum():.2f}")
    st.metric("Total Contact Cost (USD)",
              f"{df_opt.Contact.sum()*c:.2f}")
    st.metric("Donors Selected", int(df_opt.Contact.sum()))

    st.download_button(
        "📥 Download Results CSV",
        df_opt.to_csv(index=False),
        file_name="optimization_results.csv",
        mime="text/csv"
    )

    # 9) AI Strategy Insights
    st.subheader("🤖 AI-Generated Strategy Insights")
    ai_prompt = (
        "You are a nonprofit fundraising strategist reviewing these optimization results:\n\n"
        f"{df_opt.to_csv(index=False)}\n\n"
        "Please present your analysis in three clearly labeled sections:\n"
        "1. **Key Findings:** 2–3 bullet points of the most important insights.\n"
        "2. **What This Means:** A brief, non-technical explanation.\n"
        "3. **Recommendations:** 2–3 actionable steps with brief rationale & expected impact.\n"
        "Write conversationally for non-technical readers."
    )
    st.info(get_gpt_insight(ai_prompt))

# 📅 Donation Forecasting
elif selected == "📅 Donation Forecasting":
    st.header("📅 Forecasting Monthly Donation Volume")
    df = load_data()
    df['ds'] = pd.date_range(start='2022-01-01', periods=len(df), freq='MS')
    monthly = df.groupby(df['ds'].dt.to_period("M"))['Monetary'].sum().reset_index()
    monthly.columns = ['ds', 'y']
    monthly['ds'] = monthly['ds'].dt.to_timestamp()

    model = Prophet()
    model.fit(monthly)
    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    st.pyplot(fig1)

# 💬 Conversational Chatbot
elif selected == "💬 Conversational Chatbot":
    st.header("💬 AI Assistant Chatbot")
    user_input = st.text_input("Ask a question about donors, predictions, or campaigns:")
    if user_input:
        response = get_gpt_insight(
            f"You are a Blood Donation DSS assistant. Respond to: {user_input}"
        )
        st.success(response)

# 🔁 What-If Scenario
elif selected == "🔁 What-If Scenario":
    st.header("🔁 Simulate Donor Scenario")
    rec = st.slider("Recency", 0, 100, 10)
    freq = st.slider("Frequency", 0, 20, 5)
    mon = st.slider("Monetary (mL)", 100, 2000, 500)
    age = st.slider("Age", 18, 65, 35)
    donor = pd.DataFrame([{
        "Recency": rec, "Frequency": freq, "Monetary": mon,
        "Time": 30, "Age": age, "CampaignResponse": 1,
        "Monetary_per_Freq": mon / (freq + 1),
        "Intensity": freq / (rec + 1)
    }])

    st.subheader("Donor Input")
    st.dataframe(donor)

    if "advanced_models" in st.session_state:
        model = st.session_state["advanced_models"].get("Voting Ensemble")
        scaler = st.session_state["scaler"]
        feats = st.session_state["advanced_features"]
        X_sim = scaler.transform(donor[feats])
        prob = model.predict_proba(X_sim)[0][1]
        st.metric("Predicted Repeat Donation Probability", f"{prob:.2%}")
    else:
        st.warning("Model not loaded. Run the Modeling page first.")






# Voice Assistant Tab
if selected == "🎧 Voice Assistant":
    st.header("🎧 Speak Now: Mic Input with Whisper + GPT")

    @st.cache_resource
    def load_whisper_model():
        return whisper.load_model("base")

    model = load_whisper_model()

    class AudioProcessor(AudioProcessorBase):
        def __init__(self) -> None:
            self.recording = []

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            pcm = frame.to_ndarray()
            self.recording.append(pcm)
            return frame

    ctx = webrtc_streamer(
        key="speech",
        mode="sendonly",
        audio_processor_factory=AudioProcessor,
        client_settings=ClientSettings(
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
    )

    if st.button("🔍 Transcribe"):
        if ctx.audio_processor and ctx.audio_processor.recording:
            audio = np.concatenate(ctx.audio_processor.recording).astype(np.float32)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio, 16000)
                result = model.transcribe(tmp.name)
                st.success("🗣 Transcribed text:")
                st.markdown(f"**{result['text']}**")
                try:
                    openai.api_key = st.secrets.get("OPENAI_API_KEY", "")
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": f"You are a DSS bot. Respond to this: {result['text']}"}
                        ]
                    )
                    st.info(response.choices[0].message["content"].strip())
                except Exception as e:
                    st.error(f"GPT processing failed: {e}")
        else:
            st.warning("No audio recorded yet.")
