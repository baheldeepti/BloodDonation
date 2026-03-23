import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

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
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary

# Optional imports: do not crash app if unavailable
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_IMPORT_AVAILABLE = True
except Exception:
    OPENAI_IMPORT_AVAILABLE = False


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🩸 Blood Donation DSS", layout="wide")


# ---------------- OPENAI SAFE INIT ----------------
client = None
if OPENAI_IMPORT_AVAILABLE and "OPENAI_API_KEY" in st.secrets:
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception:
        client = None


def get_gpt_insight(prompt: str) -> str:
    if client is None:
        return "⚠️ OpenAI key not configured or OpenAI package unavailable."
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


# ---------------- MODEL HELPERS ----------------
@st.cache_resource
def prepare_training_data():
    df = load_data().copy()
    df["Monetary_per_Freq"] = df["Monetary"] / (df["Frequency"] + 1)
    df["Intensity"] = df["Frequency"] / (df["Recency"] + 1)

    corr_t = df.corr(numeric_only=True)["Target"].abs().sort_values(ascending=False)
    feats = corr_t.index[1:8].tolist()

    X = df[feats]
    y = df["Target"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_tr)
    X_trs = scaler.transform(X_tr)
    X_tes = scaler.transform(X_te)

    return df, feats, X, y, X_tr, X_te, y_tr, y_te, scaler, X_trs, X_tes


@st.cache_resource
def train_basic_models():
    df, feats, X, y, X_tr, X_te, y_tr, y_te, scaler, X_trs, X_tes = prepare_training_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB()
    }

    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(verbose=0, random_state=42)

    results = []
    trained_models = {}

    for name, mdl in models.items():
        mdl.fit(X_trs, y_tr)
        trained_models[name] = mdl

        y_p = mdl.predict(X_tes)
        y_pr = mdl.predict_proba(X_tes)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, y_p),
            "Precision": precision_score(y_te, y_p, zero_division=0),
            "Recall": recall_score(y_te, y_p, zero_division=0),
            "F1 Score": f1_score(y_te, y_p, zero_division=0),
            "ROC AUC": roc_auc_score(y_te, y_pr)
        })

    df_res = pd.DataFrame(results).sort_values("ROC AUC", ascending=False).reset_index(drop=True)

    return {
        "df": df,
        "feats": feats,
        "X": X,
        "y": y,
        "X_tr": X_tr,
        "X_te": X_te,
        "y_tr": y_tr,
        "y_te": y_te,
        "scaler": scaler,
        "X_trs": X_trs,
        "X_tes": X_tes,
        "trained_models": trained_models,
        "df_res": df_res,
    }


@st.cache_resource
def train_advanced_models():
    basic = train_basic_models()
    X = basic["X"]
    y = basic["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    model_params = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000),
            {"classifier__C": [0.1, 1, 10]}
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"classifier__n_estimators": [100], "classifier__max_depth": [5, 15]}
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {"classifier__n_estimators": [100], "classifier__learning_rate": [0.05, 0.1]}
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=42),
            {"classifier__n_estimators": [100, 200]}
        ),
        "XGBoost": (
            XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
            {"classifier__n_estimators": [100], "classifier__max_depth": [3, 5]}
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {"classifier__C": [0.1, 1], "classifier__kernel": ["rbf"]}
        ),
        "Naive Bayes": (
            GaussianNB(),
            {}
        )
    }

    results = []
    trained_models = {}

    for name, (estimator, params) in model_params.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", estimator)
        ])

        if params:
            grid = GridSearchCV(
                pipe, params, cv=5, scoring="roc_auc", n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            cv_auc = grid.best_score_
        else:
            pipe.fit(X_train, y_train)
            best_model = pipe
            cv_auc = cross_val_score(
                pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
            ).mean()

        trained_models[name] = best_model

        y_pred = best_model.predict(X_test)
        if hasattr(best_model.named_steps["classifier"], "predict_proba"):
            y_proba = best_model.predict_proba(X_test)[:, 1]
        else:
            y_proba = best_model.decision_function(X_test)

        results.append({
            "Model": name,
            "CV ROC AUC": round(cv_auc, 4),
            "Test Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Test Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Test Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "Test F1 Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "Test ROC AUC": round(roc_auc_score(y_test, y_proba), 4)
        })

    top3 = sorted(results, key=lambda x: x["CV ROC AUC"], reverse=True)[:3]
    ensemble = VotingClassifier(
        estimators=[(r["Model"], trained_models[r["Model"]]) for r in top3],
        voting="soft"
    )
    ensemble_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", ensemble)
    ])
    ensemble_pipe.fit(X_train, y_train)

    y_pred_e = ensemble_pipe.predict(X_test)
    y_proba_e = ensemble_pipe.predict_proba(X_test)[:, 1]
    ensemble_cv_auc = cross_val_score(
        ensemble_pipe,
        X,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1
    ).mean()

    results.append({
        "Model": "Voting Ensemble",
        "CV ROC AUC": round(ensemble_cv_auc, 4),
        "Test Accuracy": round(accuracy_score(y_test, y_pred_e), 4),
        "Test Precision": round(precision_score(y_test, y_pred_e, zero_division=0), 4),
        "Test Recall": round(recall_score(y_test, y_pred_e, zero_division=0), 4),
        "Test F1 Score": round(f1_score(y_test, y_pred_e, zero_division=0), 4),
        "Test ROC AUC": round(roc_auc_score(y_test, y_proba_e), 4)
    })

    df_results = pd.DataFrame(results).sort_values(by="CV ROC AUC", ascending=False).reset_index(drop=True)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "trained_models": trained_models,
        "ensemble_pipe": ensemble_pipe,
        "y_proba_e": y_proba_e,
        "df_results": df_results
    }


# ---------------- NAV ----------------
tabs = [
    "🏠 Overview",
    "📊 Exploratory Analysis",
    "🤖 Modeling & Recommendations",
    "📈 Budget Optimization"
]

if TORCH_AVAILABLE:
    tabs.append("🤠 Deep Learning (PyTorch)")

default_index = tabs.index(st.session_state.get("selected_tab", "🏠 Overview"))
selected = st.sidebar.radio("Go to", tabs, index=default_index)
st.session_state.selected_tab = selected


# ---------------- PAGE 1: INTRODUCTION ----------------
if selected == "🏠 Overview":
    st.title("🩸 Blood Donation Decision Support System")
    st.image(
        "https://news-notes.ufhealth.org/files/2020/02/foweb_blood_donation_art_pg26.jpg",
        caption="Donate Blood, Save Lives",
        width=700
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

    if not CATBOOST_AVAILABLE:
        st.info("ℹ️ CatBoost is unavailable in this deployment, so the rest of the app will run without it.")
    if not TORCH_AVAILABLE:
        st.info("ℹ️ PyTorch is unavailable in this deployment, so the Deep Learning tab is hidden.")


# ---------------- PAGE 2: EDA ----------------
elif selected == "📊 Exploratory Analysis":
    st.title("📊 Exploratory Data Analysis")
    df = load_data().copy()

    st.subheader("📌 KPI Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("% Repeat Donors", f"{df['Target'].mean() * 100:.2f}%")
    col3.metric("Avg Donation Volume", f"{df['Monetary'].mean():.2f} ml")

    st.subheader("📊 Categorical Distributions")
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.pie(df, names="CampaignResponse", title="Campaign Response Rate")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        target_counts = df["Target"].value_counts().rename_axis("Target").reset_index(name="Count")
        fig2 = px.bar(target_counts, x="Target", y="Count", title="Repeat Donation Counts")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📊 Boxplots by Donation Outcome")
    b1, b2 = st.columns(2)
    with b1:
        fig4 = px.box(df, x="Target", y="Monetary", title="Monetary by Outcome")
        st.plotly_chart(fig4, use_container_width=True)
    with b2:
        fig5 = px.box(df, x="Target", y="Recency", title="Recency by Outcome")
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("🔗 Frequency vs Monetary")
    fig6 = px.scatter(
        df,
        x="Frequency",
        y="Monetary",
        color=df["Target"].map({0: "No", 1: "Yes"}),
        title="Frequency vs Monetary by Repeat Donation"
    )
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("💡 AI-Generated Insights")
    prompt = (
        "Act as a Data Analyst and Generate key insights from a blood donation dataset with columns: "
        "Recency, Frequency, Monetary, Time, Age, CampaignResponse, Target, based on observed distributions and correlations."
    )
    st.info(get_gpt_insight(prompt))


# ---------------- PAGE 3: MODELING ----------------
elif selected == "🤖 Modeling & Recommendations":
    st.title("🔍 Modeling & Recommendations")
    st.markdown(
        """
        1. Feature engineering & select top-7  
        2. Multicollinearity check  
        3. Train models & ensembles  
        4. ROC & confusion matrix  
        5. AI insights & next steps  
        """
    )

    basic = train_basic_models()
    advanced = train_advanced_models()

    df = basic["df"]
    feats = basic["feats"]
    X_tes = basic["X_tes"]
    y_te = basic["y_te"]
    trained_models_basic = basic["trained_models"]
    df_res = basic["df_res"]
    scaler = basic["scaler"]

    st.subheader("🔑 Top 7 Features")
    corr_t = df.corr(numeric_only=True)["Target"].abs().sort_values(ascending=False)
    st.table(pd.DataFrame({
        "Feature": feats,
        "|Corr| w/ Target": corr_t[feats].round(3).values
    }))

    fc = df[feats].corr().abs()
    high = [
        (fc.index[i], fc.columns[j], fc.iloc[i, j])
        for i in range(len(fc)) for j in range(i) if fc.iloc[i, j] > 0.9
    ]
    if high:
        st.warning("🚨 High collinearity detected:")
        for f1, f2, v in high:
            st.write(f"• {f1} ↔ {f2}: {v:.2f}")
    else:
        st.success("✅ No high collinearity.")

    st.subheader("📉 ROC Curves")
    fig_basic, ax_basic = plt.subplots(figsize=(8, 6))
    for name, mdl in trained_models_basic.items():
        y_pr = mdl.predict_proba(X_tes)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, y_pr)
        ax_basic.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y_te, y_pr):.2f})")

    ax_basic.plot([0, 1], [0, 1], "k--")
    ax_basic.set_xlabel("False Positive Rate")
    ax_basic.set_ylabel("True Positive Rate")
    ax_basic.legend(loc="lower right")
    st.pyplot(fig_basic)

    st.subheader("📋 Model Comparison")
    metrics_basic = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    styled_basic = (
        df_res.style
        .background_gradient(subset=metrics_basic, cmap="Greens")
        .highlight_max(subset=metrics_basic, color="lightgreen")
        .highlight_min(subset=metrics_basic, color="salmon")
    )
    st.dataframe(styled_basic, use_container_width=True)

    # Persist for optimization
    st.session_state["basic_models"] = trained_models_basic
    st.session_state["basic_features"] = feats
    st.session_state["scaler"] = scaler

    # Advanced section
    trained_models_adv = advanced["trained_models"]
    ensemble_pipe = advanced["ensemble_pipe"]
    y_test_adv = advanced["y_test"]
    X_test_adv = advanced["X_test"]
    y_proba_e = advanced["y_proba_e"]
    df_results = advanced["df_results"]

    st.subheader("📊 Advanced Tuning & Ensembling Results")
    metrics_adv = [
        "CV ROC AUC", "Test Accuracy", "Test Precision",
        "Test Recall", "Test F1 Score", "Test ROC AUC"
    ]
    styled_adv = (
        df_results.style
        .background_gradient(subset=metrics_adv, cmap="Greens")
        .highlight_max(subset=metrics_adv, color="lightgreen")
        .highlight_min(subset=metrics_adv, color="salmon")
    )
    st.dataframe(styled_adv, use_container_width=True)

    fig_adv, ax_adv = plt.subplots(figsize=(10, 8))
    for name, model in trained_models_adv.items():
        if hasattr(model.named_steps["classifier"], "predict_proba"):
            y_prob = model.predict_proba(X_test_adv)[:, 1]
        else:
            y_prob = model.decision_function(X_test_adv)
        fpr, tpr, _ = roc_curve(y_test_adv, y_prob)
        ax_adv.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")

    fpr_e, tpr_e, _ = roc_curve(y_test_adv, y_proba_e)
    ax_adv.plot(fpr_e, tpr_e, "--", lw=2, label=f"Voting Ensemble (AUC = {auc(fpr_e, tpr_e):.2f})")
    ax_adv.plot([0, 1], [0, 1], "k--", lw=1)
    ax_adv.set_xlim([0, 1])
    ax_adv.set_ylim([0, 1.05])
    ax_adv.set_xlabel("False Positive Rate")
    ax_adv.set_ylabel("True Positive Rate")
    ax_adv.set_title("ROC Curves for All Models")
    ax_adv.legend(loc="lower right")
    st.pyplot(fig_adv)

    st.session_state["advanced_models"] = {**trained_models_adv, "Voting Ensemble": ensemble_pipe}
    st.session_state["advanced_features"] = feats

    st.subheader("🤖 AI Model Comparison & Recommendations")
    ai_prompt = (
        "You are a Senior Data Scientist with 20 years of experience. "
        "Below are two CSV tables: the original basic-model results (`df_res`) and the advanced "
        "hyperparameter tuning & ensemble results (`df_results`).\n\n"
        f"Original Models:\n{df_res.to_csv(index=False)}\n\n"
        f"Tuned & Ensemble Models:\n{df_results.to_csv(index=False)}\n\n"
        "1) Create a combined markdown table with columns: Model, Best Use Case, Improvement?, Δ ROC AUC.\n"
        "2) Then provide 2–3 recommendations with Why and Impact.\n"
        "Format everything in clear markdown."
    )
    st.info(get_gpt_insight(ai_prompt))


# ---------------- PAGE 4: BUDGET OPTIMIZATION ----------------
elif selected == "📈 Budget Optimization":
    st.title("📈 Campaign Budget Optimization")
    st.markdown(
        """
        **What is this?**  
        Decide which donors to contact under a fixed outreach budget for maximum expected return.

        **Where does donor data come from?**  
        Upload a CSV or enter donors manually below.
        """
    )

    # Ensure models exist even if user lands here first
    if "basic_models" not in st.session_state or "advanced_models" not in st.session_state:
        basic = train_basic_models()
        advanced = train_advanced_models()
        st.session_state["basic_models"] = basic["trained_models"]
        st.session_state["basic_features"] = basic["feats"]
        st.session_state["advanced_models"] = {**advanced["trained_models"], "Voting Ensemble": advanced["ensemble_pipe"]}
        st.session_state["advanced_features"] = basic["feats"]
        st.session_state["scaler"] = basic["scaler"]

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
        st.info("Please upload a CSV or add at least one donor.")
        st.stop()

    st.subheader("Donation Data")
    st.dataframe(df_in, use_container_width=True)
    st.markdown("**Glossary:** Recency, Frequency, Monetary, Time, Age, CampaignResponse")

    df_in["Monetary_per_Freq"] = df_in["Monetary"] / (df_in["Frequency"] + 1)
    df_in["Monetary_per_Donation"] = df_in["Monetary"] / (df_in["Frequency"] + 1)
    df_in["Intensity"] = df_in["Frequency"] / (df_in["Recency"] + 1)
    df_in["Donation_Intensity"] = df_in["Frequency"] / (df_in["Recency"] + 1)

    st.subheader("Choose Predictive Model Set")
    choice = st.radio("Which model family?", ["Original Models", "Advanced Models"])

    if choice == "Original Models":
        models = st.session_state["basic_models"]
        feats = st.session_state["basic_features"]
    else:
        models = st.session_state["advanced_models"]
        feats = st.session_state["advanced_features"]

    missing = [col for col in feats if col not in df_in.columns]
    if missing:
        st.error(f"Missing features: {missing}. Please include them and retry.")
        st.stop()

    scaler = st.session_state["scaler"]
    model_name = st.selectbox("Pick a model", list(models.keys()))
    model = models[model_name]

    # Handle basic models vs advanced pipelines
    if hasattr(model, "named_steps"):
        X_opt_input = df_in[feats]
        p_i = model.predict_proba(X_opt_input)[:, 1]
    else:
        X_opt = scaler.transform(df_in[feats])
        p_i = model.predict_proba(X_opt)[:, 1]

    df_opt = df_in.copy()
    df_opt["Predicted Probability"] = np.round(p_i, 3)
    st.subheader("Predicted Donation Probability")
    st.dataframe(df_opt, use_container_width=True)

    st.subheader("Budget & Values")
    v = st.number_input("Value per donation (USD)", value=150)
    c = st.number_input("Cost per contact (USD)", value=1)
    B = st.number_input("Total budget (USD)", value=100)

    n = len(df_opt)
    prob_lp = LpProblem("donor_alloc", LpMaximize)
    x_vars = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]

    prob_lp += lpSum([x_vars[i] * (p_i[i] * v - c) for i in range(n)])
    prob_lp += lpSum([x_vars[i] * c for i in range(n)]) <= B
    prob_lp.solve()

    df_opt["Contact"] = [int(x.value()) if x.value() is not None else 0 for x in x_vars]
    df_opt["Expected Return"] = np.round(df_opt["Predicted Probability"] * v, 2)

    st.subheader("Optimization Results")
    st.markdown("**Contact = 1** → reach out to these donors")
    st.dataframe(df_opt, use_container_width=True)
    st.metric(
        "Total Expected Return (USD)",
        f"{df_opt.loc[df_opt.Contact == 1, 'Expected Return'].sum():.2f}"
    )
    st.metric(
        "Total Contact Cost (USD)",
        f"{df_opt.Contact.sum() * c:.2f}"
    )
    st.metric("Donors Selected", int(df_opt.Contact.sum()))

    st.download_button(
        "📥 Download Results CSV",
        df_opt.to_csv(index=False),
        file_name="optimization_results.csv",
        mime="text/csv"
    )

    st.subheader("🤖 AI-Generated Strategy Insights")
    ai_prompt = (
        "You are a nonprofit fundraising strategist reviewing these optimization results:\n\n"
        f"{df_opt.to_csv(index=False)}\n\n"
        "Please present your analysis in three clearly labeled sections:\n"
        "1. Key Findings\n"
        "2. What This Means\n"
        "3. Recommendations\n"
        "Write conversationally for non-technical readers."
    )
    st.info(get_gpt_insight(ai_prompt))


# ---------------- PAGE 5: OPTIONAL PYTORCH ----------------
elif selected == "🤠 Deep Learning (PyTorch)":
    st.title("🤠 Deep Learning with PyTorch")
    st.markdown("Build and evaluate a neural network for donor prediction using PyTorch.")

    if not TORCH_AVAILABLE:
        st.error("PyTorch is not available in this deployment.")
        st.stop()

    df = load_data().copy()
    df["Monetary_per_Freq"] = df["Monetary"] / (df["Frequency"] + 1)
    df["Intensity"] = df["Frequency"] / (df["Recency"] + 1)

    features = [
        "Recency", "Frequency", "Monetary", "Time", "Age",
        "CampaignResponse", "Monetary_per_Freq", "Intensity"
    ]
    X = df[features].values
    y = df["Target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    learning_rates = [0.001, 0.005]
    batch_sizes = [16, 32]
    num_layers_options = [2, 3]

    class Net(nn.Module):
        def __init__(self, input_dim, num_layers):
            super().__init__()
            self.hidden = nn.ModuleList()
            self.hidden.append(nn.Linear(input_dim, 16))
            for _ in range(num_layers - 1):
                self.hidden.append(nn.Linear(16, 16))
            self.out = nn.Linear(16, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            for layer in self.hidden:
                x = torch.relu(layer(x))
            return self.sigmoid(self.out(x))

    best_auc = 0
    best_model = None
    best_config = None

    st.subheader("🔧 Hyperparameter Tuning & Ensemble Models")
    for num_layers in num_layers_options:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                st.write(f"Training model with layers={num_layers}, lr={lr}, batch_size={batch_size}")

                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                models = [Net(input_dim=X.shape[1], num_layers=num_layers) for _ in range(3)]
                loss_fn = nn.BCELoss()
                optimizers = [optim.Adam(m.parameters(), lr=lr) for m in models]

                for i, model in enumerate(models):
                    model.train()
                    for _ in range(10):
                        for xb, yb in train_loader:
                            pred = model(xb)
                            loss = loss_fn(pred, yb)
                            optimizers[i].zero_grad()
                            loss.backward()
                            optimizers[i].step()

                for model in models:
                    model.eval()

                with torch.no_grad():
                    preds = [model(X_test_tensor).numpy().flatten() for model in models]
                    y_pred = np.mean(preds, axis=0)
                    auc_score = roc_auc_score(y_test, y_pred)
                    if auc_score > best_auc:
                        best_auc = auc_score
                        best_model = models
                        best_config = (num_layers, lr, batch_size)

    st.subheader("📊 Best Ensemble Model Performance")
    with torch.no_grad():
        preds = [model(X_test_tensor).numpy().flatten() for model in best_model]
        y_pred = np.mean(preds, axis=0)
        y_pred_class = (y_pred > 0.5).astype(int)

    st.write(f"Best Config - Layers: {best_config[0]}, LR: {best_config[1]}, Batch Size: {best_config[2]}")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_class):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred_class, zero_division=0):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred_class, zero_division=0):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred_class, zero_division=0):.2f}")
    st.write(f"ROC AUC: {roc_auc_score(y_test, y_pred):.2f}")

    st.subheader("🤖 AI-Generated Insights")
    perf_summary = (
        f"Model Accuracy: {accuracy_score(y_test, y_pred_class):.2f}\n"
        f"Precision: {precision_score(y_test, y_pred_class, zero_division=0):.2f}\n"
        f"Recall: {recall_score(y_test, y_pred_class, zero_division=0):.2f}\n"
        f"F1 Score: {f1_score(y_test, y_pred_class, zero_division=0):.2f}\n"
        f"ROC AUC: {roc_auc_score(y_test, y_pred):.2f}"
    )
    ai_prompt = (
        "You are a seasoned ML Engineer. Based on the following performance metrics of an ensemble of fine-tuned "
        "PyTorch-based neural networks trained to predict blood donor retention, provide:\n"
        "1. Model Summary\n"
        "2. Improvement Suggestions\n"
        "3. Business Implications\n\n"
        f"{perf_summary}"
    )
    st.info(get_gpt_insight(ai_prompt))
