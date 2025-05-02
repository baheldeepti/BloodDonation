# blood_donation_dss_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import plotly.express as px
import openai

# Streamlit page configuration
st.set_page_config(page_title="🩸 Blood Donation DSS", layout="wide")

# Navigation menu using Streamlit's native radio buttons
st.sidebar.title("Navigation")
selected = st.sidebar.radio(
    label="Go to", 
    options=["Introduction", "Exploratory Analysis", "Predictive Modeling", "Decision Support"]
)

# Dummy data generator
@st.cache_data
def generate_data(n=1000):
    return pd.DataFrame({
        'Recency': np.random.randint(0, 50, n),
        'Frequency': np.random.randint(1, 10, n),
        'Monetary': np.random.randint(250, 1250, n),
        'Time': np.random.randint(2, 100, n),
        'Age': np.random.randint(18, 65, n),
        'CampaignResponse': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'Target': np.random.randint(0, 2, n)
    })

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
if selected == "Introduction":
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
elif selected == "Exploratory Analysis":
    st.title("📊 Exploratory Data Analysis")
    df = generate_data()



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

        # Numeric distributions
    st.subheader("📈 Numeric Distributions")
    cols = ['Recency', 'Frequency', 'Monetary', 'Time', 'Age']
    for col in cols:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

    # AI insights
    st.subheader("💡 AI-Generated Insights")
    prompt = (
        "Act as a Data Analyst and Generate key insights from a blood donation dataset with columns: Recency, Frequency, Monetary, "
        "Time, Age, CampaignResponse, Target, based on observed distributions and correlations."
    )
    ins = get_gpt_insight(prompt)
    st.info(ins)

# --- PAGE 3: PREDICTIVE MODELING ---
elif selected == "Predictive Modeling":
    st.title("🔍 Predictive Modeling")
    df = generate_data()
    df['Monetary_per_Freq'] = df['Monetary']/(df['Frequency']+1)
    df['Intensity'] = df['Frequency']/(df['Recency']+1)
    corr = df.corr()['Target'].abs().sort_values(ascending=False)
    feats = corr.index[1:8].tolist()
    X = df[feats]
    y = df['Target']

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_tr = scaler.transform(X_train)
    X_te = scaler.transform(X_test)

    # Models and hyperparameters
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(verbose=0),
        'SVM': SVC(probability=True)
    }
    params = {
        'Random Forest': {'n_estimators': [100], 'max_depth': [3, 5]},
        'Gradient Boosting': {'n_estimators': [100], 'learning_rate': [0.05]},
        'XGBoost': {'n_estimators': [100], 'max_depth': [3]},
        'CatBoost': {'depth': [4], 'learning_rate': [0.05]},
        'SVM': {'C': [1.0], 'kernel': ['rbf']}
    }

    results = []
    fig_roc, ax_roc = plt.subplots(figsize=(10, 6))
    trained_models = {}

    for name, model in models.items():
        if name in params:
            grid = GridSearchCV(model, params[name], scoring='roc_auc', cv=3)
            grid.fit(X_tr, y_train)
            best = grid.best_estimator_
        else:
            best = model.fit(X_tr, y_train)

        y_pred = best.predict(X_te)
        y_proba = best.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            'Model': name,
            'AUC': round(auc, 3),
            'Accuracy': round(accuracy_score(y_test, y_pred), 3),
            'F1 Score': round(f1_score(y_test, y_pred), 3),
            'Precision': round(precision_score(y_test, y_pred), 3),
            'Recall': round(recall_score(y_test, y_pred), 3)
        })
        trained_models[name] = best
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    # Display model comparison
    st.subheader("📋 Model Comparison")
    results_df = pd.DataFrame(results)
    results_df['Recommended'] = ['*' if i==0 else '' for i in results_df.index]
    metric_cols = ['AUC','Accuracy','F1 Score','Precision','Recall']
    styled_df = results_df.style         .background_gradient(subset=metric_cols, cmap='Greens')        .highlight_max(subset=metric_cols, color='lightgreen')         .highlight_min(subset=metric_cols, color='salmon')
    st.dataframe(styled_df)
 

    # Display ROC curves
    st.subheader("📉 ROC Curves")
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend()
    st.pyplot(fig_roc)

    # GPT recommendation
    st.subheader("🤖 GPT-3.5 Recommendation")
    prompt = "Given this model comparison table, recommend the best model." + results_df.to_csv(index=False)
    recommendation = get_gpt_insight(prompt)
    st.success(recommendation)

    # Save to session state
    st.session_state['trained_models'] = trained_models
    st.session_state['scaler'] = scaler
    st.session_state['features'] = feats

# --- PAGE 4: DECISION SUPPORT ---
elif selected == "Decision Support":
    st.title("💼 Decision Support System")

    trained_models = st.session_state.get('trained_models', {})
    scaler = st.session_state.get('scaler', None)
    feats = st.session_state.get('features', [])

    if not trained_models or scaler is None or not feats:
        st.warning("Please run the Predictive Modeling page first.")
    else:
        # Model selection
        models_sel = st.multiselect(
            "Select model(s)",
            options=list(trained_models.keys()),
            default=list(trained_models.keys())
        )
        
        # Data input: CSV or manual multiple entries
        uploaded_file = st.file_uploader("Upload donor data CSV", type='csv')
        if uploaded_file:
            df_in = pd.read_csv(uploaded_file)
        else:
            if 'manual_entries' not in st.session_state:
                st.session_state['manual_entries'] = []
            with st.form(key='manual_entry_form'):
                r = st.number_input('Recency', 0, 50, 10)
                f = st.number_input('Frequency', 1, 10, 2)
                m = st.number_input('Monetary (mL)', 250, 1250, 500)
                t = st.number_input('Time', 1, 100, 12)
                a = st.number_input('Age', 18, 65, 35)
                c = st.selectbox('Responded to campaign?', ['Yes', 'No'])
                submitted = st.form_submit_button('Add Entry')
                if submitted:
                    st.session_state['manual_entries'].append({
                        'Recency': r,
                        'Frequency': f,
                        'Monetary': m,
                        'Time': t,
                        'Age': a,
                        'CampaignResponse': 1 if c == 'Yes' else 0
                    })
            df_in = pd.DataFrame(st.session_state['manual_entries'])

        # --- RE-COMPUTE ENGINEERED FEATURES ---
        df_in['Monetary_per_Freq'] = df_in['Monetary'] / (df_in['Frequency'] + 1)
        df_in['Intensity'] = df_in['Frequency'] / (df_in['Recency'] + 1)

        st.subheader("🔢 Input Data")
        st.dataframe(df_in)

        # Scale & predict
        X_input = scaler.transform(df_in[feats])
        out = df_in.copy()
        for name in models_sel:
            model = trained_models[name]
            prob = model.predict_proba(X_input)[:, 1]
            rec = np.where(prob > 0.7, 'SMS Campaign', 
                           np.where(prob > 0.4, 'Email Reminder', 'Deprioritize'))
            out[f'Prob_{name}'] = np.round(prob, 3)
            out[f'Rec_{name}'] = rec
            out[f'Value_{name}'] = np.round(prob * 150, 2)

        st.subheader("📋 Recommendations")
        st.dataframe(out)
        st.download_button(
            "📥 Download Recommendations",
            out.to_csv(index=False),
            file_name='donor_recommendations.csv',
            mime='text/csv'
        )
   
