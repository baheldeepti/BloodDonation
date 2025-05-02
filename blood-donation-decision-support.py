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
    options=[
        "🏠 Overview",
        "📊 Exploratory Analysis",
        "🤖 Modeling & Recommendations"
    ])

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

# --- PAGE: MODELING & DECISION SUPPORT ---
elif selected == "🤖 Modeling & Recommendations":
    st.title("🔍 Modeling & Decision Support")
    st.markdown(
        """
        This single page will:
        1. Engineer features & select top-7 by correlation  
        2. Check multicollinearity  
        3. Train base & ensemble models, compare via ROC  
        4. Show confusion matrix for the best model  
        5. Accept new donor data and produce personalized recommendations  
        6. Generate AI-powered insights
        """
    )

    # 1) Load data & feature engineering
    df = generate_data()
    df['Monetary_per_Freq'] = df['Monetary'] / (df['Frequency'] + 1)
    df['Intensity']         = df['Frequency'] / (df['Recency'] + 1)

    # 2) Select top-7 features by |corr| with Target
    corr_t   = df.corr()['Target'].abs().sort_values(ascending=False)
    feats    = corr_t.index[1:8].tolist()
    st.subheader("🔑 Top 7 Features")
    st.table(pd.DataFrame({
        'Feature': feats,
        '|Corr| w/ Target': corr_t[feats].round(3).values
    }))

    # 3) Multicollinearity check
    fcorr = df[feats].corr().abs()
    pairs = [
        (fcorr.index[i], fcorr.columns[j], fcorr.iloc[i, j])
        for i in range(len(fcorr)) for j in range(i)
        if fcorr.iloc[i, j] > .9
    ]
    if pairs:
        st.warning("🚨 High collinearity:")
        for f1, f2, v in pairs:
            st.write(f"• {f1} ↔ {f2}: {v:.2f}")
        st.info("Drop/combine one of each pair.")
    else:
        st.success("✅ No high collinearity.")

    # 4) Train/test split & scaling
    X = df[feats]; y = df['Target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    X_trs, X_tes = scaler.transform(X_tr), scaler.transform(X_te)

    # 5) Base models + hyperparams
    base = {
        'LogReg': LogisticRegression(max_iter=1000),
        'RF':      RandomForestClassifier(),
        'GB':      GradientBoostingClassifier(),
        'XGB':     XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'CB':      CatBoostClassifier(verbose=0),
        'SVM':     SVC(probability=True)
    }
    params = {
        'RF':  {'n_estimators':[100],'max_depth':[3,5]},
        'GB':  {'n_estimators':[100],'learning_rate':[0.05]},
        'XGB': {'n_estimators':[100],'max_depth':[3]},
        'CB':  {'depth':[4],'learning_rate':[0.05]},
        'SVM': {'C':[1.0],'kernel':['rbf']}
    }

    results = []; trained = {}
    fig_roc, ax_roc = plt.subplots(figsize=(8,6))

    # 6) Fit base models
    for name, mdl in base.items():
        if name in params:
            gs = GridSearchCV(mdl, params[name], scoring='roc_auc', cv=3)
            gs.fit(X_trs, y_tr); best = gs.best_estimator_
        else:
            best = mdl.fit(X_trs, y_tr)
        y_p = best.predict(X_tes); y_pr = best.predict_proba(X_tes)[:,1]
        fpr, tpr, _ = roc_curve(y_te, y_pr)
        auc = roc_auc_score(y_te, y_pr)
        results.append({
            'Model': name, 'AUC':round(auc,3),
            'Acc':round(accuracy_score(y_te,y_p),3),
            'F1': round(f1_score(y_te,y_p),3),
            'Prec':round(precision_score(y_te,y_p),3),
            'Rec': round(recall_score(y_te,y_p),3)
        })
        trained[name] = best
        ax_roc.plot(fpr, tpr, label=f"{name} ({auc:.2f})")

    # 7) Ensembles: soft Voting & Stacking
    top3 = [r['Model'] for r in sorted(results, key=lambda x: x['AUC'], reverse=True)[:3]]
    voting = VotingClassifier(
        estimators=[(n, trained[n]) for n in top3], voting='soft'
    ).fit(X_trs, y_tr)
    y_v, y_vp = voting.predict(X_tes), voting.predict_proba(X_tes)[:,1]
    fpr_v, tpr_v, _ = roc_curve(y_te, y_vp); auc_v = roc_auc_score(y_te, y_vp)
    results.append({'Model':'Voting','AUC':round(auc_v,3),
                    'Acc':round(accuracy_score(y_te,y_v),3),
                    'F1':round(f1_score(y_te,y_v),3),
                    'Prec':round(precision_score(y_te,y_v),3),
                    'Rec':round(recall_score(y_te,y_v),3)})
    trained['Voting'] = voting
    ax_roc.plot(fpr_v, tpr_v, '--', label=f"Voting ({auc_v:.2f})")

    stacking = StackingClassifier(
        estimators=[(n, trained[n]) for n in top3],
        final_estimator=LogisticRegression(), cv=3
    ).fit(X_trs, y_tr)
    y_s, y_sp = stacking.predict(X_tes), stacking.predict_proba(X_tes)[:,1]
    fpr_s, tpr_s, _ = roc_curve(y_te, y_sp); auc_s = roc_auc_score(y_te, y_sp)
    results.append({'Model':'Stacking','AUC':round(auc_s,3),
                    'Acc':round(accuracy_score(y_te,y_s),3),
                    'F1':round(f1_score(y_te,y_s),3),
                    'Prec':round(precision_score(y_te,y_s),3),
                    'Rec':round(recall_score(y_te,y_s),3)})
    trained['Stacking'] = stacking
    ax_roc.plot(fpr_s, tpr_s, '-.', label=f"Stacking ({auc_s:.2f})")

    # 8) Show comparison & ROC
    st.subheader("📋 Model Comparison")
    df_res = pd.DataFrame(results).sort_values('AUC', ascending=False).reset_index(drop=True)
    st.dataframe(df_res.style
        .background_gradient(subset=['AUC','Acc','F1','Prec','Rec'], cmap='Greens')
        .highlight_max(subset=['AUC','Acc','F1','Prec','Rec'], color='lightgreen')
        .highlight_min(subset=['AUC','Acc','F1','Prec','Rec'], color='salmon')
    )

    st.subheader("📉 ROC Curves")
    ax_roc.plot([0,1],[0,1],'k--'); ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
    ax_roc.legend(loc='lower right'); st.pyplot(fig_roc)

    # 9) Confusion matrix of best
    from sklearn.metrics import confusion_matrix
    best = df_res.loc[0,'Model']; m = trained[best]
    cm = confusion_matrix(y_te, m.predict(X_tes))
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set(title=f"Confusion Matrix: {best}", xlabel='Pred', ylabel='Actual')
    st.subheader("🔍 Confusion Matrix")
    st.pyplot(fig_cm)
 

    # 11) New data & recommendations
    st.subheader("🔄 Data Input & Personalized Recommendations")
    st.markdown("---\n### ➤ Provide new donor data")
    if 'manual' not in st.session_state: st.session_state['manual']=[]
    with st.form("form", clear_on_submit=True):
        r=st.number_input('Recency',0,50,10); f=st.number_input('Frequency',1,10,2)
        m=st.number_input('Monetary',0,2000,500); t=st.number_input('Time',1,100,12)
        a=st.number_input('Age',18,100,35)
        c=st.selectbox('CampaignResponse',['Yes','No'])
        if st.form_submit_button("Add"):
            st.session_state['manual'].append({
                'Recency':r,'Frequency':f,'Monetary':m,'Time':t,
                'Age':a,'CampaignResponse':1 if c=='Yes' else 0
            }); st.success("Added")

    df_new = pd.DataFrame(st.session_state['manual'])
    if df_new.empty:
        st.info("Upload CSV or add an entry.")
    else:
        st.dataframe(df_new)
        df_new['Monetary_per_Freq']=df_new['Monetary']/(df_new['Frequency']+1)
        df_new['Intensity']=df_new['Frequency']/(df_new['Recency']+1)
        Xn = scaler.transform(df_new[feats])
        out = df_new.copy()
        for name, mdl in trained.items():
            prob = mdl.predict_proba(Xn)[:,1]
            rec  = np.where(prob>0.7,'SMS','Email','Deprioritize')
            out[f'Prob_{name}']=prob.round(3)
            out[f'Rec_{name}']=rec
            out[f'Value_{name}']=(prob*150).round(2)
        st.subheader("📋 Recommendations"); st.dataframe(out)
        st.download_button("Download CSV", out.to_csv(index=False), "recs.csv","text/csv")
    # 11) AI insights
    st.subheader("🤖 AI Insights & Next Steps")
    prompt = (
        "Senior data scientist with 20 yrs exp. Given:\n\n"
        f"{df_res.to_csv(index=False)}\n\n"
        f"ConfMatrix({best}): {cm.tolist()}\n\n"
        "Recommend best model, improvements, next steps."
    )
    st.info(get_gpt_insight(prompt))

