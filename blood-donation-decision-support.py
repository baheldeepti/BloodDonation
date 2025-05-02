# blood_donation_dss_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, roc_curve, confusion_matrix
)
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
    label="Go to",
    options=[
        "🏠 Overview",
        "📊 Exploratory Analysis",
        "🤖 Modeling & Recommendations",
         "📈 Budget Optimization"]
)
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

    # Feature engineering
    df = generate_data()
    df['Monetary_per_Freq'] = df['Monetary'] / (df['Frequency'] + 1)
    df['Intensity'] = df['Frequency'] / (df['Recency'] + 1)

    corr_t = df.corr()['Target'].abs().sort_values(ascending=False)
    feats = corr_t.index[1:8].tolist()
    st.subheader("🔑 Top 7 Features")
    st.table(pd.DataFrame({
        'Feature': feats,
        '|Corr| w/ Target': corr_t[feats].round(3).values
    }))

    # Multicollinearity
    fc = df[feats].corr().abs()
    high = [(fc.index[i], fc.columns[j], fc.iloc[i,j])
            for i in range(len(fc)) for j in range(i) if fc.iloc[i,j]>0.9]
    if high:
        st.warning("🚨 High collinearity detected:")
        for f1,f2,v in high:
            st.write(f"• {f1} ↔ {f2}: {v:.2f}")
    else:
        st.success("✅ No high collinearity.")

    # Split & scale
    X, y = df[feats], df['Target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    X_trs, X_tes = scaler.transform(X_tr), scaler.transform(X_te)

    # Define models
    base = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(verbose=0),
        'Support Vector Machine': SVC(probability=True)
    }
    params = {
        'Random Forest': {'n_estimators': [100], 'max_depth': [3,5]},
        'Gradient Boosting': {'n_estimators': [100], 'learning_rate': [0.05]},
        'XGBoost': {'n_estimators': [100], 'max_depth': [3]},
        'CatBoost': {'depth': [4], 'learning_rate': [0.05]},
        'Support Vector Machine': {'C': [1.0], 'kernel': ['rbf']}
    }

    results, trained = [], {}
    fig_roc, ax_roc = plt.subplots(figsize=(8,6))

    # Train & evaluate
    for name, mdl in base.items():
        if name in params:
            gs = GridSearchCV(mdl, params[name], scoring='roc_auc', cv=3)
            gs.fit(X_trs, y_tr)
            best = gs.best_estimator_
        else:
            best = mdl.fit(X_trs, y_tr)
        y_p = best.predict(X_tes)
        y_pr = best.predict_proba(X_tes)[:,1]
        fpr, tpr, _ = roc_curve(y_te, y_pr)
        auc = roc_auc_score(y_te, y_pr)
        results.append({
            'Model': name,
            'AUC': round(auc,3),
            'Accuracy': round(accuracy_score(y_te,y_p),3),
            'F1 Score': round(f1_score(y_te,y_p),3),
            'Precision': round(precision_score(y_te,y_p),3),
            'Recall': round(recall_score(y_te,y_p),3)
        })
        trained[name] = best
        ax_roc.plot(fpr, tpr, label=f"{name} ({auc:.2f})")

    # Ensembles
    top3 = [r['Model'] for r in sorted(results, key=lambda x: x['AUC'], reverse=True)[:3]]
    voting = VotingClassifier(
        estimators=[(n,trained[n]) for n in top3], voting='soft'
    ).fit(X_trs,y_tr)
    y_vp = voting.predict_proba(X_tes)[:,1]
    fpr_v,tpr_v,_ = roc_curve(y_te,y_vp)
    auc_v = roc_auc_score(y_te,y_vp)
    results.append({
        'Model': 'Soft Voting Ensemble',
        'AUC': round(auc_v,3),
        'Accuracy': round(accuracy_score(y_te,voting.predict(X_tes)),3),
        'F1 Score': round(f1_score(y_te,voting.predict(X_tes)),3),
        'Precision': round(precision_score(y_te,voting.predict(X_tes)),3),
        'Recall': round(recall_score(y_te,voting.predict(X_tes)),3)
    })
    trained['Soft Voting Ensemble'] = voting
    ax_roc.plot(fpr_v,tpr_v,'--',label=f"Voting ({auc_v:.2f})")

    stacking = StackingClassifier(
        estimators=[(n,trained[n]) for n in top3],
        final_estimator=LogisticRegression(), cv=3
    ).fit(X_trs,y_tr)
    y_sp = stacking.predict_proba(X_tes)[:,1]
    fpr_s,tpr_s,_ = roc_curve(y_te,y_sp)
    auc_s = roc_auc_score(y_te,y_sp)
    results.append({
        'Model': 'Stacking Ensemble',
        'AUC': round(auc_s,3),
        'Accuracy': round(accuracy_score(y_te,stacking.predict(X_tes)),3),
        'F1 Score': round(f1_score(y_te,stacking.predict(X_tes)),3),
        'Precision': round(precision_score(y_te,stacking.predict(X_tes)),3),
        'Recall': round(recall_score(y_te,stacking.predict(X_tes)),3)
    })
    trained['Stacking Ensemble'] = stacking
    ax_roc.plot(fpr_s,tpr_s,'-.',label=f"Stacking ({auc_s:.2f})")

    # Display results
    st.subheader("📋 Model Comparison")
    df_res = pd.DataFrame(results).sort_values('AUC', ascending=False).reset_index(drop=True)
    st.dataframe(df_res.style
        .background_gradient(subset=['AUC','Accuracy','F1 Score','Precision','Recall'], cmap='Greens')
        .highlight_max(subset=['AUC','Accuracy','F1 Score','Precision','Recall'], color='lightgreen')
        .highlight_min(subset=['AUC','Accuracy','F1 Score','Precision','Recall'], color='salmon')
    )
    st.subheader("📉 ROC Curves")
    ax_roc.plot([0,1],[0,1],'k--')
    ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
    ax_roc.legend(loc='lower right')
    st.pyplot(fig_roc)

    # Confusion matrix
    best_model = df_res.loc[0,'Model']
    cm = confusion_matrix(y_te, trained[best_model].predict(X_tes))
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set(title=f"Confusion Matrix: {best_model}", xlabel='Predicted', ylabel='Actual')
    st.subheader("🔍 Confusion Matrix")
    st.pyplot(fig_cm)

    # Persist for optimization
    st.session_state['trained_models'] = trained
    st.session_state['scaler'] = scaler
    st.session_state['features'] = feats

    # Data input: CSV upload or manual
    st.subheader("🔄 Data Input & Personalized Recommendations")
    uploaded = st.file_uploader("Upload donor data CSV", type=['csv'])
    if uploaded:
        df_csv = pd.read_csv(uploaded)
        st.session_state['manual_entries'] = df_csv.to_dict('records')
        st.success("CSV loaded!")
    if 'manual_entries' not in st.session_state:
        st.session_state['manual_entries'] = []
    with st.form("entry_form", clear_on_submit=True):
        r = st.number_input('Recency',0,50,10)
        f = st.number_input('Frequency',1,10,2)
        m = st.number_input('Monetary',0,2000,500)
        t = st.number_input('Time',1,100,12)
        a = st.number_input('Age',18,100,35)
        c = st.selectbox('CampaignResponse',['Yes','No'])
        if st.form_submit_button("Add"):
            st.session_state['manual_entries'].append({
                'Recency':r,'Frequency':f,'Monetary':m,
                'Time':t,'Age':a,'CampaignResponse':1 if c=='Yes' else 0
            })
            st.success("Entry added!")

    df_new = pd.DataFrame(st.session_state['manual_entries'])
    if df_new.empty:
        st.info("No data: upload CSV or add entries above.")
    else:
        df_new['Monetary_per_Freq'] = df_new['Monetary']/(df_new['Frequency']+1)
        df_new['Intensity'] = df_new['Frequency']/(df_new['Recency']+1)
        Xn = scaler.transform(df_new[feats])
        out = df_new.copy()
        for name, mdl in trained.items():
            prob = mdl.predict_proba(Xn)[:,1]
            conditions = [prob>0.7, (prob>0.4)&(prob<=0.7)]
            choices = ['SMS','Email']
            rec = np.select(conditions, choices, default='Deprioritize')
            out[f'Prob_{name}'] = np.round(prob,3)
            out[f'Rec_{name}'] = rec
            out[f'Value_{name}'] = np.round(prob*150,2)

        st.subheader("📋 Recommendations")
        st.dataframe(out)
        st.download_button("Download CSV", out.to_csv(index=False), "recs.csv", "text/csv")

        # AI insights & next steps
        st.subheader("🤖 AI Insights & Next Steps")
        prompt = (
            "Senior data scientist with 20 yrs exp. Given:\n\n"
            f"{df_res.to_csv(index=False)}\n\n"
            f"Confusion Matrix ({best_model}): {cm.tolist()}\n\n"
            "Recommend best model, improvements, next steps."
        )
        st.info(get_gpt_insight(prompt))

# --- PAGE 5: OPTIMIZATION ---
elif selected == "📈 Budget Optimization":
    st.title("📈 Budget Optimization")
    df_in = pd.DataFrame(st.session_state.get('manual_entries', []))
    if df_in.empty:
        st.warning("No donor data available. Please add entries in Modeling & Recommendations.")
    else:
        models = st.session_state['trained_models']
        scaler = st.session_state['scaler']
        feats = st.session_state['features']

        model_name = st.selectbox("Choose model", list(models.keys()))
        model = models[model_name]
        X_opt = scaler.transform(df_in[feats])
        p_i = model.predict_proba(X_opt)[:,1]
        df_opt = df_in.copy()
        df_opt['p_i'] = p_i

        st.subheader("💲 Set Parameters")
        v = st.number_input("Value per successful donation (USD)", value=150)
        c = st.number_input("Cost per contact (USD)", value=1)
        B = st.number_input("Total budget (USD)", value=100)

        n = len(df_opt)
        prob_lp = LpProblem("donor_allocation", LpMaximize)
        x_vars = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]
        prob_lp += lpSum([x_vars[i]*(df_opt.loc[i,'p_i']*v - c) for i in range(n)])
        prob_lp += lpSum([x_vars[i]*c for i in range(n)]) <= B
        prob_lp.solve()

        df_opt['select'] = [int(x_vars[i].value()) for i in range(n)]
        df_opt['expected_value'] = df_opt['p_i'] * v

        st.subheader("📊 Optimization Results")
        st.dataframe(df_opt)
        st.metric("Total Expected Savings (USD)", f"{df_opt.loc[df_opt['select']==1,'expected_value'].sum():.2f}")
        st.metric("Total Contact Cost (USD)", f"{df_opt['select'].sum()*c:.2f}")
        st.metric("Donors Selected", int(df_opt['select'].sum()))

        st.download_button(
            "📥 Download Optimization Results",
            df_opt.to_csv(index=False),
            file_name='optimization_results.csv',
            mime='text/csv'
        )
