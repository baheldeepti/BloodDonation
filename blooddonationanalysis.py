# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Simulate enriched dataset
np.random.seed(42)
n_samples = 748  # same as UCI dataset

# Original UCI dataset fields
data = pd.DataFrame({
    'Recency': np.random.randint(0, 50, n_samples),
    'Frequency': np.random.randint(1, 10, n_samples),
    'Monetary': np.random.randint(250, 1250, n_samples),
    'Time': np.random.randint(2, 100, n_samples),
    'Target': np.random.randint(0, 2, n_samples)
})

# Add demographic data
data['Age'] = np.random.randint(18, 65, n_samples)
data['Gender'] = np.random.choice(['Male', 'Female'], n_samples)
data['BloodType'] = np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'], n_samples)

# Behavioral data
data['CommunicationPreference'] = np.random.choice(['Email', 'SMS', 'Phone'], n_samples)
data['CampaignResponse'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

# Time-series features
data['AvgDonationInterval'] = data['Time'] / data['Frequency']
data['SeasonalDonation'] = np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n_samples)

# 2. Encode categorical features
data = pd.get_dummies(data, columns=['Gender', 'BloodType', 'CommunicationPreference', 'SeasonalDonation'], drop_first=True)

# 3. Feature Engineering
data['Recency_Frequency'] = data['Recency'] * data['Frequency']
data['Monetary_per_Interval'] = data['Monetary'] / (data['AvgDonationInterval'] + 1)

# 4. Feature selection: Remove low-variance features
features = data.drop(columns=['Target'])
target = data['Target']

selector = VarianceThreshold(threshold=0.01)
features_var = selector.fit_transform(features)
selected_columns_var = features.columns[selector.get_support()]
features = pd.DataFrame(features_var, columns=selected_columns_var)

# 5. Select top 7 important features using RandomForest
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(features, target)
importances = forest.feature_importances_
importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
top_features = importance_df.head(7)['Feature'].tolist()
features = features[top_features]

# 6. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.3, random_state=42)

# 8. Define models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'CatBoost': CatBoostClassifier(verbose=0)
}

# 9. Hyperparameter tuning grids
param_grids = {
    'RandomForest': {'n_estimators': [50, 100], 'max_depth': [3, 5, 10]},
    'XGBoost': {'n_estimators': [50, 100], 'max_depth': [3, 5]},
    'CatBoost': {'depth': [3, 5], 'learning_rate': [0.03, 0.1]}
}

# 10. Train models and evaluate
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    if name in param_grids:
        grid = GridSearchCV(model, param_grids[name], cv=3, scoring='roc_auc')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
    else:
        model.fit(X_train, y_train)
        best_model = model

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"{name} ROC AUC: {auc_score:.2f}")
    print(classification_report(y_test, y_pred))
    results[name] = (best_model, auc_score)

# 11. SHAP explainability for best model
best_model_name = max(results.items(), key=lambda x: x[1][1])[0]
best_model = results[best_model_name][0]
print(f"\nBest model: {best_model_name}")

explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test[:100])

# Visualize SHAP values
shap.plots.beeswarm(shap_values, max_display=10)


# Prophet for time-series forecasting
from prophet import Prophet
import matplotlib.pyplot as plt

# --- Step 1: Simulate monthly donation volume ---
# Assume each donor gives at most one donation per month
# Randomly assign months to donation records
date_range = pd.date_range(start="2022-01-01", periods=24, freq='M')
donation_dates = np.random.choice(date_range, size=len(data))
data['DonationMonth'] = donation_dates

# Aggregate monthly donation counts
monthly_donations = data.groupby('DonationMonth').size().reset_index(name='DonationCount')

# Rename for Prophet
df_prophet = monthly_donations.rename(columns={'DonationMonth': 'ds', 'DonationCount': 'y'})

# --- Step 2: Forecast using Prophet ---
model = Prophet()
model.fit(df_prophet)

# Future dates
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# --- Step 3: Plot the forecast ---
fig1 = model.plot(forecast)
plt.title('Forecasted Blood Donations (Next 12 Months)')
plt.xlabel('Date')
plt.ylabel('Predicted Donation Volume')
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Decompose trends & seasonality
fig2 = model.plot_components(forecast)
