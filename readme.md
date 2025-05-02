# Blood Donation Decision Support System

## Problem Statement

Blood banks and healthcare organizations need to identify and engage repeat donors efficiently to ensure a stable blood supply. Manual processes and generic outreach often miss high‑value donors or waste resources on unlikely donors. This application provides an AI‑powered decision support system to:

* Predict which donors are most likely to give again
* Forecast monthly donation volumes
* Recommend personalized outreach strategies
* Quantify expected cost savings per donor segment

## Data Source

All data displayed and modeled in this app is **synthetically generated** to simulate real blood donation records. Features include:

* **Recency**: Months since last donation
* **Frequency**: Total number of past donations
* **Monetary**: Total blood volume donated (mL)
* **Time**: Months since first donation
* **Age**: Donor age in years
* **CampaignResponse**: 1 if donor responded to previous campaign, else 0
* **Target**: 1 if donor donated again, else 0

## Data Dictionary

| Column           | Description                               |
| ---------------- | ----------------------------------------- |
| Recency          | Months since last donation                |
| Frequency        | Total number of donations                 |
| Monetary         | Total blood volume donated (in mL)        |
| Time             | Months since first donation               |
| Age              | Donor age (years)                         |
| CampaignResponse | 1 if donor responded to previous campaign |
| Target           | 1 if donor donated again, else 0          |

## Methodology

1. **Data Simulation & EDA**

   * Generate synthetic donor records
   * Visualize distributions, KPIs, correlations, and categorical breakdowns
   * Obtain AI‑generated insights via GPT-3.5

2. **Feature Engineering & Selection**

   * Derive `Monetary_per_Frequency` and `Donation_Intensity`
   * Select top 7 predictors based on correlation with `Target`

3. **Modeling & Tuning**

   * Train and tune six classifiers: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, CatBoost, and SVM
   * Evaluate via ROC‑AUC, accuracy, F1, precision, recall
   * Compare models in a styled table and ROC plots
   * Leverage GPT-3.5 to recommend the top model

4. **Decision Support**

   * Allow batch or manual donor input
   * Enable multi‑model selection for ensemble comparison
   * Compute donation probability, recommended outreach (SMS, Email, Deprioritize), and expected monetary value (USD)
   * Export recommendations as CSV

## Application Pages

* **Introduction**: App overview, data dictionary and navigation
* **Exploratory Analysis**: Interactive charts, KPI metrics, correlation heatmap, and AI insights
* **Predictive Modeling**: Feature pipeline, model training & hyperparameter tuning, performance comparison, and GPT recommendations
* **Decision Support**: Donor data input, multi‑model predictions, outreach decisions, cost‑savings analysis, and CSV export

---

For detailed setup and dependencies, see `requirements.txt`.
