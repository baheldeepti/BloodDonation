# Blood Donation Decision Support System

## Problem Statement

Blood banks and healthcare organizations seek to efficiently allocate outreach resources to maximize repeat donations while minimizing marketing effort. Generic campaigns waste budget on unlikely donors, whereas targeted outreach can yield higher returns. We formulate an optimization problem to decide which donors to target under a budget constraint.

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

## Optimization Problem Formulation

We define an integer programming problem:

**Decision Variables**:

* $x_i \in \{0,1\}$: binary indicator, 1 if donor $i$ is selected for campaign outreach

**Parameters**:

* $p_i$: predicted probability that donor $i$ will donate again (from ML models)
* $v$: monetary value saved per successful donation (USD)
* $c_i$: cost of contacting donor $i$ (e.g., SMS, email, phone)
* $B$: total marketing budget (USD)

**Objective**:

$$
\max_{x} \sum_i x_i \;(p_i \times v) \quad - \quad \sum_i x_i \;c_i
$$

Maximize expected net savings = expected donation value minus outreach costs.

**Constraint**:

$$
\sum_i x_i \;c_i \le B
$$

Total contact cost must not exceed budget $B$.

**Additional Constraints** (optional):

* Limit number of SMS vs email campaigns: $\sum_{i\in SMS} x_i \le S_{max},$
* Minimum expected donations target: $\sum_i x_i \;p_i \ge D_{min}$

By solving this optimization, blood banks can allocate their outreach budget to the donors with the highest expected return on investment.

---

For detailed setup and dependencies, see `requirements.txt`.
