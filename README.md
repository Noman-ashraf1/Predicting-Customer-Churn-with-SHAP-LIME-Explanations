# Predicting-Customer-Churn-with-SHAP-LIME-Explanations# Customer Churn Prediction with Explainable AI (XAI)

Predict which customers are likely to churn from a service while providing **interpretable explanations** for the predictions using SHAP. This system helps business stakeholders make data-driven decisions to reduce churn.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Sample Predictions](#sample-predictions)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Explainability](#explainability)  
7. [Technologies Used](#technologies-used)  
8. [Folder Structure](#folder-structure)  
9. [Author](#author)  

---

## Project Overview

Customer churn is a major challenge for subscription-based services and telecom companies. This project predicts **whether a customer will leave a service** and explains **why** using Explainable AI (XAI) tools.  

The pipeline handles:

- Binary, categorical, and numeric features  
- Preprocessing: Label Encoding, One-Hot Encoding, and Scaling  
- Batch and single-row predictions  
- Model explainability with SHAP/LIME  

This allows stakeholders to **understand the drivers of churn** and take preventive actions.

---

## Features

- **Binary & Multi-class Encoding**  
  - Label Encoding for binary features (Gender, Partner, Dependents, etc.)  
  - One-Hot Encoding for multi-class features (Internet Service, Contract, Payment Method)  

- **Scaling**  
  - Standardizes numeric features like `Monthly Charges`, `Total Charges`, `Tenure Months`  

- **Machine Learning Model**  
  - Predicts **Churn (0/1)**  
  - Outputs **Probability of churn**  

- **Explainable AI (XAI)**  
  - Integrates **SHAP or LIME** for each prediction  
  - Shows key factors contributing to churn in a format understandable by business stakeholders  

- **Batch & Single Prediction**  
  - Handles multiple customers in one batch  
  - Can process single customers interactively  

---

## Sample Predictions

| Gender | Senior Citizen | Partner | Tenure Months | Internet Service | Contract | Payment Method | Churn | Probability |
|--------|----------------|--------|----------------|-----------------|---------|----------------|-------|------------|
| Male   | 1              | No     | 24             | DSL             | One year | Bank transfer   | 0     | 0.036283   |
| Female | 0              | Yes    | 12             | Fiber optic     | Month-to-month | Electronic check | 0 | 0.476118 |

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Noman-ashraf1/Customer-Churn-Prediction-XAI.git


 [Explainability](#explainability)  


import shap

# Prepare processed features
df_processed = preprocess_sample(df_sample)  # apply your label + one-hot + scaling

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df_processed)

# Visualize summary plot
shap.summary_plot(shap_values, df_processed)


Technologies Used

Python 3.x

pandas, NumPy

scikit-learn

Joblib (for saving/loading models)

SHAP / LIME (for explainability)









cd Customer-Churn-Prediction-XAI
pip install -r requirements.txt
