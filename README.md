# Retail Customer Churn Analysis

**Author:** Harishma

## Project Overview

This project focuses on building a machine learning model to predict customer churn for a retail business, identifying key factors influencing churn, and translating these insights into actionable retention strategies. Using the **Online Retail II dataset from Kaggle**, this project demonstrates a full data science workflow — from data acquisition and cleaning to feature engineering, advanced machine learning modeling (Gradient Boosting), and interactive visualization using **Streamlit**.  

The goal is to empower business stakeholders with a **data-driven tool** to identify high-risk customers and understand the reasons behind churn, enabling targeted interventions to improve customer retention and lifetime value.  

## Problem Statement

Customer churn significantly impacts revenue and growth for retail businesses. By predicting churn early, businesses can implement **targeted retention strategies** such as personalized offers, loyalty programs, or improved customer service.  

This project addresses:  
- Developing a predictive model for customer churn  
- Identifying the most influential behavioral and transactional factors contributing to churn  
- Providing an interactive dashboard for stakeholders to explore churn insights and identify at-risk customers  

## Dataset

- **Source**: [Online Retail II - Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)  
- **Description**: Transactional data from a UK-based online retail store, Dec 2009 – Dec 2011  
- **Key Columns**: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `CustomerID`, `Country`  

> **Churn Definition:**  
> Customers with no transactions within a **3-month window** after their last purchase are labeled as churned.  

## Project Phases & Methodology

### 1. Project Planning & Setup
- Defined scope, objectives, tools, and repository structure  

### 2. Data Acquisition & Understanding
- Loaded and cleaned data (handling missing `CustomerID`s, negative quantities/prices, cancelled orders)  
- Explored the data with **EDA** to understand trends and distributions  

### 3. Data Preprocessing & Feature Engineering
- Created features at customer level:  
  - **RFM (Recency, Frequency, Monetary)**  
  - **Tenure**: Days since first purchase  
  - One-hot encoding for `Country`  
- Created binary target variable `is_churned`  

### 4. Modeling & Evaluation
- Split dataset into train/test sets  
- Trained **Logistic Regression, Random Forest, and Gradient Boosting** models  
- Gradient Boosting selected as final model after hyperparameter tuning (GridSearchCV)  
- Evaluated performance using: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix  
- Determined feature importance to identify churn drivers  

### 5. Visualization & Communication
- Built an interactive **Streamlit dashboard** showing:  
  - Overall churn rate  
  - Churn by country  
  - Top churn drivers  
  - Risk probability for individual customers  

## Technologies Used

- **Python**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `streamlit`  
- **Model Persistence**: `joblib`  
- **Version Control**: Git & GitHub  
- **Environment**: Jupyter Notebook  

## How to Run the Project

### Clone the Repository
```bash
git clone https://github.com/YourUsername/Retail-Customer-Churn-Analysis.git
cd Retail-Customer-Churn-Analysis
