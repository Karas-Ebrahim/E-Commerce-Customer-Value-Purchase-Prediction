# E-Commerce Customer Intelligence & Purchase Prediction - DEPI Grad Project

An end-to-end Data Science project that turns real-world e-commerce transaction data into actionable business insights using **SQL, Python, Machine Learning, MLflow, Interactive Visualization, and FastAPI**.

The project is based on the **Olist Brazilian E-Commerce Public Dataset**, which is provided as multiple related CSV tables covering customers, orders, products, payments, reviews, sellers, and related information.

---

## 🎯 Business Problem

The company has a large amount of historical e-commerce data, but it lacks an integrated understanding of:

- Customer purchasing behavior
- Order and product performance
- Payment behavior
- Customer satisfaction
- Repeat-purchase behavior
- Future customer outcomes

The goal is to transform this raw data into a system that provides both:

1. **Business intelligence** through SQL, analysis, and dashboards.
2. **Predictive intelligence** through supervised machine learning.

> **Important:** We will not claim to predict true profit unless reliable cost/profit information is available in the data. Any ML target must be supported by the actual dataset.

---

# 🚀 Features

## 1. Multi-Source Data Ingestion

The project starts from the Olist dataset, which is split into nine related CSV files:

- `customers`
- `orders`
- `order_items`
- `order_payments`
- `order_reviews`
- `products`
- `sellers`
- `geolocation`
- `product_category_name_translation`

The first stage is to understand the purpose, columns, data types, grain, keys, and relationships of every table.

---

## 2. Relational Database & SQL Analytics

The raw CSV files will be organized into a relational SQL database.

The database layer will define:

- Primary Keys
- Foreign Keys
- Table relationships
- One-to-many relationships
- Data-quality checks
- Useful indexes where appropriate

SQL will then be used for business analysis such as:

- Customer purchase frequency
- Revenue analysis
- Average Order Value (AOV)
- Product performance
- Category performance
- Seller performance
- Payment-method analysis
- Repeat-purchase analysis
- Geographic analysis

Example relationship:

```text
Customers
    |
    | customer_id
    v
Orders
    |
    | order_id
    +------------------> Order Payments
    |
    +------------------> Order Reviews
    |
    v
Order Items
    |
    +---- product_id --> Products
    |
    +---- seller_id ---> Sellers
```

SQL is responsible for the relational and analytical layer. It does not replace the Python Data Science layer.

---

## 3. Data Cleaning & Data Quality

The project will work with real-world data rather than a clean educational dataset.

The cleaning stage will investigate and handle, where appropriate:

- Missing values
- Incorrect data types
- Inconsistent values
- Invalid records
- Duplicate records
- Outliers / extreme values
- Date and timestamp issues
- Categorical inconsistencies

Missing values will not automatically be handled with one generic strategy. Their business meaning will be investigated first.

---

## 4. Exploratory Data Analysis (EDA)

Python will be used for deeper Data Science analysis after the analytical dataset is prepared.

EDA will include:

- Descriptive statistics
- Distributions
- Frequency analysis
- Group-by analysis
- Correlation analysis
- Customer behavior analysis
- Product/category analysis
- Temporal trends
- Geographic patterns
- Data-quality visualization

The goal is to discover patterns that can support business decisions, not just create charts.

---

## 5. Feature Engineering

Raw transaction records are not necessarily suitable as direct ML inputs.

The project will create meaningful features from historical customer and transaction behavior, such as:

- Total number of orders
- Total spending
- Average Order Value
- Purchase frequency
- Number of purchased products
- Number of product categories
- Average review score
- Time since last purchase
- Historical order behavior
- Payment behavior
- Geographic/customer features

The final modeling dataset must have a clearly defined **grain**.

Example:

```text
orders:
1 row = 1 order

order_items:
1 row = 1 item inside an order

ML customer dataset:
1 row = 1 customer
```

This prevents incorrect aggregation and data leakage.

---

## 6. Supervised Machine Learning

The project will use supervised learning to predict a business-relevant target supported by the data.

### Classification

A possible direction is predicting future purchase behavior, for example:

```text
0 → No Repeat Purchase
1 → Occasional Repeat
2 → Frequent Repeat
```

This is especially useful for a **multi-class + imbalanced** business problem.

### Regression

A regression problem can also be used if a meaningful numerical future outcome is supported by the dataset, such as an appropriate customer-value or future-spending measure.

> The final target will be selected after data understanding and validation instead of being assumed beforehand.

---

## 7. Imbalanced Multi-Class Learning

When the selected target contains multiple classes, we will analyze class distribution and handle imbalance where necessary.

The project may use:

- Class weighting
- Appropriate resampling techniques
- Per-class evaluation
- Macro F1
- Precision
- Recall
- Confusion Matrix

Accuracy will not be the only evaluation metric.

---

## 8. Model Comparison

Multiple suitable ML models will be trained and compared.

The comparison will consider:

- Overall performance
- Per-class performance
- Generalization
- Model complexity
- Business usefulness

The goal is to select the most appropriate model for the problem, not simply the model with the highest Accuracy.

---

## 9. Bayesian Hyperparameter Search

Selected models will use **Bayesian hyperparameter optimization** to search for better configurations.

```text
Model
   ↓
Search Space
   ↓
Bayesian Optimization
   ↓
Best Hyperparameters
   ↓
Final Evaluation
```

---

## 10. MLflow Experiment Tracking

MLflow will be used to track machine-learning experiments.

Tracked information may include:

- Model type
- Hyperparameters
- Metrics
- Artifacts
- Experiment runs
- Selected/best model

Example:

```text
Run 01 → Logistic Regression
Run 02 → Decision Tree
Run 03 → Random Forest
Run 04 → Tuned Random Forest
```

This makes experimentation and model selection more reproducible.

---

## 11. Interactive Business Dashboard

The dashboard will turn the analysis and ML results into a business-facing view.

### Executive Overview

- Total customers
- Total orders
- Revenue
- Average Order Value
- Repeat-purchase rate

### Customer Analytics

- Customer behavior
- Purchase frequency
- Customer value
- Geographic distribution
- Customer characteristics

### Product & Sales Analytics

- Product performance
- Category performance
- Seller performance
- Payment trends
- Sales trends

### Machine Learning

- Prediction distribution
- Model performance
- Confusion Matrix
- Important features
- Prediction results

The dashboard is intended to support decisions rather than simply display charts.

---

## 12. FastAPI Model Serving

The selected model will be exposed through **FastAPI**.

Example endpoint:

```http
POST /predict
```

Example request:

```json
{
  "total_orders": 4,
  "total_spending": 850,
  "avg_order_value": 212.5,
  "avg_review_score": 4.5
}
```

Example response:

```json
{
  "prediction": "Frequent Repeat",
  "confidence": 0.81
}
```

API flow:

```text
Client
  ↓
FastAPI
  ↓
Input Validation
  ↓
Preprocessing Pipeline
  ↓
Trained Model
  ↓
Prediction
  ↓
JSON Response
```

I will take responsibility for the **Backend/API side**, including the FastAPI endpoints and integration with the trained model.

---

## 13. Clustering Extension

Customer segmentation may be added later as an extension after the clustering material is covered.

Possible flow:

```text
Customer Features
      ↓
K-Means
      ↓
Customer Segments
      ↓
Segment Profiles
      ↓
Dashboard
```

Clustering is **not a core dependency** of the project, so the main project remains complete without it.

---

# 🧩 Overall Project Flow

```text
                  RAW OLIST DATA
                   9 CSV TABLES
                        |
                        v
                DATA UNDERSTANDING
                        |
                        v
             RELATIONAL SQL DATABASE
                        |
             +----------+----------+
             |                     |
             v                     v
       SQL RELATIONSHIPS      SQL ANALYTICS
       PK / FK / JOINS       BUSINESS QUESTIONS
             |                     |
             +----------+----------+
                        |
                        v
               ML-READY DATASET
                        |
                        v
                 PYTHON / PANDAS
                        |
            +-----------+-----------+
            |           |           |
            v           v           v
         CLEANING      EDA     FEATURE ENGINEERING
            |           |           |
            +-----------+-----------+
                        |
                        v
                 TARGET DEFINITION
                        |
                        v
            TRAIN / VALIDATION / TEST
                        |
                        v
                  PREPROCESSING
                        |
                        v
               MACHINE LEARNING
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
       MODELS     BAYESIAN SEARCH   EVALUATION
          |             |             |
          +-------------+-------------+
                        |
                        v
                      MLFLOW
                        |
                        v
                   BEST MODEL
                    /      \
                   /        \
                  v          v
             DASHBOARD     FASTAPI
                               |
                               v
                        MODEL PREDICTION
```

---

# 🛠️ Technology Stack

| Area | Technology |
|---|---|
| Data Storage | SQL Database |
| SQL Analytics | SQL |
| Data Processing | Python, Pandas |
| Visualization | Python Visualization Libraries + Interactive Dashboard |
| Machine Learning | Scikit-learn / Appropriate ML Libraries |
| Hyperparameter Optimization | Bayesian Search |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Version Control | Git & GitHub |

### Project Constraints

- **Azure will not be used.**
- **Docker will not be used.**

The deployment plan will focus on FastAPI and other technologies that fit the project without depending on Azure or Docker.

---

# 📁 Suggested Project Structure

```text
ecommerce-customer-intelligence/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
│
├── sql/
│   ├── 01_schema.sql
│   ├── 02_data_validation.sql
│   ├── 03_business_queries.sql
│   └── 04_ml_dataset.sql
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   └── utils/
│
├── mlflow/
│
├── api/
│   ├── main.py
│   ├── routes/
│   ├── schemas/
│   └── services/
│
├── dashboard/
│
├── tests/
│
├── reports/
│
├── README.md
└── requirements.txt
```

---

# 👥 Team Workflow

The project is designed for a team of **5 members**.

A possible division:

| Member | Main Responsibility |
|---|---|
| Member 1 | Data ingestion, SQL schema, relational design |
| Member 2 | Data cleaning, EDA, statistical analysis |
| Member 3 | Feature engineering, ML preprocessing |
| Member 4 | ML models, Bayesian Search, MLflow |
| Member 5 | FastAPI, dashboard integration, deployment |

Responsibilities are not isolated. Everyone should understand the complete pipeline and collaborate during integration, evaluation, documentation, and presentation.

---

# ✅ Expected Outcome

The final project aims to demonstrate:

- Real-world data handling
- Relational database design
- SQL analysis
- Data cleaning
- Exploratory Data Analysis
- Feature engineering
- Supervised Machine Learning
- Multi-class classification and imbalance handling where applicable
- Bayesian hyperparameter optimization
- MLflow experiment tracking
- Interactive business visualization
- FastAPI model serving
- Reproducible project structure
- Business-oriented interpretation of results

The goal is to move from:

```text
Raw Data
```

to:

```text
Business Insights + Predictive Model + Usable API
```

rather than stopping at a notebook-only model.

---

## 📌 Project Status

This README describes the planned project architecture and capabilities.

The final ML target, exact features, selected models, and deployment details will be finalized after the initial **data-understanding and validation phase**, based on what the Olist data actually supports.
