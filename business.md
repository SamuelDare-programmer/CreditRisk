# 📋 Business Context

> **Purpose:** Defines what this project is, why it exists, who it serves,
> and all relevant entities. Read this before writing any feature or endpoint.

---

## 1. Project Overview

The **Credit Risk Scoring System** is a machine-learning-powered REST API that
predicts the probability that a loan applicant will default on their loan.

It is built as a **final year Computer Science project** at the University of Benin,
designed to reflect a **production-grade system** that real Nigerian fintech lenders
(e.g. Carbon, FairMoney, PalmCredit) could integrate directly into their underwriting
workflows.

### Core Value Proposition

| Problem                                              | This System's Solution                              |
|------------------------------------------------------|-----------------------------------------------------|
| Manual credit assessment is slow and subjective      | ML model returns a risk score in milliseconds       |
| High default rates due to limited data signals       | Feature-rich model trained on contextual synthetic loan datasets    |
| Borrowers get rejections with no explanation         | SHAP values explain every decision in plain English |
| Lenders can't audit black-box model decisions        | Full explainability layer baked into every response |

---

## 2. Domain Context — Nigerian Digital Lending

Nigeria's digital lending market is one of the fastest-growing in Africa. Key facts
relevant to this project:

- The **Central Bank of Nigeria (CBN)** regulates digital lenders and mandates
  responsible lending practices
- Credit bureau penetration remains low — many borrowers have **no formal credit history**
- Features like **BVN (Bank Verification Number)** linkage, mobile money activity,
  and utility payments are increasingly used as alternative credit signals
- **Zindi Africa** challenges highlight the need for a dual assessment approach:
  1) **Willingness to pay** and 2) **Ability to pay**.
- Credit models must differentiate between **New Business Risk** (first loan) and
  **Behavioral Risk** (repeat customers).
- Target lenders: **Carbon**, **FairMoney**, **PalmCredit**, **Renmoney**, **Migo**

> ⚠️ **Design implication:** The model must perform well on thin-file borrowers
> (those with limited credit history). Feature engineering should prioritise
> behavioural and income-based signals over traditional bureau scores.

---

## 3. System Entities

### 3.1 Lender (API Consumer)
A financial institution or fintech company that integrates with the API.

| Attribute       | Description                                         |
|-----------------|-----------------------------------------------------|
| `lender_id`     | Unique identifier                                   |
| `name`          | Company name (e.g. FairMoney Nigeria Ltd)           |
| `api_key`       | JWT-authenticated key for API access                |
| `rate_limit`    | Max requests per minute (default: 60)               |
| `created_at`    | Account creation timestamp                          |

---

### 3.2 Borrower Profile (API Input)
The loan applicant's data submitted per scoring request.

| Attribute              | Type    | Description                                |
|------------------------|---------|--------------------------------------------|
| `age`                  | int     | Applicant age (18–80)                      |
| `annual_income`        | float   | Gross annual income in NGN                 |
| `loan_amount`          | float   | Requested loan amount in NGN               |
| `loan_purpose`         | str     | Purpose (personal, business, education...) |
| `employment_status`    | str     | Employed, Self-employed, Unemployed        |
| `has_bvn`              | bool    | BVN verification status                    |
| `telco_provider`       | str     | MTN, Airtel, Glo, 9mobile                  |
| `monthly_airtime_spend`| float   | Monthly airtime usage in NGN               |
| `active_betting_account`| bool   | Active betting account indicator           |
| `ussd_bank_usage`      | int     | Number of USSD banking sessions/month      |
| `is_repeat_customer`   | bool    | Indicates if repeat borrower               |
| `loan_term_months`     | int     | Requested repayment period                 |

---

### 3.3 Risk Score
---

### 3.3 Risk Score (API Output)
The system's response for each scoring request.

| Attribute            | Type    | Description                                         |
|----------------------|---------|-----------------------------------------------------|
| `request_id`         | str     | UUID for this scoring request                       |
| `risk_score`         | float   | Default probability (0.0 – 1.0)                     |
| `risk_label`         | str     | `LOW` / `MEDIUM` / `HIGH`                           |
| `top_risk_factors`   | list    | Ranked SHAP explanations (feature, impact, direction)|
| `recommendation`     | str     | Plain-English summary for the lender                |
| `model_version`      | str     | Which model version produced this score             |
| `scored_at`          | datetime| Timestamp of the scoring event                      |

---

### 3.4 ML Model
The trained machine learning model powering the API.

| Attribute          | Description                                          |
|--------------------|------------------------------------------------------|
| `algorithm`        | Best performer from: LR, RF, XGBoost, LightGBM      |
| `training_data`    | Real Zindi Competition Datasets (Primary) + Synthetic Dataset (Supplementary) |
| `target_variable`  | `TARGET` — 1 = defaulted, 0 = repaid                |
| `metric`           | AUC-ROC ≥ 0.85 on held-out test set                 |
| `imbalance_strategy`| SMOTE oversampling on minority (default) class      |
| `artefact_location`| AWS S3 bucket (model.pkl + preprocessor.pkl)        |
| `explainability`   | SHAP TreeExplainer on final model                    |

---

### 3.5 Scoring Request Log (Database Record)
Every API call is persisted to PostgreSQL for auditing.

| Attribute       | Description                              |
|-----------------|------------------------------------------|
| `id`            | Primary key (UUID)                       |
| `lender_id`     | Foreign key → Lender                    |
| `input_payload` | JSON of borrower profile sent           |
| `output_payload`| JSON of risk score returned             |
| `latency_ms`    | API response time in milliseconds        |
| `created_at`    | Timestamp                                |

---

## 4. System Boundaries

### In Scope ✅
- REST API for real-time single-applicant scoring
- Async batch scoring via Celery for bulk requests
- JWT authentication for lender access
- SHAP-based explainability on every response
- AWS cloud deployment with monitoring

### Out of Scope ❌
- A borrower-facing frontend or mobile app
- Direct integration with Nigerian credit bureaus (CRC, FirstCentral)
- Real-time BVN verification
- Loan origination or disbursement workflows
- Model retraining pipelines (static model for this version)

---

## 5. Success Criteria

| Metric                   | Target                             |
|--------------------------|------------------------------------|
| Model AUC-ROC            | ≥ 0.85 on test set                 |
| API Response Time        | < 500ms for single scoring request |
| System Uptime            | 99.9% on AWS                       |
| Explainability           | Top 5 SHAP factors on every response|
| Documentation            | Full Swagger UI + Postman collection|
| Academic Deliverables    | Written report + oral presentation  |
