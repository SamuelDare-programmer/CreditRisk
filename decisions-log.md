# 📓 Decisions Log

> **Purpose:** A permanent record of every significant choice made in this project —
> tools selected, approaches rejected, and the reasoning behind each call.
> **Before suggesting an alternative tool or approach, check this file first.**
>
> Format per entry:
> `## [DEC-XXX] — Decision Title`
> `Date · Category · Status: Accepted / Superseded / Under Review`

---

## [DEC-001] — Use LightGBM as the Primary Model Target
**Date:** Project inception · **Category:** Machine Learning · **Status:** Accepted

**Decision:** LightGBM is the expected winner and will receive the deepest tuning effort,
though all four models (LR, RF, XGBoost, LightGBM) will be trained and benchmarked.

**Rationale:**
- LightGBM consistently outperforms XGBoost on tabular financial datasets with
  high cardinality categorical features and class imbalance
- Faster training time than Random Forest at scale
- Native support for SHAP `TreeExplainer` — essential for the explainability requirement
- Well-documented on Kaggle for exactly the Home Credit dataset being used

**Rejected Alternatives:**
- Neural networks (too opaque for SHAP integration; overkill for tabular data)
- CatBoost (less community material for this specific dataset; similar performance to LGBM)

---

## [DEC-002] — Use Home Credit Default Risk as Primary Dataset (REPLACED BY DEC-011)
**Date:** Project inception · **Category:** Data · **Status:** Superseded

**Decision:** Home Credit Default Risk (Kaggle) is the primary training dataset,
supplemented by Lending Club data.

**Rationale:**
- Home Credit dataset is richer (122 features, 300k+ rows) and more realistic
  for the target Nigerian context (thin-file borrowers, alternative data signals)
- Well-studied on Kaggle with public notebooks providing good EDA baselines
- Lending Club adds diversity but has a US bias — used for supplementary features only

**Rejected Alternatives:**
- Synthetic dataset generation (would undermine real-world validity claims in the report)

---

## [DEC-003] — Use FastAPI over Flask or Django
**Date:** Project inception · **Category:** Backend Framework · **Status:** Accepted

**Decision:** FastAPI is the web framework for the REST API layer.

**Rationale:**
- Automatic Swagger/OpenAPI documentation generation (a project deliverable)
- Native async support — pairs cleanly with Celery for batch scoring
- Pydantic integration for schema validation is first-class, not bolted on
- Significantly faster than Flask for I/O-bound tasks
- More modern and increasingly standard in Python ML APIs

**Rejected Alternatives:**
- Flask (no built-in validation, no auto-docs, slower)
- Django REST Framework (too heavyweight; ORM conflicts with SQLAlchemy preference)

---

## [DEC-004] — Use PostgreSQL over MySQL or MongoDB
**Date:** Project inception · **Category:** Database · **Status:** Accepted

**Decision:** PostgreSQL is the application database (via AWS RDS in production,
local Docker container in development).

**Rationale:**
- JSON/JSONB column support — ideal for storing input/output payloads in the
  `ScoringRequestLog` table without schema migrations
- SQLAlchemy has first-class PostgreSQL support
- AWS RDS PostgreSQL is well-documented and has a free tier

**Rejected Alternatives:**
- MongoDB (schema-less design would make auditing and querying logs harder)
- MySQL (weaker JSON support; less feature-rich for analytical queries)

---

## [DEC-005] — Use SHAP TreeExplainer for Explainability
**Date:** Project inception · **Category:** Explainability · **Status:** Accepted

**Decision:** SHAP (SHapley Additive exPlanations) with `TreeExplainer` is the
explainability method. Every API response includes the top 5 ranked risk factors.

**Rationale:**
- Theoretically grounded (game theory / Shapley values)
- TreeExplainer is optimised for gradient boosting models (LightGBM/XGBoost) —
  fast enough for real-time API responses
- Output is feature-level, directional, and magnitude-ranked — easy to translate
  into plain-English explanations for lenders
- Industry standard for regulated financial ML models

**Rejected Alternatives:**
- LIME (less stable across runs; not as theoretically rigorous)
- Feature importance from the model directly (global, not per-prediction)

---

## [DEC-006] — Deploy on AWS (EC2 + RDS + S3 + CloudWatch)
**Date:** Project inception · **Category:** Infrastructure · **Status:** Accepted

**Decision:** AWS is the cloud provider for deployment.

**Rationale:**
- AWS Free Tier covers EC2 (t2.micro/t3.micro), RDS, and S3 within project scope
- CloudWatch provides native alerting without additional tooling
- Most widely used cloud platform — demonstrates industry-relevant skills
- Strong documentation and academic resources available

**Rejected Alternatives:**
- Google Cloud / GCP (less free tier flexibility for this stack)
- Railway / Render (simpler but limited control; less impressive academically)
- Heroku (no longer has a free tier)

---

## [DEC-007] — Use Docker + Docker Compose for Containerisation
**Date:** Project inception · **Category:** DevOps · **Status:** Accepted

**Decision:** The entire application stack (API, worker, DB, Redis) is containerised
with Docker and orchestrated locally via Docker Compose.

**Rationale:**
- Ensures environment parity between local development and AWS EC2
- Docker Compose makes it trivial to spin up PostgreSQL + Redis locally
- Containerisation is a required deliverable and demonstrates production thinking

**Rejected Alternatives:**
- Kubernetes (overkill for a single-developer academic project)
- Manual server setup without Docker (fragile, not reproducible)

---

## [DEC-008] — Use SQLAlchemy + Alembic over raw SQL or Django ORM
**Date:** Project inception · **Category:** Database ORM · **Status:** Accepted

**Decision:** SQLAlchemy (ORM) with Alembic (migrations) is the database layer.

**Rationale:**
- Alembic provides version-controlled schema migrations — essential for RDS
- SQLAlchemy is framework-agnostic (works with FastAPI cleanly)
- Alembic + SQLAlchemy is the de facto standard for FastAPI projects

---

## [DEC-009] — Use SMOTE for Class Imbalance
**Date:** Project inception · **Category:** Machine Learning · **Status:** Accepted

**Decision:** SMOTE (Synthetic Minority Oversampling Technique) from `imbalanced-learn`
is applied to the training set only (never the test set).

**Rationale:**
- The Home Credit dataset has ~8% default rate — severe imbalance will bias the model
  toward predicting "no default" on everything
- SMOTE applied only to training set prevents data leakage into evaluation
- Alternative: `class_weight='balanced'` in models — this will be tested as a baseline

**Rejected Alternatives:**
- Undersampling the majority class (loses too much data on a dataset this size)
- No handling at all (model would have poor recall on the default class)

---

## [DEC-010] — OOP / Class-Based Architecture Throughout
**Date:** Project inception · **Category:** Code Architecture · **Status:** Accepted

**Decision:** All Python code follows Object-Oriented Programming patterns.
Services, repositories, trainers, and explainers are implemented as classes.

**Rationale:**
- Developer's stated preference
- Produces more maintainable, testable, and extensible code
- Aligns well with FastAPI's dependency injection system
- Easier to mock in unit tests

**See also:** `preferences.md` for full coding standards.

---

## [DEC-005] — Project Folder Structure
**Date:** April 22, 2026 · **Category:** Project Structure · **Status:** Accepted

**Decision:** Adopt a modular FastAPI project structure with clear separation of concerns.

**Structure:**
```
app/
├── __init__.py
├── main.py                 # FastAPI app factory
├── api/
│   ├── __init__.py
│   └── v1/
│       ├── __init__.py
│       └── endpoints.py    # API routes
├── core/
│   ├── __init__.py
│   └── config.py           # Settings and configuration
├── models/
│   ├── __init__.py
│   ├── pydantic/           # Request/Response models
│   └── sqlalchemy/         # Database models
└── services/
    ├── __init__.py
    └── predictor_service.py # Business logic and ML
data/                       # Raw and processed datasets
saved_models/               # Trained model artifacts
tests/                      # Unit and integration tests
```

**Rationale:**
- Follows FastAPI best practices for scalability
- Clear separation between API, business logic, and data models
- Easy to extend with new API versions or services
- Aligns with OOP principles in `preferences.md`

**Rejected Alternatives:**
- Flat structure (harder to maintain as project grows)
- Django-style monolithic app (overkill for API-only service)

---

## [DEC-006] — Feature-Based API Structure
**Date:** April 22, 2026 · **Category:** Project Structure · **Status:** Accepted

**Decision:** Restructure API to feature-based modules with shared core components.

**New Structure:**
```
app/
├── core/                        # Shared API components
│   ├── __init__.py
│   ├── auth.py                  # Authentication utilities
│   ├── config.py                # Application settings
│   ├── errors.py                # Custom exceptions
│   └── middleware.py            # Custom middleware
├── credit_scoring/              # Credit scoring feature
│   ├── __init__.py
│   ├── models.py                # Database models
│   ├── routes.py                # FastAPI routes
│   ├── schemas.py               # Pydantic models
│   └── services.py              # Business logic
├── models/                      # Shared database models (if any)
└── main.py                      # App entry point
```

**Rationale:**
- Feature-based organization improves maintainability and scalability
- Each feature is self-contained with its own router, schemas, and services
- Shared core components (auth, config, errors, middleware) avoid duplication
- Easier to add new features without affecting existing code
- Follows domain-driven design principles

**Rejected Alternatives:**
- Monolithic router file (hard to maintain as features grow)
- Feature-based without shared core (code duplication)

---

## 📋 Open Decisions (To Be Resolved)

| ID      | Question                                              | Target Date |
|---------|-------------------------------------------------------|-------------|
| TBD-001 | Hyperparameter tuning: GridSearchCV vs Optuna?        | Phase 2     |
| TBD-002 | JWT token storage strategy (DB vs stateless)?         | Phase 3     |
| TBD-003 | Celery broker: Redis (current plan) vs RabbitMQ?      | Phase 3     |
| TBD-004 | Model versioning: manual S3 keys vs MLflow?           | Phase 2–4   |

---

## [DEC-011] — Move from Kaggle Datasets to Nigerian-Context Synthetic Data
**Date:** June 6, 2024 · **Category:** Data · **Status:** Superseded

**Decision:** Pivot from using Kaggle datasets (Home Credit, Lending Club) to a custom-generated synthetic dataset specifically designed for the Nigerian fintech ecosystem, inspired by schemas from Zindi African Credit Scoring challenges.

**Rationale:**
- Kaggle datasets are heavily skewed towards US/European demographics and traditional banking systems.
- A Nigerian-context model requires alternative data signals (e.g., BVN, USSD usage, Telco provider, active betting accounts, airtime spend).
- Generating a synthetic dataset allows us to explicitly model "willingness to pay" vs. "ability to pay", and differentiate between "new" and "repeat" customers, as seen in regional hackathons.
- Demonstrates deep domain awareness and custom engineering for the final year project defense.

**Rejected Alternatives:**
- Continuing with Home Credit dataset (lacks cultural relevance and alternative data signals critical in Nigeria).


---

## [DEC-012] — Use Real Zindi Datasets as Primary Data Source
**Date:** June 6, 2024 · **Category:** Data · **Status:** Accepted

**Decision:** The primary training data will be the real datasets from the Zindi "African Credit Scoring Challenge" and the "Loan Default Prediction Challenge" (SuperLender). The synthetic dataset generated in DEC-011 will be retained as a supplementary tool.

**Rationale:**
- Real data is mathematically and academically superior to synthetic data for a final year project, as it contains authentic real-world noise, missing values, and behavioral patterns.
- The Zindi datasets perfectly capture the African/Nigerian ecosystem (e.g., differentiating between willingness vs. ability to pay, and new vs. repeat business risk).
- The synthetic generator remains valuable for edge-case testing, data augmentation, and demonstrating engineering capability.

**Rejected Alternatives:**
- Relying purely on synthetic data (lacks the statistical rigor of real human behavioral data).

---

## [DEC-013] — Exclude Macroeconomic Indicators from Pipeline
**Date:** July 2, 2026 · **Category:** Data · **Status:** Accepted

**Decision:** Macroeconomic indicators (from `economic_indicators.csv`) will be omitted from the models.

**Rationale:**
- The macro dataset is highly aggregate (country + year level) and contains many missing values for key metrics.
- Merging country-level, annual data (e.g. GDP, inflation) into individual micro-loan records is likely to cause overfitting, as the models can easily memorize country-year pairs.
- Alternative behavioural features (telco provider, USSD bank usage, previous late repayments) provide far stronger risk signals for credit underwriting digital borrowers.

---

## [DEC-014] — Benchmark Imbalance Strategies & Resolve CV Data Leakage
**Date:** July 2, 2026 · **Category:** Machine Learning · **Status:** Accepted

**Decision:** Benchmark two main class-imbalance strategies (SMOTE vs Class Weight) and resolve cross-validation leakage by applying SMOTE strictly inside training splits (via `imblearn` pipeline) rather than pre-resampling the entire training set.

**Rationale:**
- A common mistake in credit scoring pipelines is running cross-validation on pre-resampled datasets, leading to inflated validation metrics (~0.92 AUC-ROC) due to synthetic samples leaking into validation folds. Wrapping SMOTE inside an `imblearn` pipeline exposes the true generalization capability (~0.68 AUC-ROC).
- Benchmarking both SMOTE and cost-sensitive class weights (`class_weight='balanced'`, `scale_pos_weight`, `is_unbalance=True`) provides a rigorous comparison for academic project defense.
- Logistic Regression was selected as the best baseline performer (0.7191 AUC-ROC on test set).

