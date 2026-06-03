# 📅 Daily Digest

> **Purpose:** A chronological, running log of all progress made on the project.
> Update this at the end of every working session — even short ones.
> This is the source of truth for "what has actually been done."
>
> **Entry format:**
> ```
> ## [YYYY-MM-DD] — Short Session Title
> **Phase:** Current phase name
> **Time Spent:** Approximate hours
> **Summary:** What was accomplished
> **Changes Made:** Specific files/code touched
> **Decisions Made:** Any new calls (add to decisions-log.md too)
> **Blockers:** Anything that slowed progress
> **Next Session:** Exactly what to do first next time
> ```

---

## [2026-04-22] — Project Kickoff & Knowledge Base Setup

**Phase:** Phase 0 — Setup
**Time Spent:** ~1 hour
**Summary:**
Defined the full project structure and created the knowledge base documentation
vault. No code written yet. This session was focused on planning, architecture,
and documentation scaffolding to ensure the project starts on solid foundations.

**Changes Made:**
- Created `credit-risk-docs/` knowledge base with 5 files:
  - `README.md` — AI navigation root file
  - `business.md` — Project context, entities, scope
  - `active-projects.md` — Phase tracker with all tasks
  - `decisions-log.md` — 10 foundational decisions logged (DEC-001 to DEC-010)
  - `preferences.md` — OOP coding standards, folder structure, tooling
  - `daily-digest.md` — This file (initialised)

**Decisions Made:**
- LightGBM as primary model target (DEC-001)
- FastAPI as web framework (DEC-003)
- PostgreSQL as database (DEC-004)
- SHAP TreeExplainer for explainability (DEC-005)
- AWS as cloud provider (DEC-006)
- OOP architecture throughout (DEC-010)
- *(See `decisions-log.md` for full list)*

**Blockers:** None

**Next Session:**
> 🎯 **Phase 0 — Environment Setup**
> 1. Create GitHub repository `credit-risk-api`
> 2. Scaffold the folder structure from `preferences.md`
> 3. Set up Python virtual environment
> 4. Install base dependencies and create `requirements.txt`
> 5. Download Home Credit dataset from Kaggle

---

## [2026-04-22] — Environment Setup & Project Scaffolding

**Phase:** Phase 0 — Environment Setup
**Time Spent:** ~2 hours
**Summary:**
Completed the initial project scaffolding by setting up the folder structure,
virtual environment, dependencies, and configuration files. The project is now
ready for development with a solid foundation following the defined standards.

**Changes Made:**
- Restructured API to feature-based architecture:
  - Created `app/api/core/` for shared components (auth, config, errors, middleware)
  - Created `app/api/features/credit_scoring/` with router, schemas, and services
  - Moved and updated PredictorService to feature-specific services
  - Updated main.py imports and router includes
  - Removed old v1/ and services/ directories
- Set up Python 3.11 virtual environment and installed base dependencies
- Created configuration files: requirements.txt, .env, .gitignore, pyproject.toml
- Added Docker setup: Dockerfile, docker-compose.yml with PostgreSQL + Redis
- Implemented initial Pydantic schemas and custom exceptions

**Decisions Made:**
- Adopted modular FastAPI structure (DEC-005)
- Implemented feature-based API organization (DEC-006 added to decisions-log.md)

**Blockers:** Virtual environment setup required recreation due to incomplete pip installation

**Next Session:** Create GitHub repository and download Kaggle datasets

---

## [2026-04-22] — API Structure Restructure to Feature-Based Architecture

**Phase:** Phase 0 — Environment Setup
**Time Spent:** ~1.5 hours
**Summary:**
Restructured the API architecture to a feature-based organization as requested, with shared core components. This improves maintainability and scalability by grouping related functionality together while avoiding code duplication.

**Changes Made:**
- Restructured `app/api/` to feature-based modules:
  - Created `app/api/core/` for shared components (auth, config, errors, middleware)
  - Created `app/api/features/credit_scoring/` with router, schemas, and services
  - Moved PredictorService to feature-specific services module
  - Updated main.py imports and router configuration
  - Removed old monolithic v1/ and services/ directories
- Fixed Pydantic import issue (BaseSettings moved to pydantic-settings)
- Validated that the restructured app imports and runs correctly

**Decisions Made:**
- Implemented feature-based API organization (DEC-006 added to decisions-log.md)

**Blockers:** Pydantic version compatibility issue with BaseSettings import (resolved)

**Next Session:** Create GitHub repository and download Kaggle datasets

---
TEMPLATE — Copy and paste for each new session:

## [YYYY-MM-DD] — Session Title

**Phase:** Phase X — Name
**Time Spent:**
**Summary:**

**Changes Made:**
-

**Decisions Made:**
-

**Blockers:**
-

**Next Session:**
> 🎯
> 1.
> 2.
> 3.

-->

## [2026-04-22] � API Structure Restructure to Feature-Based Architecture

**Phase:** Phase 0 � Environment Setup
**Time Spent:** ~1.5 hours
**Summary:**
Restructured the API architecture to a feature-based organization as requested, with shared core components. This improves maintainability and scalability by grouping related functionality together while avoiding code duplication.

**Changes Made:**
- Restructured pp/api/ to feature-based modules:
  - Created pp/api/core/ for shared components (auth, config, errors, middleware)
  - Created pp/api/features/credit_scoring/ with router, schemas, and services
  - Moved PredictorService to feature-specific services module
  - Updated main.py imports and router configuration
  - Removed old monolithic v1/ and services/ directories
- Fixed Pydantic import issue (BaseSettings moved to pydantic-settings)
- Validated that the restructured app imports and runs correctly

**Decisions Made:**
- Implemented feature-based API organization (DEC-006 added to decisions-log.md)

**Blockers:** Pydantic version compatibility issue with BaseSettings import (resolved)

**Next Session:** Create GitHub repository and download Kaggle datasets
---

## $(date '+%Y-%m-%d') - Kaggle Integration
- **Added Kaggle dependency:** Installed `kaggle` package and added to `requirements.txt`.
- **Configured Pydantic Settings:** Updated `app/core/config.py` to securely load `KAGGLE_USERNAME` and `KAGGLE_KEY`.
- **Environment Templates:** Added `KAGGLE_USERNAME` and `KAGGLE_KEY` placeholders to `.env.example`.
- **Data Directories:** Created `data/raw/` and `data/processed/` with `.gitignore` configurations.
- **Download Script:** Created `scripts/download_data.py` to fetch and extract the Home Credit Default Risk dataset securely injecting credentials.
