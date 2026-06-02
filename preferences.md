# Backend Development Preferences

This document outlines the strict architectural and coding standards for the backend of this project. AI agents and developers must adhere to these guidelines to ensure consistency, maintainability, and scalability.

## 1. Directory Structure

The backend follows a **feature-based** directory structure. Each major feature (e.g., posts, discovery, messaging) is encapsulated in its own directory within `app/`.

```text
backend/app/
├── core/               # Centralized logic (auth, db, config, errors, media)
├── <feature_name>/     # Feature-specific directory
│   ├── __init__.py
│   ├── models.py       # Database models (Beanie/ODM)
│   ├── routes.py       # FastAPI router and endpoints
│   ├── schemas.py      # Pydantic request/response schemas
│   ├── services.py     # Business logic
│   └── dependencies.py # Feature-specific dependencies (optional)
├── Exceptions.py
├── Middleware.py
└── main.py             # App entry point
|
```

### Centralized Core
Always place shared logic in `app/core/`:
- `config.py`: Application settings and environment variables.
- `errors.py`: Custom exceptions and global exception handlers.
- `middleware.py`: Global middleware (CORS, logging, etc.).
- `db/`: Database connection and client initialization.

## 2. Separation of Concerns

Strict adherence to the following roles is required:

### Routes (`routes.py`)
- **Responsibility**: HTTP interface, request validation (via schemas), and response formatting.
- **Constraints**:
    - Use FastAPI `APIRouter`.
    - Inject dependencies (like `get_current_user`) at the endpoint level.
    - **No business logic**. Call methods from `services.py`.
    - Map service results to Pydantic response models.

### Services (`services.py`)
- **Responsibility**: Business logic, data manipulation, and coordination between different models/features.
- **Constraints**:
    - Encapsulate logic within classes (e.g., `PostService`).
    - Raise custom exceptions from `app/core/errors.py` when business rules are violated.
    - Handle complex operations (e.g., processing hashtags, sending notifications).

### Models (`models.py`)
- **Responsibility**: Database schema definitions.
- **Constraints**:
    - Use Beanie `Document` for MongoDB collections.
    - Use type hinting for all fields.

### Schemas (`schemas.py`)
- **Responsibility**: Data validation and serialization.
- **Constraints**:
    - Use Pydantic `BaseModel`.
    - Separate `Request` and `Response` models.

## 3. Coding Style

### Type Hinting
All functions, methods, and variables must be strictly type-hinted.
```python
async def get_user_posts(user_id: str, limit: int = 10) -> List[Post]:
    ...
```

### Docstrings
Provide descriptive docstrings for all classes and functions following the Google style or similar.
```python
def extract_hashtags(text: str) -> List[str]:
    """
    Extracts hashtags from a given string.
    
    Args:
        text: The string to parse.
        
    Returns:
        A list of unique hashtags found in the text.
    """
```

### PEP 8
Follow PEP 8 standards strictly (indentation, naming conventions, imports).

## 4. Error Handling

Do **not** raise generic `HTTPException` within services. Use the centralized error handling pattern:

1.  **Define** a custom exception in `app/core/errors.py`.
2.  **Register** the exception with a handler in `register_exceptions(app: FastAPI)`.
3.  **Raise** the custom exception in your service logic.

**Example Service Logic:**
```python
if not post:
    raise PostNotFoundException()
```

**Example Centralized Error Definition:**
```python
class PostNotFoundException(WeTalkException):
    """Exception raised when a post is not found."""
    pass

# In register_exceptions
app.add_exception_handler(
    PostNotFoundException,
    create_exception_handler(
        status_code=status.HTTP_404_NOT_FOUND,
        initial_detail={
            "message": "Post not found",
            "error_code": "post_not_found",
            "resolution": "Check the post ID",
        },
    ),
)
```

## 5. Technical Examples for AI Agents

### Recommended Route Pattern
```python
@router.post("/", status_code=status.HTTP_201_CREATED, response_model=PostResponse)
async def create_post(
    req: CreatePostRequest,
    current_user: User = Depends(get_current_user)
) -> PostResponse:
    service = PostService()
    new_post = await service.create_post(user_id=str(current_user.id), req=req)
    return PostResponse(**new_post.model_dump())
```

### Recommended Service Pattern
```python
class PostService:
    async def create_post(self, user_id: str, req: CreatePostRequest) -> Post:
        # Validation
        if not req.caption:
            raise ContentValidationException("Caption cannot be empty")
            
        # Logic
        new_post = Post(owner_id=user_id, caption=req.caption)
        await new_post.save()
        return new_post
```

# 2. Third-party
import pandas as pd
from fastapi import APIRouter
from sqlalchemy.orm import Session

# 3. Local / internal
from app.schemas import BorrowerProfile
from app.services.predictor_service import PredictorService
```

---

## 3. Class Structure Template

All classes should follow this internal ordering:

```python
class ExampleService:
    """
    One-line summary of what this class does.

    Longer description if needed. Explains the responsibility,
    not the implementation.

    Attributes:
        _dependency (SomeType): Description of the dependency.
    """

    # --- Class-level constants ---
    DEFAULT_THRESHOLD: float = 0.5

    def __init__(self, dependency: SomeType) -> None:
        """Initialise with required dependencies."""
        self._dependency = dependency

    # --- Public methods (alphabetical within group) ---

    def primary_action(self, input_data: InputType) -> OutputType:
        """
        Summary of what this method does.

        Args:
            input_data: Description of the input.

        Returns:
            Description of what is returned.

        Raises:
            ValueError: When input_data is invalid.
        """
        validated = self._validate(input_data)
        return self._process(validated)

    # --- Private / helper methods ---

    def _validate(self, data: InputType) -> InputType:
        """Internal validation logic."""
        ...

    def _process(self, data: InputType) -> OutputType:
        """Internal processing logic."""
        ...
```

---

## 4. Project Folder Structure

```
credit-risk-api/
│
├── app/                          # FastAPI application
│   ├── __init__.py
│   ├── main.py                   # App factory, router registration
│   ├── core/
│   │   ├── config.py             # Settings class (pydantic BaseSettings)
│   │   ├── security.py           # JWT logic
│   │   └── exceptions.py         # Custom exception classes
│   ├── routes/
│   │   ├── auth.py               # /auth endpoints
│   │   ├── scoring.py            # /score endpoints
│   │   └── health.py             # /health endpoint
│   ├── schemas/
│   │   ├── borrower.py           # BorrowerProfile Pydantic schema
│   │   └── risk_response.py      # RiskResponse Pydantic schema
│   ├── models/                   # SQLAlchemy DB models
│   │   ├── lender.py
│   │   └── scoring_log.py
│   ├── services/                 # Business logic classes
│   │   ├── predictor_service.py  # PredictorService
│   │   └── explainer_service.py  # SHAPExplainerService
│   └── repositories/             # DB access classes
│       ├── lender_repository.py
│       └── scoring_repository.py
│
├── ml/                           # Machine learning layer
│   ├── notebooks/                # Jupyter EDA + training notebooks
│   │   ├── 01_eda.ipynb
│   │   ├── 02_feature_engineering.ipynb
│   │   └── 03_model_training.ipynb
│   ├── src/
│   │   ├── data_processor.py     # DataProcessor class
│   │   ├── model_trainer.py      # ModelTrainer class
│   │   └── shap_explainer.py     # SHAPExplainer class
│   └── artefacts/                # Saved .pkl files (gitignored)
│       ├── model.pkl
│       └── preprocessor.pkl
│
├── data/
│   ├── raw/                      # Original downloaded data (gitignored)
│   └── processed/                # Cleaned datasets (gitignored)
│
├── tests/
│   ├── unit/
│   │   ├── test_predictor_service.py
│   │   └── test_explainer_service.py
│   └── integration/
│       └── test_scoring_endpoint.py
│
├── migrations/                   # Alembic migration files
├── .env                          # Local secrets (gitignored)
├── .env.example                  # Template for .env (committed)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt          # Dev-only deps (pytest, black, etc.)
└── README.md
```

---

## 5. Environment & Tooling

### IDEs
- **Primary:** VS Code
- **Secondary:** Antigravity IDE
  - ⚠️ Note: Add any Antigravity-specific formatting or plugin preferences here as
    they are discovered. For now, Black + isort formatting rules apply in both.

### Virtual Environment
- Use `python -m venv venv` — **do not use conda** for this project
- Always activate before running anything: `source venv/bin/activate`
- Pin all versions in `requirements.txt` (use `pip freeze > requirements.txt`)

### Git Conventions
- Branch naming: `feature/phase-1-eda`, `fix/shap-null-handling`
- Commit messages: `[PHASE-X] Short imperative description`
  - Example: `[PHASE-1] Add SMOTE to training pipeline`
- Never commit `.env`, `data/raw/`, `data/processed/`, or `ml/artefacts/`
- Use `.gitignore` to exclude the above

---

## 6. How Changes Should Be Made

### When suggesting code changes:
1. **Show the full class**, not just a snippet, unless the file is very large
2. **Explain the reason** for the change in one sentence before the code block
3. **Never silently change** class names, method signatures, or file locations
4. **Flag breaking changes** clearly with ⚠️ before implementing them
5. If a change touches `decisions-log.md` entries, say so

### When adding a new feature:
1. Propose the class name and its location in the folder structure first
2. Get confirmation before writing the full implementation
3. Write the class with full type hints and docstrings
4. Suggest a test for it

### When fixing a bug:
1. State what the bug is and which class/method it's in
2. Show only the affected method (not the whole file) unless context is needed
3. Explain what caused the bug in plain English

---

## 7. Testing Preferences

- **Framework:** `pytest`
- Test files mirror the `app/` structure under `tests/`
- Each public method in a service class should have at least one test
- Use `pytest-mock` for mocking dependencies
- Run tests before every commit: `pytest tests/`

---

## 8. ML-Specific Preferences

- Notebooks are for **exploration only** — production logic goes in `ml/src/` classes
- Each notebook should have a markdown summary cell at the top explaining its purpose
- Never hardcode dataset paths — use the `config.py` settings class
- Save both the model **and** the preprocessor as separate `.pkl` files
- Always set `random_state=42` for reproducibility
