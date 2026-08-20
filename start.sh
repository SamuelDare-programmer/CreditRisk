#!/bin/bash

# Default to running the API if SERVICE_TYPE is not set
SERVICE_TYPE=${SERVICE_TYPE:-api}

if [ "$SERVICE_TYPE" = "frontend" ]; then
    echo "Starting Streamlit Frontend..."
    exec streamlit run streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
else
    echo "Starting FastAPI Backend..."
    exec uvicorn app.main:app --host 0.0.0.0 --port=${PORT:-8000}
fi
