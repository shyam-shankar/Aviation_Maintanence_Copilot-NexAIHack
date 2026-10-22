#!/usr/bin/env bash
# Quick launcher for AeroGraph Copilot Streamlit UI
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ -f ".venv/bin/streamlit" ]; then
    .venv/bin/streamlit run app.py --server.port 8501
else
    echo "Virtual environment not found in $DIR/.venv"
    echo "Please activate your virtual environment or install requirements."
fi
