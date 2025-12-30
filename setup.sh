#!/bin/bash
# Development setup script for GPSO

set -e  # Exit on error

echo "🚀 Setting up GPSO development environment..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    echo "✅ uv installed successfully!"
else
    echo "✅ uv is already installed"
fi

echo ""
echo "🐍 Creating virtual environment..."
uv venv

echo ""
echo "📚 Installing dependencies..."
uv sync

echo ""
echo "� Installing Playwright browsers..."
uv run playwright install

echo ""
echo "�🎉 Setup complete!"
echo ""
echo "To get started:"
echo "  Run the pipeline: uv run python pipeline/main.py"
echo "  Start Streamlit: uv run streamlit run streamlit/app.py"
echo ""
echo "For more info, see the Quick Start section in README.md"
