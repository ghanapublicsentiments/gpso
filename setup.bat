@echo off
REM Development setup script for GPSO (Windows Command Prompt)
REM Usage: setup.bat

echo 🚀 Setting up GPSO development environment...
echo.

REM Check if uv is installed
where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo ✅ uv is already installed
) else (
    echo 📦 Installing uv...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo ❌ Failed to install uv
        exit /b 1
    )
    echo ✅ uv installed successfully!
)

echo.
echo 🐍 Creating virtual environment...
uv venv
if %errorlevel% neq 0 (
    echo ❌ Failed to create virtual environment
    exit /b 1
)

echo.
echo 📚 Installing dependencies...
uv sync
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    exit /b 1
)

echo.
echo � Installing Playwright browsers...
uv run playwright install
if %errorlevel% neq 0 (
    echo ❌ Failed to install Playwright browsers
    exit /b 1
)

echo.
echo �🎉 Setup complete!
echo.
echo To get started:
echo   Run the pipeline: uv run python pipeline\main.py
echo   Start Streamlit: uv run streamlit run streamlit\app.py
echo.
echo For more info, see the Quick Start section in README.md
echo.
