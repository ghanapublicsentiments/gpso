# Development setup script for GPSO (Windows PowerShell)
# Usage: .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up GPSO development environment..." -ForegroundColor Cyan
Write-Host ""

# Check if uv is installed
try {
    $uvVersion = & uv --version 2>$null
    Write-Host "✅ uv is already installed: $uvVersion" -ForegroundColor Green
} catch {
    Write-Host "📦 Installing uv..." -ForegroundColor Yellow
    
    # Install uv using the Windows installer
    irm https://astral.sh/uv/install.ps1 | iex
    
    # Refresh PATH for current session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    Write-Host "✅ uv installed successfully!" -ForegroundColor Green
}

Write-Host ""
Write-Host "🐍 Creating virtual environment..." -ForegroundColor Cyan
& uv venv

Write-Host ""
Write-Host "📚 Installing dependencies..." -ForegroundColor Cyan
& uv sync

Write-Host ""
Write-Host "� Installing Playwright browsers..." -ForegroundColor Cyan
& uv run playwright install

Write-Host ""
Write-Host "�🎉 Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To get started:" -ForegroundColor Yellow
Write-Host "  Run the pipeline: uv run python pipeline/main.py"
Write-Host "  Start Streamlit: uv run streamlit run streamlit/app.py"
Write-Host ""
Write-Host "For more info, see the Quick Start section in README.md"
Write-Host ""
