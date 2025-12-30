#!/bin/bash

# Test runner script for GPSO project
# This script provides convenient commands for running different test suites

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

# Function to display usage
usage() {
    cat << EOF
Usage: ./run_tests.sh [OPTION]

Options:
    all             Run all tests
    unit            Run unit tests only
    integration     Run integration tests only
    coverage        Run tests with coverage report
    fast            Run tests without slow integration tests
    watch           Run tests in watch mode (re-run on file changes)
    pipeline        Run pipeline stage tests only
    streamlit       Run streamlit component tests only
    db              Run database tests only
    html-coverage   Generate HTML coverage report
    clean           Clean test artifacts and cache
    help            Display this help message

Examples:
    ./run_tests.sh all              # Run all tests
    ./run_tests.sh coverage         # Run with coverage
    ./run_tests.sh pipeline         # Test pipeline stages only
    ./run_tests.sh html-coverage    # Generate HTML coverage report

EOF
}

# Parse command line arguments
case "${1:-all}" in
    all)
        print_info "Running all tests..."
        uv run pytest tests/ -v
        ;;
    
    unit)
        print_info "Running unit tests..."
        uv run pytest tests/ -v -m "not integration"
        ;;
    
    integration)
        print_info "Running integration tests..."
        uv run pytest tests/test_integration.py -v
        ;;
    
    coverage)
        print_info "Running tests with coverage..."
        uv run pytest tests/ --cov=pipeline --cov=app --cov=database \
               --cov-report=term-missing --cov-report=xml
        print_info "Coverage report saved to coverage.xml"
        ;;
    
    html-coverage)
        print_info "Generating HTML coverage report..."
        uv run pytest tests/ --cov=pipeline --cov=app --cov=database \
               --cov-report=html
        print_info "HTML coverage report generated in htmlcov/"
        print_info "Open htmlcov/index.html in your browser"
        ;;
    
    fast)
        print_info "Running fast tests (excluding slow integration tests)..."
        uv run pytest tests/ -v -m "not slow"
        ;;
    
    watch)
        print_info "Running tests in watch mode..."
        print_warning "Press Ctrl+C to stop"
        uv run pytest tests/ -v --looponfail
        ;;
    
    pipeline)
        print_info "Running pipeline stage tests..."
        uv run pytest tests/test_detection.py tests/test_sentiments.py \
               tests/test_smoothing.py tests/test_normalization.py -v
        ;;
    
    streamlit)
        print_info "Running Streamlit component tests..."
        uv run pytest tests/test_chat_tools.py tests/test_chat_orchestrator.py -v
        ;;
    
    db)
        print_info "Running database tests..."
        uv run pytest tests/test_db.py -v
        ;;
    
    clean)
        print_info "Cleaning test artifacts..."
        rm -rf .pytest_cache
        rm -rf htmlcov
        rm -f coverage.xml
        rm -f .coverage
        find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete
        print_info "Cleanup complete!"
        ;;
    
    help|--help|-h)
        usage
        ;;
    
    *)
        print_error "Unknown option: $1"
        echo ""
        usage
        exit 1
        ;;
esac

exit 0
