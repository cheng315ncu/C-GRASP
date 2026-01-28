#!/bin/bash
# =============================================================================
# CGRASP - Quick Setup & Run Script
# =============================================================================
# Usage:
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh [--no-rag] [--skip-install]
#
# Options:
#   --no-rag        Run without RAG (skip PDF check)
#   --skip-install  Skip dependency installation
# =============================================================================

set -e  # Exit on error

# Parse arguments
USE_RAG=true
SKIP_INSTALL=false
for arg in "$@"; do
    case $arg in
        --no-rag)
            USE_RAG=false
            shift
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
    esac
done

# Get script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "CGRASP - Clinical Grade RAG System for Psychophysiology"
echo "========================================================================"
echo ""
echo "Project root: $SCRIPT_DIR"
echo ""

# Step 1: Check Python version
echo "[Step 1/5] Checking Python version..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✓ Python version: $python_version"
else
    echo "✗ Python 3 not found. Please install Python 3.8+"
    exit 1
fi
echo ""

# Step 2: Install dependencies
if [ "$SKIP_INSTALL" = false ]; then
    echo "[Step 2/5] Installing dependencies..."
    echo "This may take a few minutes..."
    pip install -q -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "[Step 2/5] Skipping dependency installation (--skip-install)"
fi
echo ""

# Step 3: Create required directories
echo "[Step 3/5] Creating directory structure..."
mkdir -p data
mkdir -p clinical_pdfs
mkdir -p chroma_db
mkdir -p outputs/reports
mkdir -p logs

# Create .gitkeep files to preserve empty directories
touch clinical_pdfs/.gitkeep
touch data/.gitkeep

echo "✓ Directory structure created"
echo ""

# Step 4: Check PDF files (if RAG enabled)
echo "[Step 4/5] Checking configuration..."
if [ "$USE_RAG" = true ]; then
    pdf_count=$(find clinical_pdfs -name "*.pdf" 2>/dev/null | wc -l)
    if [ "$pdf_count" -eq 0 ]; then
        echo "⚠ Warning: No PDF files found in clinical_pdfs/"
        echo ""
        echo "For RAG-enhanced analysis, please add clinical PDFs:"
        echo "  - HRV interpretation guidelines"
        echo "  - Psychophysiology research papers"
        echo "  - Autonomic nervous system studies"
        echo ""
        echo "You can continue without RAG using: ./setup_and_run.sh --no-rag"
        echo ""
    else
        echo "✓ Found $pdf_count PDF file(s) for RAG"
    fi
fi

# Step 5: Check CSV data file
CSV_PATH="${CGRASP_CSV_PATH:-data/HRV_all_final.csv}"
if [ ! -f "$CSV_PATH" ]; then
    echo "⚠ Warning: CSV data file not found: $CSV_PATH"
    echo ""
    echo "Please either:"
    echo "  1. Place your HRV CSV file at: $CSV_PATH"
    echo "  2. Set CGRASP_CSV_PATH environment variable"
    echo ""
    echo "Example:"
    echo "  cp /path/to/your/data.csv data/HRV_all_final.csv"
    echo "  # or"
    echo "  export CGRASP_CSV_PATH=/path/to/your/data.csv"
    echo ""
else
    echo "✓ CSV data file found: $CSV_PATH"
fi
echo ""

# Setup complete
echo "========================================================================"
echo "✓ Setup complete!"
echo "========================================================================"
echo ""
echo "To run the analysis:"
if [ "$USE_RAG" = true ]; then
    echo "  python main.py --rag      # With RAG (requires PDFs)"
fi
echo "  python main.py --no-rag   # Without RAG"
echo ""
echo "Configuration:"
echo "  - Edit config.py for detailed settings"
echo "  - Or copy .env.example to .env for environment variables"
echo ""

# Ask to run
read -p "Run analysis now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    if [ "$USE_RAG" = true ]; then
        echo "Starting analysis with RAG..."
        python3 main.py --rag
    else
        echo "Starting analysis without RAG..."
        python3 main.py --no-rag
    fi
fi

echo ""
echo "========================================================================"
echo "Done!"
echo "========================================================================"

