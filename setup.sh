#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Hampi Revived — Environment Setup
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          HAMPI REVIVED — Environment Setup               ║"
echo "║   3D Reconstruction · Computer Vision · Data Science     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# 1. Python version check
PYTHON=$(command -v python3.11 || command -v python3.10 || command -v python3)
if [ -z "$PYTHON" ]; then
    echo "❌  Python 3.10+ required. Please install it first."
    exit 1
fi
echo "✅  Python: $($PYTHON --version)"

# 2. Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦  Creating virtual environment..."
    $PYTHON -m venv venv
else
    echo "ℹ️   Virtual environment already exists."
fi

# 3. Activate
source venv/bin/activate
echo "✅  Virtual environment activated."

# 4. Upgrade pip
pip install --upgrade pip --quiet

# 5. Install dependencies
echo "📦  Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# 6. Create output dirs
mkdir -p data/raw data/processed
mkdir -p outputs/features outputs/point_clouds outputs/meshes outputs/visualizations outputs/reports
touch data/.gitkeep outputs/.gitkeep

# 7. Env file template
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# ─── Groq API Key ──────────────────────────────────────────────────────────────
# Get yours at https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here
EOF
    echo "⚠️   .env file created — add your GROQ_API_KEY inside it."
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Setup complete! Next steps:                             ║"
echo "║                                                          ║"
echo "║  1.  source venv/bin/activate                           ║"
echo "║  2.  Edit .env → add your GROQ_API_KEY                  ║"
echo "║  3.  python pipeline.py                                  ║"
echo "║  4.  jupyter lab notebooks/hampi_full_pipeline.ipynb    ║"
echo "╚══════════════════════════════════════════════════════════╝"
