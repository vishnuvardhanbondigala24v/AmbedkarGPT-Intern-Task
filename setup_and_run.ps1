# setup_and_run.ps1
# Run from project root in PowerShell (Run as Administrator recommended)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Stopping python/streamlit processes (if any)..."
Get-Process python, streamlit -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# Ensure venv exists
if (-Not (Test-Path .\.venv)) {
    Write-Host "Creating virtual environment .venv..."
    python -m venv .venv
}

# Activate venv in this session
Write-Host "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing core packages and pinning numpy < 2.0..."
python -m pip install "numpy<2.0.0" blinker sentence-transformers chromadb streamlit --upgrade

# Install CPU-only PyTorch wheels (safer on Windows)
Write-Host "Installing CPU PyTorch wheels (if needed)..."
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio --upgrade

# If you have a requirements.txt, try to install remaining deps
if (Test-Path requirements.txt) {
    Write-Host "Installing requirements.txt..."
    try {
        python -m pip install -r requirements.txt
    } catch {
        Write-Warning "pip install -r requirements.txt failed: $_"
    }
}

# Pre-download the sentence-transformers model to avoid blocking later
Write-Host "Pre-downloading embedding model (may take a minute)..."
try {
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
    Write-Host "Model predownload complete."
} catch {
    Write-Warning "Model predownload failed: $_"
}

# Ensure Chroma telemetry is disabled for this session
$env:CHROMA_TELEMETRY_ENABLED = 'false'
Write-Host "CHROMA_TELEMETRY_ENABLED set to false for this session."

# Optional: remove existing chroma_db to force a clean rebuild
if (Test-Path .\chroma_db) {
    Write-Host "Removing existing chroma_db folder to force rebuild..."
    try {
        Remove-Item -Recurse -Force .\chroma_db
        Write-Host "Removed chroma_db."
    } catch {
        Write-Warning "Could not remove chroma_db: $_"
    }
}

# Quick sqlite write test to detect disk I/O issues
Write-Host "Running sqlite write test..."
try {
    python -c "import sqlite3, os; os.makedirs(r'C:\\temp\\chroma_test', exist_ok=True); conn=sqlite3.connect(r'C:\\temp\\chroma_test\\test.db'); conn.execute('create table if not exists t(x int)'); conn.commit(); conn.close(); print('sqlite write OK')"
} catch {
    Write-Warning "sqlite write test failed: $_"
    Write-Warning "If this fails, try moving the project out of OneDrive/network drives or run PowerShell as Administrator."
}

# Run the precompute script if present
if (Test-Path .\scripts\precompute_embeddings.py) {
    Write-Host "Running precompute script (this may take several minutes)..."
    try {
        & python .\scripts\precompute_embeddings.py
        Write-Host "Precompute script finished."
    } catch {
        Write-Warning "Precompute script failed: $_"
    }
} else {
    Write-Host "No precompute script found at scripts/precompute_embeddings.py; skipping precompute."
}

# Final: start Streamlit
Write-Host "Starting Streamlit..."
try {
    streamlit run streamlit_app.py
} catch {
    Write-Warning "Failed to start Streamlit: $_"
    Write-Host "You can start it manually with: streamlit run streamlit_app.py"
}
