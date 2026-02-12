#!/bin/bash
# Quick bootstrap for fresh Prime Intellect pods
# Run: bash bootstrap.sh

echo "=== Pod Bootstrap Starting ==="

# Detect if we're in a venv (if so, don't use --user)
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Detected venv: $VIRTUAL_ENV"
    PIP="pip install"
else
    echo "No venv detected, using --user installs"
    PIP="pip install --user"
fi

# Update pip
$PIP --upgrade pip

# Core ML/torch stack
echo "Installing PyTorch..."
$PIP torch torchvision torchaudio

# Transformers ecosystem
echo "Installing transformers ecosystem..."
$PIP transformers accelerate datasets huggingface_hub

# Evaluation
echo "Installing eval harness..."
$PIP lm-eval  # lm-evaluation-harness

# Training tools (from your CTO's toolkit) - these can be finicky
echo "Installing trl..."
$PIP trl

echo "Installing axolotl (may fail on some systems)..."
$PIP axolotl || echo "⚠ axolotl failed - may need system deps, skipping"

echo "Installing vllm (may fail on some systems)..."
$PIP vllm || echo "⚠ vllm failed - may need specific CUDA version, skipping"

# API clients for Kimi-k2.5
echo "Installing API clients..."
$PIP openai httpx

# Data handling
echo "Installing data tools..."
$PIP pandas jsonlines python-dotenv tqdm

# Export API keys from .env
if [ -f .env ]; then
    export $(cat .env | xargs)
    echo "API keys loaded from .env"
fi

# Login to HuggingFace (will prompt for token if not cached)
hf auth login --token $HF_TOKEN 2>/dev/null || echo "Set HF_TOKEN or run 'hf auth login' manually"

echo ""
echo "=== Bootstrap complete ==="
echo "If axolotl or vllm failed, they may already be in the pod image or need manual install"