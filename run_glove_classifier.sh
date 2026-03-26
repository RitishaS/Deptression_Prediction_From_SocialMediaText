#!/usr/bin/env bash
# Quick Start Guide for GloVe Depression Classifier

echo "============================================================"
echo "🚀 QUICK START: GloVe Depression Classifier"
echo "============================================================"
echo ""

# Check Python
echo "⏳ Checking Python installation..."
python --version

# Install dependencies
echo ""
echo "📦 Installing required packages..."
pip install -r requirements_glove.txt

# Run main pipeline
echo ""
echo "✨ Starting the pipeline..."
echo "============================================================"
python src/GloVe_Depression_Classifier.py

echo ""
echo "✅ Complete! Check output above for results."
