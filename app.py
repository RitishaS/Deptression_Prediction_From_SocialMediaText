"""
Depression Level Analyzer - Root Entry Point
Launches the backend Flask application
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
BACKEND_DIR = Path(__file__).parent / 'backend'
sys.path.insert(0, str(BACKEND_DIR))

# Import and run the Flask app from backend
from app import app

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Depression Level Analyzer")
    print("="*60)
    print("📊 Visit http://localhost:5000 in your browser")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
