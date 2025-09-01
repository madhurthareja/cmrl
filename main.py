#!/usr/bin/env python3
"""
E2H Medical Agent - Main Entry Point
Organized Medical AI System with Easy-to-Hard Curriculum Learning
"""

import sys
import os

# Add backend to Python path for imports
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Change to backend directory for relative file paths
os.chdir(backend_path)

from medical_agent_app import app

if __name__ == '__main__':
    print("Starting E2H Medical Agent System...")
    print("Web interface available at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
