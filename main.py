#!/usr/bin/env python3
"""
E2H Medical Agent - Main Entry Point
Organized Medical AI System with Easy-to-Hard Curriculum Learning
"""

import subprocess
import sys
import os

def main():
    print("Starting E2H Medical Agent System...")
    print("Web interface will be available at: http://localhost:5001")
    
    # Run the Flask app from the backend directory
    backend_path = os.path.join(os.path.dirname(__file__), 'backend')
    
    try:
        # Change to backend directory and run the Flask app
        subprocess.run([
            sys.executable, 'medical_agent_app.py'
        ], cwd=backend_path, check=True)
    except KeyboardInterrupt:
        print("\nShutting down E2H Medical Agent System...")
    except subprocess.CalledProcessError as e:
        print(f"Error starting system: {e}")

if __name__ == '__main__':
    main()
