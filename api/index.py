"""
Vercel Serverless Function entry point.
Exposes the Flask WSGI application for the @vercel/python runtime.
"""

import os
import sys

# Ensure the root project directory is on sys.path so modules can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app

# Vercel's Python runtime discovers 'app' as the WSGI callable
