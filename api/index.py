import sys, os

# Add Code directory to python path
base_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(base_dir, '..', 'Code')
sys.path.append(code_dir)

from app import app
