import sys
from pathlib import Path

# Add the current directory to sys.path so src can be imported
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))

# Importing the module executes the script because src/app.py is written as a script.
# This is safe on Streamlit SDK because it wraps this execution in the runtime automatically.
import src.app
