import sys
import os
from pathlib import Path

# DEBUG: Check if files actually exist on the server
print("--- DEBUG START ---")
print(f"Current WD: {os.getcwd()}")
if os.path.exists("src"):
    print(f"src contents: {os.listdir('src')}")
    if os.path.exists("src/models"):
        print(f"src/models contents: {os.listdir('src/models')}")
    else:
        print("ALERT: src/models directory NOT FOUND")
else:
    print("ALERT: src directory NOT FOUND")
print("--- DEBUG END ---")

# Add the current directory to sys.path so src can be imported
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))

# Importing the module executes the script because src/app.py is written as a script.
# This is safe on Streamlit SDK because it wraps this execution in the runtime automatically.
import src.app
