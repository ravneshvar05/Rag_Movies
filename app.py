import sys
import os
from pathlib import Path

# Add the current directory to sys.path so src can be imported
root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))

# Streamlit-compatible execution: Load the code and exec() it.
# This ensures Streamlit can re-run the script on interactions, which 'import' breaks (due to caching).
app_path = root_path / "src" / "app.py"

with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()

# Executing in the global namespace
exec(code, globals())
