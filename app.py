import sys
from pathlib import Path
import streamlit.web.cli as stcli

if __name__ == "__main__":
    # Ensure src is in python path
    root_path = Path(__file__).parent
    sys.path.insert(0, str(root_path))
    
    # Run the streamlit app
    sys.argv = ["streamlit", "run", "src/app.py"]
    sys.exit(stcli.main())
