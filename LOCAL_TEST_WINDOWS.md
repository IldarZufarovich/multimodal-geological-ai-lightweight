# Local Windows test

From Anaconda/Jupyter or PowerShell, use the same environment that contains the dependencies.

## PowerShell
```powershell
cd "PATH_TO_PROJECT"
python app.py
```
Open `http://127.0.0.1:7860` if the browser does not open automatically.

## Jupyter
```python
import subprocess, sys, os
from pathlib import Path
p = subprocess.Popen([sys.executable, "app.py"], cwd=str(Path.cwd()), env=os.environ.copy())
print("http://127.0.0.1:7860")
```

Tesseract v0.2 is auto-discovered from PATH, `%LOCALAPPDATA%\\Programs\\Tesseract-OCR`, or standard Program Files locations.
