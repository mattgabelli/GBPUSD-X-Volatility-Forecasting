import os
import sys
from pathlib import Path

# Automatically detect project root (the folder that contains .git)
def find_project_root():
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / ".git").exists():
            return parent
    # Fallback: assume parent of current file
    return current_path.parents[1]

project_root = find_project_root()
os.chdir(project_root)

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

print(f"✅ Project root set to: {project_root}")