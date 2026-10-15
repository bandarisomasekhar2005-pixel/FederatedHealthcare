"""
traditional_testing.py
-----------------------
Root-level executable wrapper for Traditional Centralized 80:20 Train/Test Evaluation.
Calls scripts/traditional_testing.py to evaluate centralized baseline while preserving
all existing federated learning pipelines and artifacts.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.traditional_testing import run_traditional_testing

if __name__ == "__main__":
    run_traditional_testing()
