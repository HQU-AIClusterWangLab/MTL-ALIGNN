"""Compatibility CLI for the requested src/train path; implementation is reused."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.analyze_literature_validation import main

if __name__ == '__main__':
    main()
