#!/usr/bin/env python3
"""KitaMap Berlin main entry point.

Berlin daycare center spatial analysis and coverage assessment.

Usage:
    python main.py           # Run complete analysis
    python main.py --help    # Show help options
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    import run_analysis

    run_analysis.main()
