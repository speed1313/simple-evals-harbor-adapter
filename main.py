#!/usr/bin/env python
"""
Entry point for running simple_evals from the command line.

This script sets up the Python path correctly so the package can be run
from within its own directory.
"""
import sys
from pathlib import Path

# Add the parent directory to the path so we can import simple_evals as a package
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now we can import using the package name
from simple_evals.simple_evals import main

if __name__ == "__main__":
    main()
