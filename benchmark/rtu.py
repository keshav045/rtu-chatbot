"""
Compatibility entry point for benchmark/rtu_benchmark.py
"""
import sys
from pathlib import Path

# Add benchmark directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))
from rtu_benchmark import main

if __name__ == "__main__":
    main()
