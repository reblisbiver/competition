#!/usr/bin/env python3
"""
Random Agent - Chooses LEFT or RIGHT with 50/50 probability.
Useful as a baseline for comparison.
"""
import sys
import random

def main():
    print(random.choice(["LEFT", "RIGHT"]))

if __name__ == "__main__":
    main()
