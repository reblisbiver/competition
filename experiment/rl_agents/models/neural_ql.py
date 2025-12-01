#!/usr/bin/env python3
"""
Neural Q-Learning Agent Wrapper
This is an adapter that calls the neural_ql_agent from the neural_ql module.

Usage: python3 neural_ql.py <user_id> <last_action> <last_reward>
Output: LEFT or RIGHT
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'neural_ql'))

from neural_ql_agent import main

if __name__ == "__main__":
    main()
