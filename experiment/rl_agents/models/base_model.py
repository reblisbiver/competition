#!/usr/bin/env python3
"""
Base RL Model Template
Copy this file to create a new RL model.

Usage: python3 your_model.py <user_id> <last_action> <last_reward>
Output: Must print exactly "LEFT" or "RIGHT" (nothing else)

Parameters:
    user_id (str): Unique session identifier
    last_action (str): Previous action "LEFT"/"RIGHT" or "None" for first trial
    last_reward (str): Reward from last action "0"/"1" or "None" for first trial

IMPORTANT:
- Your script MUST print exactly "LEFT" or "RIGHT" as the only output
- Do not print any debug messages to stdout (use stderr if needed)
- Store any persistent state in files under rl_agents/q_tables/ or similar
"""
import sys
import os

def main():
    if len(sys.argv) < 4:
        print("LEFT")
        return
    
    user_id = sys.argv[1]
    last_action = sys.argv[2] if sys.argv[2] not in ["None", "null", ""] else None
    try:
        last_reward = float(sys.argv[3]) if sys.argv[3] not in ["None", "null", ""] else 0.0
    except:
        last_reward = 0.0
    
    action = "LEFT"
    print(action)

if __name__ == "__main__":
    main()
