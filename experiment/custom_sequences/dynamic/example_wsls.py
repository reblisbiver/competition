#!/usr/bin/env python3
"""
Example Dynamic Sequence: Win-Stay-Lose-Shift Opponent

This dynamic schedule adapts based on participant behavior:
- If participant won last trial, make same choice less rewarding
- If participant lost, make opposite choice more rewarding

Usage: python3 example_wsls.py <bias_rewards_json> <unbias_rewards_json> <choices_json> <user_id>
Output: Print two comma-separated values: biased_reward, unbiased_reward
"""
import sys
import json
import random

def main():
    bias_rewards = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
    unbias_rewards = json.loads(sys.argv[2]) if len(sys.argv) > 2 else []
    choices = json.loads(sys.argv[3]) if len(sys.argv) > 3 else []
    
    if len(choices) == 0:
        biased_reward = 1 if random.random() < 0.5 else 0
        unbiased_reward = 1 if random.random() < 0.5 else 0
    else:
        last_choice_was_biased = choices[-1] == 'True'
        last_biased_reward = bias_rewards[-1] if bias_rewards else 0
        last_unbiased_reward = unbias_rewards[-1] if unbias_rewards else 0
        
        if last_choice_was_biased:
            last_won = last_biased_reward == 1
        else:
            last_won = last_unbiased_reward == 1
        
        if last_won:
            if last_choice_was_biased:
                biased_reward = 1 if random.random() < 0.3 else 0
                unbiased_reward = 1 if random.random() < 0.7 else 0
            else:
                biased_reward = 1 if random.random() < 0.7 else 0
                unbiased_reward = 1 if random.random() < 0.3 else 0
        else:
            biased_reward = 1 if random.random() < 0.5 else 0
            unbiased_reward = 1 if random.random() < 0.5 else 0
    
    print(f"{biased_reward}, {unbiased_reward}")

if __name__ == "__main__":
    main()
