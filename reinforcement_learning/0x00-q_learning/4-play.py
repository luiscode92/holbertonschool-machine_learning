#!/usr/bin/env python3
"""
Function that has the trained agent play an episode
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    You should always exploit the Q-table.
    Returns: the total rewards for the episode.
    """
    env.reset()
    state = env.reset()
    done = False
    env.render()

    for _ in range(max_steps):

        # Take the action (index) that have the maximum expected future reward
        # given that state
        action = np.argmax(Q[state, :])

        new_state, reward, done, _ = env.step(action)

        env.render()
        if done:
            break
        state = new_state
    env.close()

    return reward
