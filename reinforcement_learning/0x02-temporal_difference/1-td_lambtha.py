#!/usr/bin/env python3
"""
Performs the TD(λ) algorithm
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Returns: V, the updated value estimate
    """
    state_space_size = env.observation_space.n

    # Eligibility traces
    Et = [0 for _ in range(state_space_size)]

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            Et = list(np.array(Et) * lambtha * gamma)
            Et[state] += 1.0
            action = policy(state)
            new_state, reward, done, _ = env.step(action)
            # Goal
            if env.desc.reshape(state_space_size)[new_state] == b'G':
                reward = 1
            # Hole
            if env.desc.reshape(state_space_size)[new_state] == b'H':
                reward = -1
            # TD error
            delta_t = reward + gamma * V[new_state] - V[state]
            # V(s) <-- V(s) + alpha * delta_t * Et(s)
            V[state] = V[state] + alpha * delta_t * Et[state]
            if done:
                break
            state = new_state
    return V
