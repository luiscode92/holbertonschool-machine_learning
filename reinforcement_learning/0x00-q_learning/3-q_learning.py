#!/usr/bin/env python3
"""
Performs Q-learning
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Returns: Q, total_rewards
    """
    total_rewards = []
    max_epsilon = epsilon

    # Q-learning algorithm
    for episode in range(episodes):
        state = env.reset()
        done = False
        rewards_current_episode = 0

        for _ in range(max_steps):
            # Exploration-exploitation trade-off
            action = epsilon_greedy(Q, state, epsilon)

            new_state, reward, done, _ = env.step(action)

            if done is True and reward == 0:
                reward = -1

            # Update Q-table for Q(s, a)
            Q[state, action] = Q[state, action] * (1 - alpha) + \
                alpha * (reward + gamma * np.max(Q[new_state, :]))

            # Transition to the next state
            state = new_state
            rewards_current_episode += reward

            if done is True:
                break

        # Exploration rate decay
        epsilon = min_epsilon + \
            (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)

        total_rewards.append(rewards_current_episode)

    return Q, total_rewards
