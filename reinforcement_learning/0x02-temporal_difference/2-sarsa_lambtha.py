#!/usr/bin/env python3
"""
Performs SARSA(λ)
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Returns: Q, the updated Q table
    """
    state_space_size = env.observation_space.n
    eps = epsilon

    # Eligibility traces
    Et = np.zeros((Q.shape))

    for i in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        for _ in range(max_steps):
            Et *= lambtha * gamma
            Et[state, action] += 1
            new_state, reward, done, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon)
            # Goal
            if env.desc.reshape(state_space_size)[new_state] == b'G':
                reward = 1
            # Hole
            if env.desc.reshape(state_space_size)[new_state] == b'H':
                reward = -1
            # Action-value form of the TD error
            delta_t = \
                reward + gamma * Q[new_state, new_action] - Q[state, action]
            # It has the same update rule as TD(lambtha)
            Q[state, action] = \
                Q[state, action] + alpha * delta_t * Et[state, action]
            if done:
                break
            state = new_state
            action = new_action
        epsilon = \
            min_epsilon + (eps - min_epsilon) * np.exp(-epsilon_decay * i)
    return Q


def epsilon_greedy(Q, state, epsilon):
    """
    Returns: the next action index
    """
    p = np.random.uniform(0, 1)

    if p < epsilon:
        """
        Explore: select a random action
        """
        action = np.random.randint(Q.shape[1])
    else:
        """
        Exploit: select the action with max value (future reward)
        """
        action = np.argmax(Q[state, :])

    return action
