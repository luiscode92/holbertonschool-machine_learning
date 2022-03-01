#!/usr/bin/env python3
"""
Implements a full training.
Renders the environment every 1000 episodes computed.
"""


from policy_gradient import policy_gradient
import numpy as np


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Return: all values of the score
    (sum of all rewards during one episode loop).
    """
    # 4 states and 2 actions
    weight = np.random.rand(4, 2)
    episode_rewards = []

    for episode in range(nb_episodes):
        state = env.reset()[None, :]
        gradients = []
        rewards = []
        score = 0

        while True:
            if show_result and (episode % 1000 == 0):
                env.render()
            action, grad = policy_gradient(state, weight)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[None, :]
            gradients.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state

            if done:
                break

        for i in range(len(gradients)):
            weight += alpha * gradients[i] *\
                sum([r * gamma ** r for _, r in enumerate(rewards[i:])])

        episode_rewards.append(score)

        print("{}: {}".format(episode, score), end="\r", flush=False)

    return episode_rewards
