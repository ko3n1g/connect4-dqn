import pettingzoo

from connect4Agent.agent import DQNAgent
from connect4Agent.memory import Memory


def train(
    num_features: int,
    num_action: int,
    memory_size: int,
    env: pettingzoo.AECEnv,
    n_episodes: int,
):

    value_network = DQNAgent(in_shape=num_features, out_shape=num_action)
    target_network = DQNAgent(in_shape=num_features, out_shape=num_action).clone_from(
        value_network
    )

    memory = Memory(memory_size)

    for n in range(n_episodes):

        s, r, done, s_prime = env.last()
