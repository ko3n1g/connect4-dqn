import collections
from typing import Union
import numpy as np
from pettingzoo.classic import connect_four_v3
from tqdm.cli import tqdm
import torch
from connect4Agent.agent import RandomAgent, Agent, DQNAgent
from connect4Agent.environment import transform_obs


def create_buffer(n_samples=1_000, agent: Union[None, Agent] = None) -> collections.deque:
    replay_buffer = collections.deque(maxlen=n_samples)
    agent = agent or RandomAgent()
    for _ in tqdm(range(n_samples)):
        env = connect_four_v3.env()
        env.reset()
        done = False
        i = 0
        while not done and len(replay_buffer) <= n_samples:
            obs, _, _, _ = env.last()
            if type(agent) is DQNAgent:
                observation = torch.tensor(
                    transform_obs(obs["observation"]), dtype=torch.float
                ).flatten()[None, :]
                action = int(agent.forward(observation).argmax())
            else:
                action = agent.forward(obs)
            env.step(action)
            next_obs, reward, done, _ = env.last()
            if i % 2 == 0:
                replay_buffer.append(
                    [
                        transform_obs(obs["observation"]),
                        action,
                        -1 if reward < 1 and i % 2 == 0 else reward,
                        transform_obs(next_obs["observation"]),
                        done,
                    ]
                )
                i = 1
            else:
                i = 0

        if len(replay_buffer) == n_samples:
            break

    return replay_buffer


def save_buffer(
    replay_buffer: collections.deque, file_path: str = "data/replay_buffer.npy"
):
    with open(file_path, "wb") as file:
        np.save(file, np.array(replay_buffer))


def load_buffer(file_path: str = "data/replay_buffer.npy") -> np.array:
    with open(file_path, "rb") as file:
        return np.load(file, allow_pickle=True).tolist()


if __name__ == "__main__":
    random_agent = RandomAgent()
    rb = create_buffer(n_samples=1_000_000)
    print(len(rb))
