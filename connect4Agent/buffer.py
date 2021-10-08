import collections

import numpy as np
from pettingzoo.classic import connect_four_v3
from tqdm.cli import tqdm

from connect4Agent.agent import RandomAgent


def create_buffer(n_samples=1_000) -> collections.deque:
    replay_buffer = collections.deque(maxlen=n_samples)
    agent = RandomAgent()
    for _ in tqdm(range(n_samples)):
        env = connect_four_v3.env()
        env.reset()
        next_done = False
        while not next_done and len(replay_buffer) <= n_samples:
            obs, _, _, _ = env.last()
            action = agent.forward(obs)
            env.step(action)
            next_obs, reward, next_done, _ = env.last()
            replay_buffer.append(
                [
                    obs["observation"].sum(axis=2),
                    action,
                    reward,
                    next_obs["observation"].sum(axis=2),
                    next_done,
                ]
            )

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
