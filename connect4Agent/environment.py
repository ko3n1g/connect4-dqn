import math
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.cli import tqdm
from pettingzoo.classic import connect_four_v3

from connect4Agent.agent import Agent, DQNAgent


def train_dqn_agent(replay_buffer: np.array, config: dict, params: dict) -> DQNAgent:
    # Init agents
    agent = DQNAgent(**params)
    target_agent = DQNAgent(**params).clone_from(agent)

    opt = optim.Adam(params=agent.parameters())

    for i in tqdm(range(config.get("N_EPISODES"))):
        # Sample batch
        batch = np.array(random.sample(replay_buffer, config.get("MINIBATCH_SIZE")))

        # Extract and tensor-ize observations s and s*
        obs = torch.tensor(np.stack(batch[:, 0]), dtype=torch.float)
        obs = obs.reshape(config.get("MINIBATCH_SIZE"), -1)
        next_obs = torch.tensor(np.stack(batch[:, 3]), dtype=torch.float)
        next_obs = next_obs.reshape(config.get("MINIBATCH_SIZE"), -1)

        # Estimate q and q*
        q_values = agent.forward(obs)
        next_q_values = target_agent.forward(next_obs)

        # Compute q'
        done_idx = torch.tensor(np.stack(batch[:, 4]), dtype=torch.bool)
        rewards = torch.tensor(np.stack(batch[:, 2]), dtype=torch.float)
        q_value_mask = torch.zeros(next_q_values.shape[0])
        q_value_mask[~done_idx] = next_q_values.max(axis=1).values[~done_idx]
        q_prime = rewards + config.get("DISCOUNT") * q_value_mask

        # Apply q' to corresponding actions
        actions = torch.tensor(np.stack(batch[:, 1]), dtype=torch.long)
        q_values_truth = torch.clone(q_values)
        q_values_truth[range(config.get("MINIBATCH_SIZE")), actions] = q_prime

        # Update agent every step, and target_agent every TRG_FREQ steps
        opt.zero_grad()
        loss = nn.MSELoss()(q_values, q_values_truth)
        loss.backward()
        opt.step()
        if i % config.get("TRG_FREQ") == 0:
            target_agent = DQNAgent(**params).clone_from(agent)

        # Some logging
        if i % config.get("PRINT_FREQ") == 0:
            print("\n")
            print(f"Episode {i}: Agent loss: {loss}")
            print(f"target_agent updated")

    return agent


def play_match(agent1: Agent, agent2: Agent, n_rounds: int = 3):
    agents = itertools.cycle([agent1, agent2])

    for _ in range(n_rounds):
        env = connect_four_v3.env()
        env.reset()

        board = Board()

        for agent in agents:
            obs, _, _, _ = env.last()

            if type(agent) is DQNAgent:
                obs = torch.tensor(obs["observation"], dtype=torch.float).sum(axis=2).flatten()[None, :]
                action = int(agent.forward(obs).argmax())
            else:
                action = agent.forward(obs)

            env.step(action)
            next_obs, reward, next_done, _ = env.last()

            agent_id = 0 if agent == agent1 else 1
            board.add(action, agent_id=agent_id)
            # print(board)
            if next_done:
                break

        print(f"Agent {agent_id} got a reward of {reward}")


class Board:
    def __init__(self, rows: int = 6, cols: int = 7):
        self.state = np.zeros((rows, cols), dtype=int)

    def add(self, col: int, agent_id: int):
        row = self.state.shape[0] - (self.state[:, col] != 0).sum() - 1
        self.state[row, col] = agent_id + 1

    def __str__(self):
        txt = ""
        for j in range(self.state.shape[0]):
            for i in range(self.state.shape[1]):
                txt += str(self.state[j, i]) + ' '
            txt += "\n"
        txt += "\n"
        return txt