import itertools
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.classic import connect_four_v3
from tqdm.cli import tqdm

from connect4Agent.agent import Agent, DQNAgent


def transform_obs(obs: np.array) -> np.array:
    match_a1 = np.where(obs[:, :, 0] == 1)
    match_a2 = np.where(obs[:, :, 1] == 1)

    if match_a1[0].shape[0] > 0:
        obs[tuple((*match_a1, 0))] = 2
    if match_a2[0].shape[0] > 0:
        obs[tuple((*match_a2, 1))] = 3

    return obs


def train_dqn_agent(replay_buffer: np.array, config: dict, params: dict) -> DQNAgent:
    # Init agents
    agent = DQNAgent(**params)
    target_agent = DQNAgent(**params).clone_from(agent)

    opt = optim.Adam(params=agent.parameters())
    losses = []

    for i in tqdm(range(config.get("N_EPISODES"))):
        # Sample and extract from batch
        batch = np.array(random.sample(replay_buffer, config.get("MINIBATCH_SIZE")))
        obs = torch.tensor(np.stack(batch[:, 0]), dtype=torch.float)
        obs = obs.reshape(config.get("MINIBATCH_SIZE"), -1)
        actions = torch.tensor(np.stack(batch[:, 1]), dtype=torch.long)
        rewards = torch.tensor(np.stack(batch[:, 2]), dtype=torch.float)
        next_obs = torch.tensor(np.stack(batch[:, 3]), dtype=torch.float)
        next_obs = next_obs.reshape(config.get("MINIBATCH_SIZE"), -1)
        dones = torch.tensor(np.stack(batch[:, 4]), dtype=torch.long)

        # Estimate q and q*
        q_values_curr = agent.forward(obs)
        q_values_next = target_agent.forward(next_obs)

        # Compute q'
        q_values_next_max = torch.max(q_values_next, dim=1).values
        q_value_expct = rewards + config.get("DISCOUNT") * q_values_next_max * (1 - dones)

        # Extract the action chose
        q_values_curr = q_values_curr.gather(1, actions.view(64, 1))
        q_value_expct = q_value_expct.view(-1, 1)

        # Update agent every step, and target_agent every TRG_FREQ steps
        opt.zero_grad()
        loss = nn.MSELoss()(q_values_curr, q_value_expct)
        loss.backward()
        losses.append(loss.detach().numpy())
        opt.step()
        if i % config.get("TRG_FREQ") == 0:
            target_agent = DQNAgent(**params).clone_from(agent)

        # Some logging
        if i % config.get("PRINT_FREQ") == 0:
            print("\n")
            print(
                f"Episode {i}: Agent loss: {np.array(losses)[config.get('PRINT_FREQ'):].mean()}"
            )
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
                observation = torch.tensor(
                    transform_obs(obs["observation"]), dtype=torch.float
                ).flatten()[None, :]
                action = int(agent.forward(observation).argmax())
            else:
                action = agent.forward(obs)

            env.step(action)
            next_obs, reward, next_done, _ = env.last()

            agent_id = 0 if agent == agent1 else 1
            board.add(action, agent_id=agent_id)
            print(board)
            if next_done:
                break

        print(f"Agent {agent_id} got a reward of {reward}")
        if not obs["action_mask"][action]:
            print(board)


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
                txt += str(self.state[j, i]) + " "
            txt += "\n"
        txt += "\n"
        return txt
