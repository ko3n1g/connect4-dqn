from connect4Agent.agent.agent import Agent
import numpy as np
import random


class RandomAgent(Agent):
    def forward(self, observation: np.array) -> int:
        idx = np.argwhere(observation["action_mask"] == 1).flatten().tolist()
        return random.choice(idx)