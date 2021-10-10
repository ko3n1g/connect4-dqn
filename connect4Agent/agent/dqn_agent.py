from connect4Agent.agent.agent import Agent
import torch.nn as nn
import torch


class DQNAgent(Agent, nn.Module):
    def __init__(self, in_shape: int, out_shape: int, hidden_shape: int):
        super(DQNAgent, self).__init__()

        self.l1 = nn.Linear(in_shape, hidden_shape)
        self.l2 = nn.Linear(hidden_shape, hidden_shape)
        self.l3 = nn.Linear(hidden_shape, out_shape)

    def forward(self, observation: np.array) -> np.array:
        observation = nn.ReLU()(self.l1(observation))
        observation = nn.ReLU()(self.l2(observation))
        return self.l3(observation)

    def clone_from(self, agent: "DQNAgent") -> "DQNAgent":
        for name, param in agent.state_dict().items():
            if name not in self.state_dict():
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            self.state_dict()[name].copy_(param)
        return self

    def save_model(self, file_path: str = "model/dqn_agent.torch"):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path: str = "model/dqn_agent.torch") -> "DQNAgent":
        self.load_state_dict(torch.load(file_path))
        self.eval()
        return self
