from connect4Agent.agent.agent import Agent


class ManualAgent(Agent):
    def forward(self, observation: np.array) -> Union[int, np.array]:
        valid_actions = np.argwhere(observation["action_mask"] == 1).flatten()
        choice = -1
        while choice not in valid_actions:
            choice = input(f"Select an action out of {valid_actions}")
            try:
                choice = int(choice)
            except Exception:
                pass

        return int(choice)