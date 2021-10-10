import abc
import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn


class Agent:
    @abc.abstractmethod
    def forward(self, observation: np.array) -> Union[int, np.array]:
        pass


