from abc import ABC, abstractmethod

import torch
from torch import nn


class QuantMethod(ABC):

    @abstractmethod
    def create_weights(self, layer: nn.Module, input_size: int, output_size: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        raise NotImplementedError
