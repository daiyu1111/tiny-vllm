from abc import ABC, abstractmethod

import torch
from torch import nn


class QuantMethod(ABC):

    @abstractmethod
    def create_weights(self, layer: nn.Module, input_size: int, output_size: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        x: torch.Tensor,
        layer: nn.Module,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        raise NotImplementedError
