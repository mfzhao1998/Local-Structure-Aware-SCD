# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from typing import Union

import torch
from torch import Tensor, nn


class LayerScale(nn.Module):
    """Layer scaling used by the retained SAM 3 vision and text encoders."""

    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
