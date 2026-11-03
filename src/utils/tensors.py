"""Tensor utilities: NestedTensor, collate_fn, and helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import Any, cast, overload

import torch
import torchvision
from torch import Tensor


class NestedTensor:
    """Batch of tensors with variable spatial sizes, padded to a common size.

    Stores both the padded tensor and a boolean mask indicating padding positions.
    """

    def __init__(
        self, tensors: Tensor, mask: Tensor | None, no_padding: bool = False
    ) -> None:
        self.tensors = tensors
        self.mask = mask
        # Structural claim, not a claim about ``mask``'s contents: True only when the batch was
        # assembled from images that already filled the padded extent, so ``mask`` is all-False.
        # Consumers use it to treat mask-derived values as constants of the batch shape without
        # reading device memory (which would force a device-to-host sync). Defaults to False so
        # every other construction site keeps the conservative behaviour.
        self.no_padding = no_padding  # 这批图像没有任何一张需要被 padding

    def to(self, device: torch.device, **kwargs: Any) -> "NestedTensor":
        """Move tensors and mask to *device*.

        Args:
            device: Target device. **kwargs: Additional arguments forwarded to ``Tensor.to``.

        Returns:
            New NestedTensor on *device*.
        """
        cast_tensor = self.tensors.to(device, **kwargs)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, **kwargs)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask, self.no_padding)

    def pin_memory(self) -> "NestedTensor":
        """Pin tensor and mask memory for faster CPU→GPU transfer.

        Returns:
            New NestedTensor with pinned memory.
        """
        return NestedTensor(
            self.tensors.pin_memory(),
            self.mask.pin_memory() if self.mask is not None else None,
            self.no_padding,
        )

    def decompose(self) -> tuple[Tensor, Tensor | None]:
        """Return ``(tensors, mask)`` tuple.

        Returns:
            Tuple of the padded tensor and the boolean mask.
        """
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)
