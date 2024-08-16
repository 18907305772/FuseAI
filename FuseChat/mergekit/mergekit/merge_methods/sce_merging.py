# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel
from typing_extensions import Literal

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods.base import ConfigParameterDef, MergeMethod
from mergekit.sparsify import SparsificationMethod, sparsify


class SCEMerge(MergeMethod, BaseModel, frozen=True):

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="select_topk", required=False, default_value=1.0),
        ]

    def make_task(
            self,
            output_weight: WeightInfo,
            tensors: GatherTensors,
            base_model: Optional[ModelReference],
            parameters: ImmutableMap[str, Any],
            **_kwargs,
    ) -> Task:
        return SCETask(
            tensors=tensors,
            base_model=base_model,
            out_tensor_name=output_weight.name,
            select_topk=parameters["select_topk"],
        )


class SCETask(Task[torch.Tensor]):
    tensors: GatherTensors
    base_model: ModelReference
    out_tensor_name: str
    select_topk: float

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.tensors}

    def execute(
            self,
            tensors: Dict[ModelReference, torch.Tensor],
            **_kwargs,
    ) -> torch.Tensor:
        # collect task vectors
        tvs, base = get_task_vectors(
            self.out_tensor_name,
            self.base_model,
            tensors
        )
        if not tvs:
            return base

        deltas_list = [tv["delta"] for tv in tvs]
        deltas = torch.stack(deltas_list, dim=0)

        # select & calculate
        weights = get_sce_weight(deltas, self.select_topk)
        weights = torch.tensor(
            weights, dtype=deltas.dtype, device=deltas.device
        )
        while len(deltas.shape) > len(weights.shape):
            weights.unsqueeze_(-1)

        # erase
        mask_dtype = base.dtype
        erase_mask = get_erase_mask(
            deltas,
            mask_dtype=mask_dtype,
        )
        erased_weights = weights * erase_mask
        mixed_delta = (deltas * erased_weights).sum(dim=0)

        # normalize
        divisor = (erased_weights).sum(dim=0)
        divisor[divisor == 0] = 1
        mixed_delta /= divisor

        return (base + mixed_delta).to(base.dtype)


def get_task_vectors(
        parameter_name: str,
        base_model: ModelReference,
        tensors: ImmutableMap[ModelReference, torch.Tensor]
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    keys = list(tensors.keys())
    base = tensors[base_model]

    res = []
    for model in keys:
        if model == base_model:
            continue

        x = tensors[model].to(base.dtype)
        if x.shape != base.shape:
            if "lm_head" in parameter_name or "embed_tokens" in parameter_name:
                x = x[: base.shape[0], : base.shape[1]]
                logging.warning(f"Using submatrix of {model}:{parameter_name}")
            else:
                logging.warning(
                    f"skipping {model}:{parameter_name} due to size mismatch"
                )
                continue

        delta = x - base
        del x
        del tensors[model]

        d = {}
        d["model"] = model
        d["delta"] = delta
        res.append(d)
    return res, base


def get_erase_mask(
        delta: torch.Tensor,
        mask_dtype: Optional[torch.dtype] = None,
):
    """Returns a mask determining which delta vectors should be merged
    into the final model.
    """
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    sign_weight = delta.sum(dim=0)
    majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
    del sign_weight

    return sign == majority_sign

def get_sce_mask(deltas, density):
    variance = torch.var(deltas, dim=0, unbiased=False, keepdim=False)
    non_zero_positions_count = torch.count_nonzero(variance)
    k = int(density * non_zero_positions_count)
    mask = torch.zeros_like(variance)
    if k == 0:
        return mask
    assert k > 0, "not gonna zero out the whole tensor buddy"
    variance_abs = variance.abs().view(-1)
    topk_vari = torch.topk(variance_abs, k=k, largest=True)
    mask.view(-1)[topk_vari.indices] = 1
    return mask


def get_sce_weight(deltas, density):
    # select
    if density < 1:
        mask = get_sce_mask(deltas, density)
        deltas = [delta * mask for delta in deltas]
    # calculate
    weights = [torch.sum(delta ** 2).item() / delta.numel() for delta in deltas]
    sum_weights = sum(weights)
    if sum_weights == 0:
        weights = [1.0 / len(weights)] * len(weights)
    else:
        weights = [item / sum_weights for item in weights]
    return weights
