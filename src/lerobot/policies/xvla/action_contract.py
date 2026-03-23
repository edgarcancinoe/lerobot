from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

SO101_EEF_FEATURE_NAMES = (
    "x",
    "y",
    "z",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
    "gripper",
)

SO101_JOINT_FEATURE_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True)
class SO101ActionSliceSpec:
    action_mode: str
    start: int
    end: int
    real_dim: int
    gripper_idx: int
    names: tuple[str, ...]

    def slice_tensor(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        return x[..., self.start : self.end]

    def feature_names(self, suffix: str = "") -> tuple[str, ...]:
        if not suffix:
            return self.names
        return tuple(f"{name}{suffix}" for name in self.names)


SO101_ACTION_SPECS: dict[str, SO101ActionSliceSpec] = {
    "so101_ee6d": SO101ActionSliceSpec(
        action_mode="so101_ee6d",
        start=0,
        end=10,
        real_dim=10,
        gripper_idx=9,
        names=SO101_EEF_FEATURE_NAMES,
    ),
    "so101_joint": SO101ActionSliceSpec(
        action_mode="so101_joint",
        start=10,
        end=16,
        real_dim=6,
        gripper_idx=5,
        names=SO101_JOINT_FEATURE_NAMES,
    ),
}


def get_so101_slice_spec(action_mode: str | None) -> SO101ActionSliceSpec | None:
    if not action_mode:
        return None
    return SO101_ACTION_SPECS.get(action_mode.lower())


def build_slice_map(spec: SO101ActionSliceSpec, keys: tuple[str, ...] = ("action", "observation.state")) -> dict[str, tuple[int, int]]:
    return {key: (spec.start, spec.end) for key in keys}


def slice_feature_spec(feature_spec: dict[str, Any], spec: SO101ActionSliceSpec, suffix: str = "") -> dict[str, Any]:
    sliced = deepcopy(feature_spec)
    if "shape" in sliced:
        shape = list(sliced["shape"])
        if shape:
            shape[-1] = spec.real_dim
            sliced["shape"] = tuple(shape) if isinstance(feature_spec["shape"], tuple) else shape
    if "names" in sliced:
        sliced["names"] = list(spec.feature_names(suffix=suffix))
    return sliced


def slice_dataset_meta_in_place(dataset_meta: Any, spec: SO101ActionSliceSpec) -> None:
    for key in ("action", "observation.state"):
        if key in dataset_meta.features:
            dataset_meta.features[key] = slice_feature_spec(dataset_meta.features[key], spec)

        if key in dataset_meta.stats:
            for stat_name, stat_value in list(dataset_meta.stats[key].items()):
                dataset_meta.stats[key][stat_name] = stat_value[spec.start : spec.end]
