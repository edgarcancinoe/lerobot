from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="slice_processor")
class SliceProcessorStep(ProcessorStep):
    """
    A processor step that slices specific features in a transition.
    
    This is useful when the environment provides a larger vector (e.g., 16D with EEF, joints,
    and gripper) but the policy only expects a subset (e.g., 10D EEF or 6D joints).
    """

    # Mapping of feature key (e.g., "action", "observation.state") to (start_idx, end_idx)
    slice_map: dict[str, tuple[int, int | None]]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        
        # Slice Action
        if "action" in self.slice_map:
            action = new_transition.get(TransitionKey.ACTION)
            if action is not None:
                start, end = self.slice_map["action"]
                new_transition[TransitionKey.ACTION] = action[..., start:end]
        
        # Slice Observations (e.g., observation.state)
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_obs = dict(observation)
            for key, (start, end) in self.slice_map.items():
                if key != "action" and key in new_obs:
                    new_obs[key] = new_obs[key][..., start:end]
            new_transition[TransitionKey.OBSERVATION] = new_obs
            
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # The output features will have a reduced shape for the sliced vectors.
        # But usually `features` here are the output of the previous step.
        return features

    def get_config(self) -> dict[str, Any]:
        return {"slice_map": self.slice_map}
