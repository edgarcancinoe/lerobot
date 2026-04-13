#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import logging
import warnings
import os

from lerobot.policies import xvla

# Suppress Hugging Face transformers warnings related to Florence2 and GenerationMixin
os.environ["ACCELERATE_LOG_LEVEL"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", module=".*transformers.*")
warnings.filterwarnings("ignore", module=".*accelerate.*")
warnings.filterwarnings("ignore", message=".*Florence2ForConditionalGeneration has generative capabilities.*")
warnings.filterwarnings("ignore", message=".*Detected kernel version.*")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)

import time
from contextlib import nullcontext
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle, dataset_to_policy_features
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.xvla.action_contract import get_so101_slice_spec, slice_dataset_meta_in_place
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
import torchvision.utils as vutils
from lerobot.utils.train_utils import get_step_checkpoint_dir, get_step_identifier, load_training_state, save_checkpoint, update_last_checkpoint
from lerobot.utils.utils import format_big_number, has_method, init_logging
from lerobot.utils.constants import OBS_IMAGES

XVLA_EXPECTED_TOKENIZER_MAX_LENGTH = 64
XVLA_EXPECTED_NUM_IMAGE_VIEWS = 3
XVLA_EXPECTED_EMPTY_CAMERAS = 1
XVLA_EXPECTED_MAX_LEN_SEQ = 1024
XVLA_EXPECTED_IMAGE_KEYS = (
    f"{OBS_IMAGES}.image",
    f"{OBS_IMAGES}.image2",
    f"{OBS_IMAGES}.empty_camera_0",
)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"

    total_seconds = max(0, int(round(seconds)))
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)

    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _build_training_progress_postfix(
    step: int,
    total_steps: int,
    elapsed_s: float,
    start_step: int,
    loss: float | None = None,
) -> dict[str, str]:
    completed_steps = max(step - start_step, 0)
    avg_step_s = (elapsed_s / completed_steps) if completed_steps > 0 else None
    remaining_steps = max(total_steps - step, 0)
    remaining_s = (avg_step_s * remaining_steps) if avg_step_s is not None else None
    completion_ts = (time.time() + remaining_s) if remaining_s is not None else None

    postfix = {
        "avg_step": _format_duration(avg_step_s),
        "remaining": _format_duration(remaining_s),
        "done_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(completion_ts)) if completion_ts else "?",
    }
    if loss is not None:
        postfix["loss"] = f"{loss:.4f}"

    return postfix


def _rename_policy_feature_key(key: str, rename_map: dict[str, str] | None) -> str:
    if not rename_map:
        return key
    return rename_map.get(key, key)


def _enforce_xvla_finetune_contract(policy_cfg) -> None:
    policy_cfg.tokenizer_max_length = XVLA_EXPECTED_TOKENIZER_MAX_LENGTH
    policy_cfg.num_image_views = XVLA_EXPECTED_NUM_IMAGE_VIEWS
    policy_cfg.empty_cameras = XVLA_EXPECTED_EMPTY_CAMERAS
    policy_cfg.max_len_seq = XVLA_EXPECTED_MAX_LEN_SEQ


def _rebuild_xvla_visual_input_features(policy_cfg, dataset_meta, rename_map: dict[str, str] | None) -> None:
    dataset_policy_features = dataset_to_policy_features(dataset_meta.features)
    renamed_visual_features: dict[str, PolicyFeature] = {}
    for key, feature in dataset_policy_features.items():
        if feature.type is not FeatureType.VISUAL:
            continue
        renamed_key = _rename_policy_feature_key(key, rename_map)
        renamed_visual_features[renamed_key] = PolicyFeature(type=FeatureType.VISUAL, shape=feature.shape)

    expected_real_views = policy_cfg.num_image_views - policy_cfg.empty_cameras
    if len(renamed_visual_features) != expected_real_views:
        raise ValueError(
            "XVLA finetuning expects exactly "
            f"{expected_real_views} real camera views after rename-map application, but found "
            f"{len(renamed_visual_features)} visual inputs: {list(renamed_visual_features.keys())}."
        )

    non_visual_features = {
        key: feature
        for key, feature in policy_cfg.input_features.items()
        if feature.type is not FeatureType.VISUAL
    }

    rebuilt_input_features: dict[str, PolicyFeature] = dict(non_visual_features)
    rebuilt_input_features.update(renamed_visual_features)

    if policy_cfg.resize_imgs_with_padding is not None:
        height, width = policy_cfg.resize_imgs_with_padding
        empty_shape = (3, height, width)
    else:
        first_visual_shape = next(iter(renamed_visual_features.values())).shape
        empty_shape = first_visual_shape

    for idx in range(policy_cfg.empty_cameras):
        rebuilt_input_features[f"{OBS_IMAGES}.empty_camera_{idx}"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=empty_shape,
        )

    policy_cfg.input_features = rebuilt_input_features


def _assert_xvla_finetune_contract(policy_cfg) -> None:
    actual_image_keys = tuple(policy_cfg.image_features.keys())
    if policy_cfg.tokenizer_max_length != XVLA_EXPECTED_TOKENIZER_MAX_LENGTH:
        raise ValueError(
            f"XVLA checkpoint config drifted: tokenizer_max_length={policy_cfg.tokenizer_max_length}, "
            f"expected {XVLA_EXPECTED_TOKENIZER_MAX_LENGTH}."
        )
    if policy_cfg.num_image_views != XVLA_EXPECTED_NUM_IMAGE_VIEWS:
        raise ValueError(
            f"XVLA checkpoint config drifted: num_image_views={policy_cfg.num_image_views}, "
            f"expected {XVLA_EXPECTED_NUM_IMAGE_VIEWS}."
        )
    if policy_cfg.empty_cameras != XVLA_EXPECTED_EMPTY_CAMERAS:
        raise ValueError(
            f"XVLA checkpoint config drifted: empty_cameras={policy_cfg.empty_cameras}, "
            f"expected {XVLA_EXPECTED_EMPTY_CAMERAS}."
        )
    if policy_cfg.max_len_seq != XVLA_EXPECTED_MAX_LEN_SEQ:
        raise ValueError(
            f"XVLA checkpoint config drifted: max_len_seq={policy_cfg.max_len_seq}, "
            f"expected {XVLA_EXPECTED_MAX_LEN_SEQ}."
        )
    if actual_image_keys != XVLA_EXPECTED_IMAGE_KEYS:
        raise ValueError(
            f"XVLA visual schema drifted: actual={list(actual_image_keys)}, "
            f"expected={list(XVLA_EXPECTED_IMAGE_KEYS)}."
        )


def _validate_xvla_sequence_budget(policy, dataset, preprocessor) -> None:
    sample_batch = default_collate([dataset[0]])
    processed_batch = preprocessor(sample_batch)

    with torch.no_grad():
        inputs = policy._build_model_inputs(processed_batch)
        enc = policy.model.forward_vlm(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["image_input"],
            image_mask=inputs["image_mask"],
        )

    seq_len = (
        policy.config.chunk_size
        + enc["vlm_features"].shape[1]
        + enc["aux_visual_inputs"].shape[1]
    )
    max_len_seq = policy.model.transformer.pos_emb.shape[1]
    if seq_len > max_len_seq:
        raise ValueError(
            "XVLA multimodal sequence exceeds the configured transformer budget. "
            f"image_keys={list(policy.config.image_features.keys())}, "
            f"total_views={policy.config.num_image_views}, "
            f"tokenizer_max_length={policy.config.tokenizer_max_length}, "
            f"chunk_size={policy.config.chunk_size}, "
            f"max_len_seq={policy.config.max_len_seq}, "
            f"measured_seq_len={seq_len}."
        )

    logging.info(
        "XVLA sequence budget validated: image_keys=%s total_views=%s tokenizer_max_length=%s "
        "chunk_size=%s measured_seq_len=%s max_len_seq=%s",
        list(policy.config.image_features.keys()),
        policy.config.num_image_views,
        policy.config.tokenizer_max_length,
        policy.config.chunk_size,
        seq_len,
        max_len_seq,
    )


def _patch_xvla_gripper_stats_for_overrides(policy_cfg, dataset_stats):
    from copy import deepcopy

    if policy_cfg.type != "xvla" or not dataset_stats:
        return dataset_stats

    slice_spec = get_so101_slice_spec(getattr(policy_cfg, "action_mode", ""))
    if slice_spec is None or "action" not in dataset_stats:
        return dataset_stats

    patched_stats = {}
    for key, value in dataset_stats.items():
        if isinstance(value, dict):
            patched_stats[key] = {
                stat_name: stat_value.clone() if isinstance(stat_value, torch.Tensor) else deepcopy(stat_value)
                for stat_name, stat_value in value.items()
            }
        else:
            patched_stats[key] = deepcopy(value)

    action_stats = patched_stats["action"]
    for stat_name, target_value in (("mean", 0.0), ("std", 1.0)):
        if stat_name in action_stats and len(action_stats[stat_name]) > slice_spec.gripper_idx:
            action_stats[stat_name][slice_spec.gripper_idx] = target_value

    return patched_stats


def debug_batch(batch, tag="", step=0, only_step=0, slice_dim=None, dataset_meta=None):
    """Call this at any point in the pipeline to inspect tensors."""
    if step != only_step:
        return
    
    logging.info(colored(f"\n{'='*70}", "magenta", attrs=["bold"]))
    logging.info(colored(f"[DEBUG] {tag} | step={step}", "magenta", attrs=["bold"]))
    logging.info(colored(f"{'='*70}", "magenta", attrs=["bold"]))
    
    if not batch:
        logging.info(colored("  (Empty dictionary)", "red"))
        logging.info("")
        return

    # Keys to ignore based on user request
    ignore_keys = {"index", "info", "episode_index"}

    # Custom sort order: action -> observation.state -> action_is_pad -> others
    def sort_key(k):
        if k == "action": return (0, k)
        if k == "observation.state": return (1, k)
        if k == "action_is_pad": return (2, k)
        return (3, k)

    sorted_keys = sorted([k for k in batch.keys() if k not in ignore_keys], key=sort_key)

    for k in sorted_keys:
        v = batch[k]
        padded_key = f"  {k}:".ljust(40)
        key_str = colored(padded_key, "cyan")
        
        if isinstance(v, torch.Tensor):
            shape_str = f"shape={list(v.shape)} dtype={v.dtype}"
            
            # For specific kinematics tensors we want detailed dim-wise stats
            if k in ["action", "observation.state", "pred_action"] and v.numel() > 0 and v.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                logging.info(f"{key_str} {shape_str}")
                
                # Fetch name labels like in log_processor_stats
                # Determine how many dimensions the tensor's last axis has
                num_dims = v.size(-1)
                
                feat_names = None
                meta_k = "action" if k == "pred_action" else k
                if dataset_meta and meta_k in dataset_meta.features:
                    feature_info = dataset_meta.features[meta_k]
                    if isinstance(feature_info, dict):
                        feat_names = feature_info.get("names")
                    elif hasattr(feature_info, "names"):
                        feat_names = feature_info.names
                
                name_width = 15
                header_str = f"    {'Dim Name':<{name_width}} | {'min':<8} | {'max':<8} | {'mean':<8}"
                logging.info(colored(f"    {'-' * (len(header_str) - 4)}", "green"))
                logging.info(colored(header_str, "green", attrs=["bold"]))
                logging.info(colored(f"    {'-' * (len(header_str) - 4)}", "green"))
                
                # We calculate stats across batch & sequence (all except the last dim)
                # Reshape to (-1, num_dims)
                v_flat = v.reshape(-1, num_dims).float()
                v_min = v_flat.min(dim=0).values
                v_max = v_flat.max(dim=0).values
                v_mean = v_flat.mean(dim=0)
                
                separator_printed = False
                for i in range(num_dims):
                    dim_name = f"Dim {i}"
                    
                    # If this is pred_action and we exceed the true action space defined by dataset_meta, treat as padding
                    is_padding = False
                    if slice_dim is not None and i >= slice_dim:
                        is_padding = True
                    elif feat_names is not None:
                        if i < len(feat_names):
                            d_name = feat_names[i]
                            if isinstance(d_name, dict) and 'name' in d_name:
                                dim_name = d_name['name']
                            elif isinstance(d_name, str):
                                dim_name = d_name
                        else:
                            is_padding = True
                    
                    if "pad" in dim_name.lower():
                        is_padding = True
                    
                    if len(dim_name) > name_width:
                        dim_name = dim_name[:name_width-2] + ".."
                        
                    # Print a visually distinct separator if we transition into padding territory
                    if not separator_printed and is_padding:
                        logging.info(colored(f"    {' ':<{name_width}} | {'--- padding ---':^28}", "cyan", attrs=["bold"]))
                        separator_printed = True
                        
                    row_str = f"    {dim_name:<{name_width}} | {v_min[i].item():>8.4f} | {v_max[i].item():>8.4f} | {v_mean[i].item():>8.4f}"
                    logging.info(colored(row_str, "green"))
                logging.info(colored(f"    {'-' * (len(header_str) - 4)}", "green"))
                
            else:
                extra = ""
                # Overall stats for images or other numeric tensors
                if v.numel() > 0 and v.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    extra = f" | min={v.min().item():.4f} max={v.max().item():.4f} mean={v.mean().item():.4f}"
                elif v.numel() == 0:
                    extra = " | (empty tensor)"
                
                logging.info(f"{key_str} {shape_str}{extra}")
        else:
            val_str = str(v)
            if len(val_str) > 60:
                val_str = val_str[:57] + "..."
            logging.info(f"{key_str} type={type(v).__name__} value={val_str}")
            
    logging.info("")

def log_processor_stats(step, dataset_meta=None):
    """Log the normalizer processor stats (min, max, mean, std, count)."""
    if hasattr(step, "stats") and step.stats:
        logging.info(colored(f"\n{'='*70}", "cyan", attrs=["bold"]))
        logging.info(colored(f"  Normalization Stats [{type(step).__name__}]", "cyan", attrs=["bold"]))
        logging.info(colored(f"{'='*70}", "cyan", attrs=["bold"]))
        for feat_key, feat_stats in step.stats.items():
            if feat_key not in ["action", "observation.state"]:
                continue
            logging.info(colored(f"\n  Feature: {feat_key}", "green", attrs=["bold"]))
            
            wanted_stats = ["min", "max", "mean", "std"]
            available_stats = [k for k in wanted_stats if k in feat_stats]
            
            # Extract count separately
            count_val = feat_stats.get("count", "N/A")
            if hasattr(count_val, "tolist"): count_val = count_val.tolist()
            if isinstance(count_val, (list, tuple)) and len(count_val) >= 1: count_val = count_val[0]
            logging.info(colored(f"  Count:   {count_val}", "green"))
            
            if not available_stats:
                continue
                
            # Attempt to find feature names from dataset_meta
            feat_names = None
            if dataset_meta and feat_key in dataset_meta.features:
                feature_info = dataset_meta.features[feat_key]
                if isinstance(feature_info, dict):
                    feat_names = feature_info.get("names")
                elif hasattr(feature_info, "names"):
                    feat_names = feature_info.names

            first_stat = feat_stats[available_stats[0]]
            if hasattr(first_stat, "tolist"): first_stat = first_stat.tolist()
            if not isinstance(first_stat, (list, tuple)): first_stat = [first_stat]
            num_dims = len(first_stat)

            name_width = 15
            header_str = f"  {'Dim Name':<{name_width}}" + "".join([f" | {s:<8}" for s in available_stats])
            logging.info(colored(f"  {'-' * (len(header_str) - 2)}", "green"))
            logging.info(colored(header_str, "green", attrs=["bold"]))
            logging.info(colored(f"  {'-' * (len(header_str) - 2)}", "green"))
            for i in range(num_dims):
                dim_name = f"Dim {i}"
                if feat_names is not None and i < len(feat_names):
                    d_name = feat_names[i]
                    if isinstance(d_name, dict) and 'name' in d_name:
                        dim_name = d_name['name']
                    elif isinstance(d_name, str):
                        dim_name = d_name

                if len(dim_name) > name_width:
                    dim_name = dim_name[:name_width-2] + ".."
                
                row_str = f"  {dim_name:<{name_width}}"
                for stat_k in available_stats:
                    val = feat_stats[stat_k]
                    if hasattr(val, "tolist"): val = val.tolist()
                    if isinstance(val, (list, tuple)):
                        v = val[i] if i < len(val) else None
                    else:
                        v = val if i == 0 else None
                    
                    if v is not None:
                        if isinstance(v, float):
                            row_str += f" | {v:>8.4f}"
                        else:
                            row_str += f" | {str(v)[:8]:>8}"
                    else:
                        row_str += f" | {'-':>8}"
                logging.info(colored(row_str, "green"))
                
            logging.info(colored(f"  {'-' * (len(header_str) - 2)}", "green"))
        logging.info(colored(f"{'='*70}", "cyan", attrs=["bold"]))

def save_debug_images(batch, output_dir, step=0, prefix="post"):
    """Save the first frame of the first batch for all image modalities to visualize what the VLM sees."""
    if step != 0:
        return
        
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Find all keys that look like images
    image_keys = [k for k in batch.keys() if "images" in k and isinstance(batch[k], torch.Tensor)]
    if not image_keys:
        return
        
    logging.info(colored(f"\n[DEBUG] Saving {prefix} visualization grids to: {vis_dir}", "yellow", attrs=["bold"]))
    for k in image_keys:
        img_tensor = batch[k]
        if img_tensor.ndim >= 4:  # [B, T, C, H, W] or [B, C, H, W]
            # Take the first sequence step of the first batch element
            if img_tensor.ndim == 5:
                img = img_tensor[0, 0]  # [C, H, W]
            else:
                img = img_tensor[0]     # [C, H, W]
            
            # Convert to float32 for processing
            img = img.float()
            
            # Smart normalization:
            # If the values are already in [0, 1] (or close), just use them.
            # If they are in [0, 255], scale them.
            # Otherwise (standardized), undo ImageNet normalization if 3 channels.
            # Format filename safely to use in logging
            safe_k = k.replace(".", "_")
            img_min = img.min()
            img_max = img.max()
            if img_min >= -0.01 and img_max <= 1.01:
                # Already mostly in [0, 1]
                logging.info(f"    [{prefix}_{safe_k}] Using identity normalization (values in [0, 1])")
                img_normalized = img
            elif img_min >= -0.1 and img_max > 5.0 and img_max <= 256.0:
                # Likely [0, 255]
                logging.info(f"    [{prefix}_{safe_k}] Using /255 normalization (values in [0, 255])")
                img_normalized = img / 255.0
            elif img_max > img_min:
                if img.shape[0] == 3:
                    # Likely standardized. Inverse ImageNet normalization
                    logging.info(f"    [{prefix}_{safe_k}] Using inverse ImageNet normalization")
                    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
                    img_normalized = img * std + mean
                    img_normalized = torch.clamp(img_normalized, 0.0, 1.0)
                else:
                    # Fallback to min-max stretch if not 3 channels
                    logging.info(f"    [{prefix}_{safe_k}] Using min-max stretch fallback (not 3 channels)")
                    img_normalized = (img - img_min) / (img_max - img_min)
            else:
                logging.info(f"    [{prefix}_{safe_k}] Using fallback identity (flat image)")
                img_normalized = img
            
            out_path = os.path.join(vis_dir, f"step_{step}_{prefix}_{safe_k}.jpg")
            try:
                vutils.save_image(img_normalized, out_path)
                logging.info(f"  Saved {out_path}")
            except Exception as e:
                logging.warning(f"  Failed to save {out_path}: {e}")
    logging.info("")


GRIPPER_DEBUG_COUNT_KEYS = (
    "target_zero_count",
    "target_one_count",
    "pred_zero_count",
    "pred_one_count",
    "true_negative_count",
    "true_positive_count",
    "false_positive_count",
    "false_negative_count",
)


@dataclasses.dataclass
class GripperDebugWindow:
    start_step: int
    end_step: int = 0
    counts: dict[str, int] = dataclasses.field(default_factory=lambda: {key: 0 for key in GRIPPER_DEBUG_COUNT_KEYS})

    def update(self, step: int, count_dict: dict[str, Any]) -> None:
        self.end_step = step
        for key in GRIPPER_DEBUG_COUNT_KEYS:
            value = count_dict[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            self.counts[key] += int(value)

    def reset(self, next_start_step: int) -> None:
        self.start_step = next_start_step
        self.end_step = next_start_step - 1
        self.counts = {key: 0 for key in GRIPPER_DEBUG_COUNT_KEYS}

    def as_reduced_dict(self, accelerator: Accelerator) -> dict[str, float]:
        device = accelerator.device
        local_counts = torch.tensor([self.counts[key] for key in GRIPPER_DEBUG_COUNT_KEYS], device=device, dtype=torch.float32)
        reduced_counts = accelerator.reduce(local_counts, reduction="sum")
        reduced = {key: float(reduced_counts[idx].item()) for idx, key in enumerate(GRIPPER_DEBUG_COUNT_KEYS)}

        tn = reduced["true_negative_count"]
        tp = reduced["true_positive_count"]
        fp = reduced["false_positive_count"]
        fn = reduced["false_negative_count"]
        total = tn + tp + fp + fn
        class0_total = reduced["target_zero_count"]
        class1_total = reduced["target_one_count"]

        reduced["accuracy"] = (tn + tp) / total if total > 0 else 0.0
        reduced["class0_accuracy"] = tn / class0_total if class0_total > 0 else 0.0
        reduced["class1_accuracy"] = tp / class1_total if class1_total > 0 else 0.0
        reduced["window_start_step"] = float(self.start_step)
        reduced["window_end_step"] = float(self.end_step)
        return reduced

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict, bool]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        rabc_weights_provider: Optional RABCWeights instance for sample weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    grad_norm = torch.tensor(0.0, device=accelerator.device)
    did_step = False

    with accelerator.accumulate(policy):
        # Let accelerator handle mixed precision
        with accelerator.autocast():
            # Use per-sample loss when RA-BC is enabled for proper weighting
            if rabc_batch_weights is not None:
                # Get per-sample losses
                per_sample_loss, output_dict = policy.forward(batch, reduction="none")

                # Apply RA-BC weights: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
                # rabc_batch_weights is already normalized to sum to batch_size
                epsilon = 1e-6
                loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
                # Log raw mean weight (before normalization) - this is the meaningful metric
                output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
                output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
                output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
            else:
                loss, output_dict = policy.forward(batch)

            # TODO(rcadene): policy.unnormalize_outputs(out_dict)

        # Use accelerator's backward method
        accelerator.backward(loss)

        # Only step optimizer at accumulation boundaries.
        if accelerator.sync_gradients:
            # Clip gradients if specified
            if grad_clip_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), float("inf"), error_if_nonfinite=False
                )

            # Optimizer step
            with lock if lock is not None else nullcontext():
                optimizer.step()

            optimizer.zero_grad()

            # Step through pytorch scheduler at every optimizer update
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Update internal buffers if policy has update method
            if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
                accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
            did_step = True

    # Batched Augmentation on GPU
    if hasattr(policy, "image_augmenter") and policy.image_augmenter is not None:
        image_keys = [k for k in batch.keys() if "image" in k]
        for key in image_keys:
            batch[key] = policy.image_augmenter(batch[key])

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict, did_step

@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when policy.device is set to CPU.
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        )

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    log_file = None
    if is_main_process and hasattr(cfg, "output_dir") and cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)
        log_file = cfg.output_dir / "train.log"
        
    init_logging(accelerator=accelerator, log_file=log_file)

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    ##############################################################################################################
    ##############################################################################################################
    # === CUSTOM DATA SLICING FOR ACTION AND STATE STRATEGIES === VLA_THESIS
    ##############################################################################################################
    ##############################################################################################################

    xvla_slice_spec = None
    if cfg.policy.type == "xvla":
        xvla_slice_spec = get_so101_slice_spec(getattr(cfg.policy, "action_mode", ""))

    if hasattr(dataset, "meta") and xvla_slice_spec is not None:
        slice_dataset_meta_in_place(dataset.meta, xvla_slice_spec)

    if cfg.policy.type == "xvla":
        _enforce_xvla_finetune_contract(cfg.policy)
        _rebuild_xvla_visual_input_features(cfg.policy, dataset.meta, cfg.rename_map)

    if is_main_process and xvla_slice_spec is not None:
        logging.info(colored("\n--- DATA SLICING APPLIED ---", "yellow", attrs=["bold"]))
        logging.info(colored(f"  > Detected Action Mode: {xvla_slice_spec.action_mode}", "cyan"))
        logging.info(colored(f"  > Sliced 'action' & 'observation.state' dim bounds: [{xvla_slice_spec.start}:{xvla_slice_spec.end}]", "cyan"))

    ##############################################################################################################
    ##############################################################################################################

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    
    policy = make_policy( cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    # Safety: explicitly stamp the CLI-provided normalization_mapping onto the policy config.
    # policy.config IS cfg.policy (same object), but this guard makes the intent explicit and
    # protects against future refactors that might deep-copy the config inside make_policy.
    policy.config.normalization_mapping = cfg.policy.normalization_mapping

    if is_main_process:
        logging.info(colored("\n--- POLICY CONFIGURATION ---", "yellow", attrs=["bold"]))
        logging.info(f"  > Input Features:  {', '.join(policy.config.input_features.keys())}")
        logging.info(f"  > Output Features: {', '.join(policy.config.output_features.keys())}")
        if cfg.policy.type == "xvla":
            logging.info(
                "  > XVLA Finetune Contract: tokenizer_max_length=%s num_image_views=%s empty_cameras=%s max_len_seq=%s",
                policy.config.tokenizer_max_length,
                policy.config.num_image_views,
                policy.config.empty_cameras,
                policy.config.max_len_seq,
            )
        # logging.info(colored("  > Norm Mapping (embedded in every saved config.json):", "green"))
        # for k, v in policy.config.normalization_mapping.items():
        #     logging.info(colored(f"      {k}: {v}", "green"))

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        # Convert CLI peft config to dict for overrides
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Move custom augmenter from dataset to policy and to device for GPU-side augmentation
    if hasattr(dataset, "image_augmenter") and dataset.image_augmenter is not None:
        policy.image_augmenter = dataset.image_augmenter.to(device)
        if is_main_process:
            logging.info("✓ Custom image augmenter transferred to GPU (batched mode enabled)")

    action_mode = getattr(cfg.policy, "action_mode", "N/A")

    if is_main_process:
        # Debugging for xVLA action mode and dimensions
        if hasattr(policy, "model") and hasattr(policy.model, "action_space"):
            dim_action = policy.model.dim_action
            logging.info(colored("XVLA Action Configuration:", "cyan", attrs=["bold"]))
            logging.info(f"  > Action Mode: {action_mode}")
            logging.info(f"  > Expected Action Dim (Model): {dim_action}")
            
            # Check if this matches the dataset action dim if possible
            if "action" in dataset.meta.features:
                feat = dataset.meta.features["action"]
                ds_action_dim = feat["shape"][0] if isinstance(feat, dict) else feat.shape[0]
                logging.info(f"  > Dataset Action Dim: {ds_action_dim}")
                if action_mode == "auto" and ds_action_dim != dim_action:
                    logging.warning(f"  ! Action dim mismatch in 'auto' mode: model={dim_action}, dataset={ds_action_dim}")

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    # For SARM, always provide dataset_meta for progress normalization
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if cfg.policy.pretrained_path is not None:
        processor_stats = _patch_xvla_gripper_stats_for_overrides(cfg.policy, dataset.meta.stats)
        # Preprocessor
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": processor_stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }

        # Custom remap for matching camera names.
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        # Postprocesor
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": processor_stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }


    if is_main_process:
        logging.info(colored("\n--- DATASET PIPELINE CONFIGURATION ---", "yellow", attrs=["bold"]))
        logging.info(f"  > Dataset Stat Keys: {', '.join(dataset.meta.stats.keys())}")
        logging.info(f"  > Input Features:    {', '.join(policy.config.input_features.keys())}")
        # logging.info("  > Norm Map:")
        # for k, v in policy.config.normalization_mapping.items():
        #     logging.info(f"      {k}: {v}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )
    
    ####################################################################################################################
    # VLA_THESIS: Insert SliceProcessorStep into preprocessor if xvla slicing is needed and not already present
    ######################################################################################################################
    if xvla_slice_spec is not None:
        from lerobot.processor.slice_processor import SliceProcessorStep
        from lerobot.processor.normalize_processor import NormalizerProcessorStep
        
        slice_map = {}
        if "action" in dataset.meta.features:
            slice_map["action"] = (xvla_slice_spec.start, xvla_slice_spec.end)
        if "observation.state" in dataset.meta.features:
            slice_map["observation.state"] = (xvla_slice_spec.start, xvla_slice_spec.end)
            
        if slice_map:
            has_slice_step = any(
                isinstance(step_, SliceProcessorStep) and getattr(step_, "slice_map", None) == slice_map
                for step_ in preprocessor.steps
            )
            
            # MAKE SURE TO INSERT THE SLICE STEP BEFORE ANY NORMALIZATION STEP, so that the normalizer stats are computed on the sliced data
            if not has_slice_step:
                slice_step = SliceProcessorStep(slice_map=slice_map)
                insert_idx = len(preprocessor.steps)
                for idx, step_ in enumerate(preprocessor.steps):
                    if isinstance(step_, NormalizerProcessorStep):
                        insert_idx = idx
                        break
                preprocessor.steps.insert(insert_idx, slice_step)

    if is_main_process:
        logging.info(colored("\n--- PRE-POST PROCESSORS INITIALIZED ---", "cyan", attrs=["bold"]))
        
        logging.info("  > Preprocessor steps:")
        for s in preprocessor.steps:
            logging.info(f"      - {type(s).__name__}")
            # if type(s).__name__ == "NormalizerProcessorStep":
            # log_processor_stats(s, dataset.meta)
                
        logging.info("  > Postprocessor steps:")
        for s in postprocessor.steps:
            logging.info(f"      - {type(s).__name__}")
            # if type(s).__name__ == "UnnormalizerProcessorStep":
            # log_processor_stats(s, dataset.meta)

        # Log Rename Map details
        if cfg.rename_map:
            logging.info(colored("  Image Mapping (Rename Map):", "cyan"))
            for src, dst in cfg.rename_map.items():
                if "image" in src or "image" in dst:
                    logging.info(f"    [MAPPING] {src}  --->  {dst}")

    if cfg.policy.type == "xvla":
        _assert_xvla_finetune_contract(policy.config)
        if is_main_process:
            _validate_xvla_sequence_budget(policy, dataset, preprocessor)
        accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes * cfg.gradient_accumulation_steps
        logging.info(
            "Effective batch size: "
            f"{cfg.batch_size} x {num_processes} x {cfg.gradient_accumulation_steps} = {effective_bs}"
        )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    
    dataloader = torch.utils.data.DataLoader( dataset, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=shuffle and not cfg.dataset.streaming, sampler=sampler, pin_memory=device.type == "cuda", drop_last=False, prefetch_factor=2 if cfg.num_workers > 0 else None,)

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(policy, optimizer, dataloader, lr_scheduler)
    dl_iter = cycle(dataloader)
    policy.train()
    train_metrics = { "loss": AverageMeter("loss", ":.3f"), "grad_norm": AverageMeter("grdn", ":.3f"), "lr": AverageMeter("lr", ":0.1e"), "update_s": AverageMeter("updt_s", ":.3f"), "dataloading_s": AverageMeter("data_s", ":.3f"),}
    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps
    train_tracker = MetricsTracker( effective_batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step, accelerator=accelerator,)

    if is_main_process:
        logging.info(f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}")

    gripper_debug_window = (
        GripperDebugWindow(start_step=step + 1)
        if getattr(accelerator.unwrap_model(policy), "config", None) is not None
        and getattr(accelerator.unwrap_model(policy).config, "enable_gripper_debug_stats", False)
        else None
    )

    ##############################################################################################################
    ##############################################################################################################
    ######################################xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx########################################
    ######################################   START OF THE TRAINING LOOP   ########################################
    ######################################xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx########################################
    ##############################################################################################################
    ##############################################################################################################

    progress_refresh_freq = cfg.log_freq if cfg.log_freq > 0 else 1
    progress_start_step = step
    progress_start_time = time.time()
    progress_bar = (tqdm(total=cfg.steps, initial=step, desc="Training", dynamic_ncols=True, leave=True) if is_main_process else nullcontext())

    print("-------------------------------------------------------------------------------------------------------")
    print("Starting Training Loop")
    print("-------------------------------------------------------------------------------------------------------")
    with progress_bar as training_progress_bar:
        optimizer.zero_grad()
        while step < cfg.steps:

            start_time = time.perf_counter()
            raw_batch = next(dl_iter)

            # PRE PROCESSING
            # debug_batch(raw_batch, tag="RAW (before preprocess)", step=step, dataset_meta=dataset.meta if hasattr(dataset, "meta") else None)
            # if is_main_process:
            #     # Happens only on step 0
            #     save_debug_images(batch, cfg.output_dir, step=0, prefix="raw")
    
            batch = preprocessor(raw_batch)
            # debug_batch(batch, tag="POST (after preprocess)", step=step, dataset_meta=dataset.meta if hasattr(dataset, "meta") else None)
            # if is_main_process:
            #     save_debug_images(batch, cfg.output_dir, step=0, prefix="post")

            # Use data
            train_tracker.dataloading_s = time.perf_counter() - start_time
            train_tracker, output_dict, did_step = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                accelerator=accelerator,
                lr_scheduler=lr_scheduler,
                rabc_weights_provider=rabc_weights,
            )
            if not did_step:
                continue
            
            slice_dim = xvla_slice_spec.real_dim if xvla_slice_spec is not None else None

            # debug_batch(output_dict, tag="MODEL OUTPUT dict", step=step, slice_dim=slice_dim, dataset_meta=dataset.meta if hasattr(dataset, "meta") else None)
            gripper_debug_counts = output_dict.pop("gripper_debug_counts", None)

            ##############################################################################################################
            ##############################################################################################################
            # LOGS AND STUFF
            ##############################################################################################################
            ##############################################################################################################

            # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
            # increment `step` here.
            step += 1
            train_tracker.step()
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
            should_log_gripper_window = gripper_debug_window is not None and (
                (cfg.log_freq > 0 and step % cfg.log_freq == 0) or step == cfg.steps
            )
            is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
            is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

            if training_progress_bar is not None:
                training_progress_bar.update(1)
                should_refresh_progress = step % progress_refresh_freq == 0 or step == cfg.steps
                if should_refresh_progress:
                    loss_value = train_tracker.loss.val if train_tracker.loss.count > 0 else None
                    training_progress_bar.set_postfix(
                        _build_training_progress_postfix(
                            step=step,
                            total_steps=cfg.steps,
                            elapsed_s=time.time() - progress_start_time,
                            start_step=progress_start_step,
                            loss=loss_value,
                        ),
                        refresh=True,
                    )

            if gripper_debug_window is not None and gripper_debug_counts is not None:
                gripper_debug_window.update(step=step, count_dict=gripper_debug_counts)

            if is_log_step:
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    # Log RA-BC statistics if enabled
                    if rabc_weights is not None:
                        rabc_stats = rabc_weights.get_stats()
                        wandb_log_dict.update(
                            {
                                "rabc_delta_mean": rabc_stats["delta_mean"],
                                "rabc_delta_std": rabc_stats["delta_std"],
                                "rabc_num_frames": rabc_stats["num_frames"],
                            }
                        )
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            if should_log_gripper_window:
                gripper_window_dict = gripper_debug_window.as_reduced_dict(accelerator)
                if is_main_process:
                    window_start = int(gripper_window_dict["window_start_step"])
                    window_end = int(gripper_window_dict["window_end_step"])
                    logging.info(
                        "gripper window %d-%d target0=%.0f target1=%.0f pred0=%.0f pred1=%.0f "
                        "tn=%.0f tp=%.0f fp=%.0f fn=%.0f acc=%.4f class0_acc=%.4f class1_acc=%.4f",
                        window_start,
                        window_end,
                        gripper_window_dict["target_zero_count"],
                        gripper_window_dict["target_one_count"],
                        gripper_window_dict["pred_zero_count"],
                        gripper_window_dict["pred_one_count"],
                        gripper_window_dict["true_negative_count"],
                        gripper_window_dict["true_positive_count"],
                        gripper_window_dict["false_positive_count"],
                        gripper_window_dict["false_negative_count"],
                        gripper_window_dict["accuracy"],
                        gripper_window_dict["class0_accuracy"],
                        gripper_window_dict["class1_accuracy"],
                    )
                    if wandb_logger:
                        wandb_logger.log_dict(
                            {
                                "gripper/window_start_step": window_start,
                                "gripper/window_end_step": window_end,
                                "gripper/target_zero_count": gripper_window_dict["target_zero_count"],
                                "gripper/target_one_count": gripper_window_dict["target_one_count"],
                                "gripper/pred_zero_count": gripper_window_dict["pred_zero_count"],
                                "gripper/pred_one_count": gripper_window_dict["pred_one_count"],
                                "gripper/tn": gripper_window_dict["true_negative_count"],
                                "gripper/tp": gripper_window_dict["true_positive_count"],
                                "gripper/fp": gripper_window_dict["false_positive_count"],
                                "gripper/fn": gripper_window_dict["false_negative_count"],
                                "gripper/accuracy": gripper_window_dict["accuracy"],
                                "gripper/class0_accuracy": gripper_window_dict["class0_accuracy"],
                                "gripper/class1_accuracy": gripper_window_dict["class1_accuracy"],
                            },
                            step,
                        )
                gripper_debug_window.reset(next_start_step=step + 1)

            ##############################################################################################################
            ##############################################################################################################
            # SAVING CHECKPOINTS
            ##############################################################################################################
            ##############################################################################################################

            if cfg.save_checkpoint and is_saving_step:
                if is_main_process:
                    logging.info(f"Checkpoint policy after step {step}")
                    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                    if cfg.policy.type == "xvla":
                        _assert_xvla_finetune_contract(accelerator.unwrap_model(policy).config)
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=step,
                        cfg=cfg,
                        policy=accelerator.unwrap_model(policy),
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                    )
                    update_last_checkpoint(checkpoint_dir)
                    if wandb_logger:
                        wandb_logger.log_policy(checkpoint_dir)

                accelerator.wait_for_everyone()

            ##############################################################################################################
            ##############################################################################################################
            # PUSH STEP FOR CUSTOM PUSH EVERY
            ##############################################################################################################
            ##############################################################################################################

            is_push_step = cfg.push_every > 0 and step % cfg.push_every == 0 and step != cfg.steps
            if is_push_step:
                if is_main_process:
                    logging.info(f"Pushing checkpoint to Hub after step {step}")
                    norm_map_str = ", ".join(f"{k}={v}" for k, v in accelerator.unwrap_model(policy).config.normalization_mapping.items())
                    logging.info(colored(f"  > Saving config.json with normalization_mapping: [{norm_map_str}]", "green"))
                    original_repo_id = cfg.policy.repo_id
                    # Smart naming: append step count
                    cfg.policy.repo_id = f"{original_repo_id}-step-{step}"

                    unwrapped_policy = accelerator.unwrap_model(policy)
                    if cfg.policy.type == "xvla":
                        _assert_xvla_finetune_contract(unwrapped_policy.config)
                    
                    # Push the files to the repo in a single commit by saving to a local tmp dir
                    from tempfile import TemporaryDirectory
                    from pathlib import Path
                    from huggingface_hub import HfApi

                    api = HfApi()
                    api.create_repo(repo_id=cfg.policy.repo_id, private=cfg.policy.private, exist_ok=True)
                    
                    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                        saved_path = Path(tmp) / cfg.policy.repo_id.split("/")[-1]
                        
                        if cfg.policy.use_peft:
                            unwrapped_policy.save_pretrained(saved_path)
                            unwrapped_policy.config.save_pretrained(saved_path)
                        else:
                            unwrapped_policy.save_pretrained(saved_path)

                        cfg.save_pretrained(saved_path)
                        if preprocessor:
                            preprocessor.save_pretrained(saved_path)
                        if postprocessor:
                            postprocessor.save_pretrained(saved_path)
                            
                        card = unwrapped_policy.generate_model_card(
                            cfg.dataset.repo_id, unwrapped_policy.config.type, unwrapped_policy.config.license, unwrapped_policy.config.tags
                        )
                        card.save(str(saved_path / "README.md"))

                        api.upload_folder(repo_id=cfg.policy.repo_id, repo_type="model", folder_path=str(saved_path), commit_message=f"Upload checkpoint for step {step}", allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"], ignore_patterns=["*.tmp", "*.log"])

                    # Restore original repo_id for the next steps
                    cfg.policy.repo_id = original_repo_id

                accelerator.wait_for_everyone()

            if cfg.env and is_eval_step:
                if is_main_process:
                    step_id = get_step_identifier(step, cfg.steps)
                    logging.info(f"Eval policy at step {step}")
                    with torch.no_grad(), accelerator.autocast():
                        eval_info = eval_policy_all(
                            envs=eval_env,  # dict[suite][task_id] -> vec_env
                            policy=accelerator.unwrap_model(policy),
                            env_preprocessor=env_preprocessor,
                            env_postprocessor=env_postprocessor,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            n_episodes=cfg.eval.n_episodes,
                            videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                            max_episodes_rendered=4,
                            start_seed=cfg.seed,
                            max_parallel_tasks=cfg.env.max_parallel_tasks,
                        )
                    # overall metrics (suite-agnostic)
                    aggregated = eval_info["overall"]

                    # optional: per-suite logging
                    for suite, suite_info in eval_info.items():
                        logging.info("Suite %s aggregated: %s", suite, suite_info)

                    # meters/tracker
                    eval_metrics = {"avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),"pc_success": AverageMeter("success", ":.1f"),"eval_s": AverageMeter("eval_s", ":.3f"),}
                    eval_tracker = MetricsTracker(cfg.batch_size,dataset.num_frames,dataset.num_episodes,eval_metrics,initial_step=step,accelerator=accelerator,)
                    eval_tracker.eval_s = aggregated.pop("eval_s")
                    eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                    eval_tracker.pc_success = aggregated.pop("pc_success")
                    if wandb_logger:
                        wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                        wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                        wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

                accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.type == "xvla":
                _assert_xvla_finetune_contract(unwrapped_policy.config)
            norm_map_str = ", ".join(f"{k}={v}" for k, v in unwrapped_policy.config.normalization_mapping.items())
            logging.info(colored(f"Final push — saving config.json with normalization_mapping: [{norm_map_str}]", "green"))
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
