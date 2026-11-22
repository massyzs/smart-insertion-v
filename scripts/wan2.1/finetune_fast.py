"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

Stage 2 (insertion fine-tuning), throughput-optimised copy of finetune.py for clusters whose nodes do not
share a filesystem. The training maths is the original: same model, loss, timestep strata, noise, optimizer,
learning-rate schedule, gradient clipping, bf16 mixed precision and FSDP wrapping. Every changed block is
marked with "[FAST]":

  * data: InsertionDatasetFast (image entries become single-frame items, empty ref_img entries are dropped)
    with a rank-aware, deterministic sampler; decoding runs in dataloader workers;
  * encoders: the VAE, Qwen3-VL and T5 are no longer moved between CPU and GPU several times per step, and
    T5 is no longer FSDP-sharded (--qwen_placement / --t5_placement);
  * no per-step barriers, cache flushes or per-rank prints;
  * checkpoints: one self-contained file per rank on the node-local disk, the consolidated weights are
    exported once (not twice) per checkpoint, the AdamW state is really restored on resume, the data position
    after a resume is exact, and the last step is always saved;
  * a plain-text progress line with timings and ETA, and a stop file (<output_dir>/STOP) for a clean early stop.
"""
#!/usr/bin/env python
# coding=utf-8
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

import argparse
import gc
import hashlib
import logging
import math
import os
import pickle
import shutil
import sys
import threading

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, FullStateDictConfig, ShardedStateDictConfig, ShardedOptimStateDictConfig)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
import json
import random
import time
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers
from qwen_encoder import MLLMInContext, MLLMInContextConfig
import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.data.bucket_sampler import (ASPECT_RATIO_512,
                                           ASPECT_RATIO_RANDOM_CROP_512,
                                           ASPECT_RATIO_RANDOM_CROP_PROB,
                                           AspectRatioBatchImageVideoSampler,
                                           RandomSampler, get_closest_ratio)
from videox_fun.data.dataset_image_video import (ImageVideoDataset,PretrainVideoDataset,
                                                ImageVideoSampler,
                                                get_random_mask)
# [FAST] dataset with single-frame image items and the rank-aware step sampler
from videox_fun.data.dataset_insertion_fast import InsertionDatasetFast as InsertionDataset
from videox_fun.data.dataset_insertion_fast import StepGroupedBatchSampler, insertion_worker_init_fn
from videox_fun.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel,
                              )
# [FAST] SI_FAST_MODEL=1 selects the self-contained fast copy of the model file (same classes, parameter names and
# state_dict; fewer full-size elementwise passes per block, differences at the rounding level). Default: the original.
SI_FAST_MODEL = os.environ.get("SI_FAST_MODEL", "0") == "1"
if SI_FAST_MODEL:
    from videox_fun.models.wan_transformer3d_insertion_fast import WanTransformer3DModel
else:
    from videox_fun.models.wan_transformer3d_insertion_new import WanTransformer3DModel
from videox_fun.pipeline import WanPipeline, WanI2VPipeline
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.utils import get_image_to_video_latent, save_videos_grid

if is_wandb_available():
    import wandb


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def get_random_downsample_ratio(sample_size, image_ratio=[],
                                all_choices=False, rng=None):
    def _create_special_list(length):
        if length == 1:
            return [1.0]
        if length >= 2:
            first_element = 0.75
            remaining_sum = 1.0 - first_element
            other_elements_value = remaining_sum / (length - 1)
            special_list = [first_element] + [other_elements_value] * (length - 1)
            return special_list
            
    if sample_size >= 1536:
        number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio 
    elif sample_size >= 1024:
        number_list = [1, 1.25, 1.5, 2] + image_ratio
    elif sample_size >= 768:
        number_list = [1, 1.25, 1.5] + image_ratio
    elif sample_size >= 512:
        number_list = [1] + image_ratio
    else:
        number_list = [1]

    if all_choices:
        return number_list

    number_list_prob = np.array(_create_special_list(len(number_list)))
    if rng is None:
        return np.random.choice(number_list, p = number_list_prob)
    else:
        return rng.choice(number_list, p = number_list_prob)

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value

def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)


class QuietDiscreteSampling(DiscreteSampling):
    """[FAST] DiscreteSampling without its per-step print on every rank. The draws are the same: with
    uniform sampling each rank keeps drawing from its own timestep stratum."""

    def __call__(self, n_samples, generator=None, device=None):
        if self.is_distributed and self.uniform_sampling:
            group_index = self.rank // self.group_width
            idx = torch.randint(
                    self.start_num_idx + group_index * self.sigma_interval,
                    self.start_num_idx + (group_index + 1) * self.sigma_interval,
                    (n_samples,),
                    generator=generator, device=device,
                )
        else:
            idx = torch.randint(
                    self.start_num_idx, self.start_num_idx + self.num_idx, (n_samples,),
                    generator=generator, device=device,
                )
        return idx


class HostSwappedModule:
    """[FAST] A frozen module whose weights live in page-locked host memory and are on the GPU only while used.

    The original script moved such modules with module.to("cpu") / module.to(device) every step: pageable
    copies in both directions. The weights never change, so the copy back is unnecessary; here the host
    copy is pinned once and each use is one host-to-GPU copy:

        load_async()  allocate GPU tensors and start the copies on a side stream (returns immediately, so
                      the caller can run other GPU work, e.g. the VAE, while the copies are in flight)
        wait()        make the current stream wait for the copies; call before using the module
        unload()      drop the GPU tensors; their memory goes back to the allocator for the transformer

    The values the module computes with are bit-identical to module.to(device).
    """

    def __init__(self, module, device, name):
        self.module = module
        self.device = device
        self.name = name
        self.tensors = list(module.parameters()) + list(module.buffers())
        self.host = [t.detach().to("cpu").contiguous().pin_memory() for t in self.tensors]
        self.placeholders = {}
        for t in self.tensors:
            if t.dtype not in self.placeholders:
                self.placeholders[t.dtype] = torch.empty(0, dtype=t.dtype, device=device)
        self.stream = torch.cuda.Stream(device=device)
        self.loaded = False
        self.unload()
        self.nbytes = sum(h.numel() * h.element_size() for h in self.host)

    def load_async(self):
        if self.loaded:
            return
        current = torch.cuda.current_stream(self.device)
        gpu = [torch.empty(h.shape, dtype=h.dtype, device=self.device) for h in self.host]
        # The blocks just handed out may still be referenced by kernels queued on the current stream
        # (temporaries of the previous step): the side stream must not write to them before those ran.
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            for g, h in zip(gpu, self.host):
                g.copy_(h, non_blocking=True)
        for t, g in zip(self.tensors, gpu):
            t.data = g
        self.loaded = True

    def wait(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

    def unload(self):
        for t in self.tensors:
            t.data = self.placeholders[t.dtype]
        self.loaded = False

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=(
            "A csv containing the training data. "
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--selective_ac",
        type=float,
        default=0,
        help="Rate for transformer block apply checkpointing.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--high_lr",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--low_lr",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_model_info", action="store_true", help="Whether or not to report more info about model (such as norm, grad)."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    
    parser.add_argument(
        "--snr_loss", action="store_true", help="Whether or not to use snr_loss."
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader", action="store_true", help="Whether or not to use text encoder in dataloader."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_frame_crop", action="store_true", help="Whether enable random frame crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--training_with_video_token_length", action="store_true", help="The training stage of the model in training.",
    )
    parser.add_argument(
        "--auto_tile_batch_size", action="store_true", help="Whether to auto tile batch size.",
    )
    parser.add_argument(
        "--motion_sub_loss", action="store_true", help="Whether enable motion sub loss."
    )
    parser.add_argument(
        "--motion_sub_loss_ratio", type=float, default=0.25, help="The ratio of motion sub loss."
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--keep_all_node_same_token_length",
        action="store_true", 
        help="Reference of the length token.",
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=512,
        help="Sample size of the token.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="Sample size of the image.",
    )
    parser.add_argument(
        "--video_sample_size_h",
        type=int,
        default=480,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--video_sample_size_w",
        type=int,
        default=832,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--fix_sample_size", 
        nargs=2, type=int, default=None,
        help="Fix Sample size [height, width] when using bucket and collate_fn."
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=4,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=17,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Num of repeat video.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )

    parser.add_argument(
        '--trainable_modules', 
        nargs='+', 
        # help='Enter a list of trainable modules'
        help='Trainable modules: "all" = train everything, '
             '"transformer" = train transformer (exclude project_in), '
             '"project_in" = only train qwen_project_in. (default: project_in)'
             "text_embedding_vlm.0.weight" 
    )
    parser.add_argument(
        '--trainable_modules_low_learning_rate', 
        nargs='+', 
        default=[],
        help='Enter a list of trainable modules with lower learning rate'
    )
    parser.add_argument(
        '--tokenizer_max_length', 
        type=int,
        default=512,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--use_fsdp", action="store_true", help="Whether or not to use fsdp."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="normal",
        help=(
            'The format of training data. Support `"normal"`'
            ' (default), `"i2v"`.'
        ),
    )
    parser.add_argument(
        "--use_t5",
        action="store_true",
    )
    parser.add_argument(
        "--abnormal_norm_clip_start",
        type=int,
        default=1000,
        help=(
            'When do we start doing additional processing on abnormal gradients. '
        ),
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=5,
        help=(
            'The initial gradient is relative to the multiple of the max_grad_norm. '
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--check_json",
        action="store_true",
    )
    
    parser.add_argument(
        "--ref_condition_json_dir",
        type=str,
        nargs='+',
        default=None,
        help="Path(s) to JSON file(s) or directory(ies) for RefConditionVideoDataset. "
             "Each JSON should have gt_video, input_video, ref_img, and prompt fields. "
             "Supports single or multiple paths separated by spaces.",
    )
    parser.add_argument(
        "--pretrain_dataset",
        type=str,
        nargs='+',
        default=None,
        help="Path(s) to JSON file(s) or directory(ies) for PretrainVideoDataset. "
             "Supports single or multiple paths separated by spaces. "
             "Examples: "
             "--pretrain_dataset /path/to/data.json (single file), "
             "--pretrain_dataset /path/to/data1.json /path/to/data2.json (multiple files), "
             "--pretrain_dataset /path/to/json_dir/ (directory), "
             "--pretrain_dataset /path/to/data1.json /path/to/dir/ /path/to/data2.json (mixed)",
    )
    
    # ====== Qwen Encoder Arguments (this script requires the Qwen encoder) ======
    parser.add_argument(
        "--qwen_encoder_mode",
        type=str,
        default=None,  # no built-in default: the launch script must pass a mode ("edit" in finetune.sh)
        choices=["t2v", "edit","i2v"],  # a Qwen encoder mode is required for this script
        help="Qwen encoder mode. 't2v': empty prompt. 'i2v': first frame + empty prompt. 'edit': reference image + source first frame + instruction. (Required for this script)",
    )
    parser.add_argument(
        "--qwen_encoder_path",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Path to Qwen-VL model.",
    )
    parser.add_argument(
        "--insertion_cache",
        type=str,
        default="PATH/insertion_cache.json",
        help="",
    )

    # ====== [FAST] arguments of finetune_fast.py ======
    parser.add_argument(
        "--no_group_by_type", dest="group_by_type", action="store_false",
        help="By default every global step holds samples of one kind (all video clips or all single-frame images), "
             "so image ranks never wait for video ranks. Pass this flag to mix kinds inside a step.",
    )
    parser.set_defaults(group_by_type=True)
    parser.add_argument(
        "--qwen_placement", type=str, default="swap", choices=["swap", "resident"],
        help="Where the frozen Qwen3-VL lives: 'swap' = pinned host memory, copied to the GPU for its two calls each step; "
             "'resident' = always on the GPU (about 16 GiB more GPU memory during the transformer pass).",
    )
    parser.add_argument(
        "--t5_placement", type=str, default="swap", choices=["swap", "resident", "fsdp"],
        help="Where the frozen T5 lives: 'swap' / 'resident' as for Qwen (about 11 GiB), or 'fsdp' = sharded across "
             "all ranks as in the original script (an all-gather over all ranks every step).",
    )
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2, help="Batches prefetched per dataloader worker.")
    parser.add_argument("--dataloader_timeout", type=int, default=480,
                        help="Seconds before a stuck dataloader worker raises (kept below the 10-minute NCCL collective timeout, "
                             "so the rank with the stuck worker reports it before the other ranks abort).")
    parser.add_argument("--stall_minutes", type=float, default=30.0,
                        help="Exit the process when no training progress has been made for this long (a rank waiting forever on a "
                             "dead peer keeps its GPU busy and its launcher alive). 0 disables.")
    parser.add_argument("--allow_fresh_start", action="store_true",
                        help="With --resume_from_checkpoint latest: start from step 0 even if some ranks (but not all) hold checkpoints.")
    parser.add_argument("--log_every", type=int, default=10, help="Print a progress line (and profile one step) every N steps.")
    parser.add_argument(
        "--min_free_gpu_gib", type=float, default=70.0,
        help="At start-up wait (up to 10 minutes) until the GPU has this much free memory, then fail. "
             "Protects against launching next to another job that still holds the card.",
    )
    parser.add_argument(
        "--require_fa3", action="store_true",
        help="Fail at start-up unless FlashAttention-3 is importable on every rank (it is found through PYTHONPATH; "
             "without this flag a missing FlashAttention-3 build silently falls back to FlashAttention-2).",
    )

    args = parser.parse_args()
    # [FAST] combinations this script does not support
    if not args.group_by_type and args.train_batch_size > 1:
        raise ValueError("--no_group_by_type needs --train_batch_size 1 (a per-rank batch could mix 1-frame and 33-frame samples)")
    if args.enable_text_encoder_in_dataloader:
        raise ValueError("--enable_text_encoder_in_dataloader is not supported by finetune_fast.py")
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    config = OmegaConf.load(args.config_path)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # [FAST] Create the CUDA context right away (model loading below is minutes of CPU-only work during which the
    # card would otherwise look idle) and make sure the card is actually free before loading 100+ GB of models.
    if torch.cuda.is_available() and accelerator.device.type == "cuda":
        torch.zeros(1, device=accelerator.device)
        _waited = 0
        while True:
            _free_gib = torch.cuda.mem_get_info(accelerator.device)[0] / 2**30
            if _free_gib >= args.min_free_gpu_gib:
                break
            if _waited >= 600:
                raise RuntimeError(f"[rank {accelerator.process_index}] only {_free_gib:.1f} GiB free on {accelerator.device} "
                                   f"after waiting {_waited}s (need {args.min_free_gpu_gib} GiB): another job is holding the card")
            if _waited % 60 == 0:
                print(f"[rank {accelerator.process_index}] waiting for GPU memory: {_free_gib:.1f} GiB free on {accelerator.device}", flush=True)
            time.sleep(10)
            _waited += 10
        print(f"[rank {accelerator.process_index}] {accelerator.device}: {_free_gib:.1f} GiB free at start", flush=True)

    # [FAST] Stall guard. If the ranks of the other node die, this rank can sit in a collective until PyTorch's NCCL
    # timeout fires, or forever before the first collective; its GPU then looks busy and the launcher never reaches
    # its exit handling. A daemon thread ends the process when no progress mark has been set for --stall_minutes.
    progress = {"t": time.time(), "what": "start-up"}

    def mark_progress(what):
        progress["t"] = time.time()
        progress["what"] = what

    if args.stall_minutes > 0:
        def _stall_guard():
            while True:
                time.sleep(30)
                idle = time.time() - progress["t"]
                if idle > args.stall_minutes * 60:
                    print(f"[STALL] rank {accelerator.process_index}: no progress for {idle / 60:.1f} min "
                          f"(last: {progress['what']}); exiting so the launcher can clean up", flush=True)
                    os._exit(86)
        threading.Thread(target=_stall_guard, daemon=True).start()

    deepspeed_plugin = accelerator.state.deepspeed_plugin if hasattr(accelerator.state, "deepspeed_plugin") else None
    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        fsdp_stage = 0
        print(f"Using DeepSpeed Zero stage: {zero_stage}")

        args.use_deepspeed = True
        if zero_stage == 3:
            print(f"Auto set save_state to True because zero_stage == 3")
            args.save_state = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        print(f"Using FSDP stage: {fsdp_stage}")

        args.use_fsdp = True
        if fsdp_stage == 3:
            print(f"Auto set save_state to True because fsdp_stage == 3")
            args.save_state = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("DeepSpeed is not enabled.")

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    # print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        
        # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        
        text_encoder = text_encoder.eval()
        transformer_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
        # print(f"[QWEN] Initializing: {args.qwen_encoder_path}, mode: {args.qwen_encoder_mode}")
        qwen_config = MLLMInContextConfig(mllm_id=args.qwen_encoder_path,mode=args.qwen_encoder_mode,crop_system_tokens=True,crop_vision_tokens=False)
        qwen_encoder = MLLMInContext(qwen_config).eval()
        # print(f"[QWEN] Hidden size: {qwen_encoder.mllm_hidden_size}")
        transformer_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
        transformer_kwargs['use_qwen_encoder'] = True
        transformer_kwargs['qwen_hidden_size'] = qwen_encoder.mllm_hidden_size
        transformer_kwargs['use_t5'] = args.use_t5
        # print(f"[------DEBUG-----]: {args.use_t5}")

        # Get Vae
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        vae.eval()
        # Get Clip Image Encoder
        if args.train_mode != "normal":
            clip_image_encoder = CLIPModel.from_pretrained(
                os.path.join(args.pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
            )
            clip_image_encoder = clip_image_encoder.eval()
            
    # Get Transformer
    transformer3d = WanTransformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=transformer_kwargs,
    ).to(weight_dtype)

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    qwen_encoder.requires_grad_(False)

    transformer3d.requires_grad_(False)
    if args.train_mode != "normal":
        clip_image_encoder.requires_grad_(False)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
    
    # A good trainable modules is showed below now.
    # For 3D Patch: trainable_modules = ['ff.net', 'pos_embed', 'attn2', 'proj_out', 'timepositionalencoding', 'h_position', 'w_position']
    # For 2D Patch: trainable_modules = ['ff.net', 'attn2', 'timepositionalencoding', 'h_position', 'w_position']

    transformer3d.train()
    # transformer3d.eval()

    # if accelerator.is_main_process:
    #     accelerator.print(
    #         f"Trainable modules '{args.trainable_modules}'."
    #     )

#?set grad
    high_kws = list(args.trainable_modules or [])
    low_kws  = list(args.trainable_modules_low_learning_rate or [])

    if len(high_kws) == 0 and len(low_kws) == 0:
        raise ValueError("Please pass --trainable_modules and/or --trainable_modules_low_learning_rate keywords.")

    # Freeze everything first
    for p in transformer3d.parameters():
        p.requires_grad = False

    high_params, low_params = [], []
    seen = set()

    def _match(name: str, kws):
        return any(kw in name for kw in kws)

    for name, param in transformer3d.named_parameters():
        # low takes priority: if a parameter matches both low/high, assign it to low (avoids duplicates and is more conservative)
        in_low  = _match(name, low_kws)
        in_high = _match(name, high_kws)

        if not (in_low or in_high):
            continue

        param.requires_grad = True

        # Deduplication guard (should never repeat in theory, but kept as a safeguard)
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)

        if in_low:
            low_params.append(param)
            # if accelerator.is_main_process:
                # print(f"  [TRAINABLE][LOW_LR]  {name}: {param.shape}")
        else:
            high_params.append(param)
            # if accelerator.is_main_process:
            #     print(f"  [TRAINABLE][HIGH_LR] {name}: {param.shape}")

    trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))

    # ===== build optimizer param groups =====
    trainable_params_optim = []
    if len(high_params) > 0:
        trainable_params_optim.append({"params": high_params, "lr": args.high_lr})
        if accelerator.is_main_process:
            print(f"Set {len(high_params)} parameter tensors to lr : {args.high_lr}")
    if len(low_params) > 0:
        trainable_params_optim.append({"params": low_params, "lr": args.low_lr})
        
        if accelerator.is_main_process:
            print(f"Set {len(low_params)} parameter tensors to lr : {args.low_lr}")

    # Create EMA for the transformer3d.
    if args.use_ema:
        if zero_stage == 3:
            raise NotImplementedError("FSDP does not support EMA.")

        ema_transformer3d = WanTransformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        ).to(weight_dtype)

        ema_transformer3d = EMAModel(ema_transformer3d.parameters(), model_cls=WanTransformer3DModel, model_config=ema_transformer3d.config)

    # [FAST] The accelerate save/load hooks of the original script are not used: checkpoints are written by
    # save_rank_state / export_weights below (node-local, self-contained per rank; see the notes there).

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
    elif args.selective_ac > 0:
        from videox_fun.utils.ac_handle import apply_checkpointing, partial
        from videox_fun.models.wan_transformer3d import WanAttentionBlock
        apply_selective_ac = partial(apply_checkpointing, block=WanAttentionBlock)
        apply_selective_ac(transformer3d, p=args.selective_ac)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
                
        
        try:
            # import bitsandbytes as bnb
            from torchao.optim import AdamW8bit
        except ImportError:
            raise ImportError(
                "Please install torchao to use 8-bit Adam. You can do so by running `pip install torchao`"
            )

        optimizer_cls = AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    # trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
    # trainable_params_optim = [
    #     {'params': [], 'lr': args.learning_rate},
    #     {'params': [], 'lr': args.learning_rate / 2},
    # ]
    # param_groups
    # in_already = []
    # for name, param in transformer3d.named_parameters():
    #     high_lr_flag = False
    #     if name in in_already:
    #         continue
    #     for trainable_module_name in args.trainable_modules:
    #         if trainable_module_name in name:
    #             in_already.append(name)
    #             high_lr_flag = True
    #             trainable_params_optim[0]['params'].append(param)
    #             if accelerator.is_main_process:
    #                 print(f"Set {name} to lr : {args.learning_rate}")
    #             break
    #     if high_lr_flag:
    #         continue
    #     for trainable_module_name in args.trainable_modules_low_learning_rate:
    #         if trainable_module_name in name:
    #             in_already.append(name)
    #             trainable_params_optim[1]['params'].append(param)
    #             if accelerator.is_main_process:
    #                 print(f"Set {name} to lr : {args.learning_rate / 2}")
    #             break

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            # weight_decay=args.adam_weight_decay,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Get the training dataset
    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio

    # ------------------------------------------------------------------------------------------------
    # [FAST] Dataset, sampler and dataloader.
    #   * InsertionDatasetFast: same video items as InsertionDatasetExpand, plus single-frame items for
    #     image entries (see videox_fun/data/dataset_insertion_fast.py).
    #   * StepGroupedBatchSampler is rank-aware, so the dataloader is NOT passed to accelerator.prepare
    #     (accelerate would shard it a second time). The position of every rank is a pure function of
    #     (seed, epoch, step): identical plan on all ranks and an exact data position after a resume.
    #   * Decoding runs in worker processes instead of the training process.
    # ------------------------------------------------------------------------------------------------
    print(f"[DATASET] Using InsertionDatasetFast from: {args.pretrain_dataset}")
    train_dataset = InsertionDataset(
        ann_path=args.pretrain_dataset,
        data_root=args.train_data_dir,
        video_sample_size_h=args.video_sample_size_h,
        video_sample_size_w=args.video_sample_size_w,
        video_sample_stride=args.video_sample_stride,
        video_sample_n_frames=args.video_sample_n_frames,
        enable_bucket=args.enable_bucket,
        check=args.check_json,  # optionally verify that every media file exists (result cached in check_output_json)
        check_output_json=args.output_dir+'/check_json.json',
        expand_cache_json=args.insertion_cache,
    )

    batch_sampler = StepGroupedBatchSampler(
        train_dataset.item_is_image,
        batch_size=args.train_batch_size,
        num_processes=accelerator.num_processes,
        process_index=accelerator.process_index,
        seed=args.seed,
        group_by_type=args.group_by_type,
    )
    dataloader_kwargs = {}
    if args.dataloader_num_workers != 0:
        dataloader_kwargs = dict(
            persistent_workers=True,
            prefetch_factor=args.dataloader_prefetch_factor,
            worker_init_fn=insertion_worker_init_fn,
            timeout=args.dataloader_timeout,   # a hung decode becomes an error instead of a silent stall
        )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.dataloader_num_workers,
        **dataloader_kwargs,
    )

    # Every rank must see exactly the same item list, otherwise the shared plan maps to different samples.
    _fp = hashlib.sha1()
    for (_di, _s, _n) in train_dataset.expanded_items:
        _e = train_dataset._raw_dataset[_di]
        _fp.update(f"{_e['input_video']}|{_e['gt_video']}|{_e['ref_img']}|{_e['gt_ref_img']}|{_s}|{_n}\n".encode("utf-8"))
    dataset_fingerprint = int.from_bytes(_fp.digest()[:7], "big")
    # ... and the same sampling plan, the same code and the same settings (each node runs its own copy of everything).
    _plan_fp = int.from_bytes(hashlib.sha1(batch_sampler.plan(0).tobytes()).digest()[:7], "big")
    _code = hashlib.sha1()
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _model_file = "videox_fun/models/wan_transformer3d_insertion_fast.py" if SI_FAST_MODEL else "videox_fun/models/wan_transformer3d_insertion_new.py"
    for _rel in ("scripts/wan2.1/finetune_fast.py", "scripts/wan2.1/qwen_encoder.py", "videox_fun/data/dataset_insertion_fast.py",
                 _model_file, "videox_fun/models/attention_utils.py",
                 "videox_fun/models/wan_vae.py", "videox_fun/models/wan_text_encoder.py", "videox_fun/utils/discrete_sampler.py"):
        with open(os.path.join(_repo_root, _rel), "rb") as _f:
            _code.update(_f.read())
    # the model file in use and its switches are part of what must agree on all ranks
    _code.update(f"|{_model_file}|{os.environ.get('SI_FAST_OPTS', '')}".encode("utf-8"))
    _code_fp = int.from_bytes(_code.digest()[:7], "big")
    _args_fp = int.from_bytes(hashlib.sha1(json.dumps(
        {k: v for k, v in sorted(vars(args).items()) if k not in ("local_rank",)}, sort_keys=True, default=str).encode("utf-8")).digest()[:7], "big")
    from videox_fun.models import attention_utils as _attention_utils
    _fa3 = int(bool(getattr(_attention_utils, "FLASH_ATTN_3_AVAILABLE", False)))   # the attention kernel must be the same on all ranks
    _mine_fp = torch.tensor([dataset_fingerprint, len(train_dataset), _plan_fp, _code_fp, _args_fp, _fa3], dtype=torch.int64, device=accelerator.device)
    _all_fp = accelerator.gather(_mine_fp.unsqueeze(0))
    if not bool((_all_fp == _all_fp[0:1]).all()):
        raise RuntimeError(f"[rank {accelerator.process_index}] ranks disagree on (dataset fingerprint, dataset length, sampling plan, "
                           f"code, arguments, FlashAttention-3 available); rows are ranks: {_all_fp.tolist()}")
    accelerator.print(f"[ATTN] attention kernel: {'FlashAttention-3' if _fa3 else 'FlashAttention-2'} on all ranks")
    accelerator.print(f"[MODEL] model file: {_model_file}" + (f" (SI_FAST_OPTS={os.environ.get('SI_FAST_OPTS', 'all')})" if SI_FAST_MODEL else ""))
    if args.require_fa3 and not _fa3:
        # every rank takes this branch (the flag was compared across ranks above)
        raise RuntimeError("--require_fa3 was given but flash_attn_interface (FlashAttention-3) is not importable: "
                           "put the FlashAttention-3 site directory on PYTHONPATH on every node")
    mark_progress("dataset built and compared across ranks")
    accelerator.print(f"[DATASET] {len(train_dataset)} items ({len(train_dataset.video_item_ids)} video segments, "
                      f"{len(train_dataset.image_item_ids)} single-frame images); identical on all {accelerator.num_processes} ranks "
                      f"(fingerprint {dataset_fingerprint:x}; sampling plan, code and arguments also identical); "
                      f"{len(batch_sampler)} global steps per epoch, group_by_type={args.group_by_type}")

    # Scheduler and math around the number of training steps.
    # [FAST] the position math below assumes one optimizer step per loader batch.
    if args.gradient_accumulation_steps != 1:
        raise ValueError("finetune_fast.py supports --gradient_accumulation_steps=1 only")
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    # [FAST] same call as the original minus the dataloader (see above).
    transformer3d, optimizer, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, lr_scheduler
    )

    # ------------------------------------------------------------------------------------------------
    # [FAST] Frozen encoders. The original (--low_vram) moved the VAE, Qwen3-VL and T5 between CPU and GPU
    # several times per step (about 54 GiB of pageable copies plus four empty_cache() calls) and wrapped T5
    # in FSDP (a 9 GB all-gather across all ranks per step). Here every encoder keeps exactly the same
    # weights, dtype and inputs and only its placement changes:
    #   resident : weights stay on the GPU.
    #   swap     : weights stay in page-locked host memory; each step they are copied to the GPU on a side
    #              stream while the VAE encodes, used, and dropped before the transformer needs the memory.
    #   fsdp     : (T5 only) FSDP-sharded as in the original script.
    # ------------------------------------------------------------------------------------------------
    vae.to(accelerator.device, dtype=weight_dtype)

    t5_swap = None
    if not args.enable_text_encoder_in_dataloader:
        if args.t5_placement == "fsdp":
            from functools import partial
            from videox_fun.dist import set_multi_gpus_devices, shard_model
            shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype)
            text_encoder = shard_fn(text_encoder)
            text_encoder.to(accelerator.device, dtype=weight_dtype)
        elif args.t5_placement == "swap":
            text_encoder.to(dtype=weight_dtype)
            t5_swap = HostSwappedModule(text_encoder, accelerator.device, "t5")
        else:
            text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.qwen_placement == "resident" and args.t5_placement == "resident":
        accelerator.print("[WARN] Qwen3-VL and T5 both resident: a 33-frame 832x480 step peaked at 65.7 GiB allocated with Qwen "
                          "resident and T5 swapped; resident T5 adds 10.6 GiB, which does not leave room on an 80 GB card.")
    qwen_swap = None
    if args.qwen_placement == "swap":
        qwen_swap = HostSwappedModule(qwen_encoder, accelerator.device, "qwen")
    else:
        qwen_encoder = qwen_encoder.to(accelerator.device)

    if args.use_ema:
        ema_transformer3d.to(accelerator.device)

    if args.train_mode != "normal":
        clip_image_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        keys_to_pop = [k for k, v in tracker_config.items() if isinstance(v, list)]
        for k in keys_to_pop:
            tracker_config.pop(k)
            print(f"Removed tracker_config['{k}']")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # ------------------------------------------------------------------------------------------------
    # [FAST] Checkpoints.
    # There is no filesystem shared by the nodes, so accelerate's save_state / load_state cannot be used:
    # its sharded files reference each other across nodes, and its loader does not restore the AdamW moments.
    # Each rank therefore writes one self-contained file to its node-local output directory:
    #     checkpoint-N/rankRR.pt   fp32 shards of every FSDP flat parameter, the rank's optimizer state,
    #                              the lr scheduler state, the RNG states and the global step
    #     checkpoint-N/COMPLETE    written on every node once all ranks have finished
    # A resume uses the newest step that is complete on every rank and restores all of the above, and the
    # data position follows from the global step. Rank 0 also exports the consolidated bf16 weights
    #     checkpoint-N/diffusion_pytorch_model.safetensors (+ config.json, WEIGHTS_READY)
    # exactly as the original did (same get_state_dict call, same cast), but once per checkpoint instead of
    # twice, and the cast + file write run in a background thread while training continues.
    # ------------------------------------------------------------------------------------------------
    fsdp_units = [m for m in FSDP.fsdp_modules(transformer3d) if m._has_params] if args.use_fsdp else []
    if not fsdp_units:
        raise RuntimeError("finetune_fast.py checkpoints the FSDP shards: launch with accelerate launch --use_fsdp and a sharded strategy")
    for _unit in fsdp_units:
        if not _unit._handle.is_sharded(_unit._flat_param) or _unit._flat_param.dtype != torch.float32:
            raise RuntimeError("every FSDP unit must hold a sharded fp32 flat parameter (all parameters trainable, mixed precision on); "
                               f"found sharded={_unit._handle.is_sharded(_unit._flat_param)} dtype={_unit._flat_param.dtype}")
    mark_progress("model wrapped")
    rank_file_name = f"rank{accelerator.process_index:02d}.pt"
    export_threads = []

    def _checkpoint_dir(step):
        return os.path.join(args.output_dir, f"checkpoint-{step}")

    def _local_complete_steps():
        steps = []
        if os.path.isdir(args.output_dir):
            for d in os.listdir(args.output_dir):
                if not d.startswith("checkpoint-"):
                    continue
                try:
                    s = int(d.split("-")[1])
                except ValueError:
                    continue
                p = os.path.join(args.output_dir, d)
                if os.path.isfile(os.path.join(p, "COMPLETE")) and os.path.isfile(os.path.join(p, rank_file_name)):
                    steps.append(s)
        return sorted(steps)

    def _moment_checksums(states):
        """Sums of the AdamW first / second moments held by this rank (for the save / load self check)."""
        m = v = 0.0
        n = 0
        for st in states:
            if "exp_avg" in st:
                m += float(st["exp_avg"].double().sum())
                v += float(st["exp_avg_sq"].double().sum())
                n += 1
        return {"exp_avg": m, "exp_avg_sq": v, "count": n}

    def _close(a, b):
        return abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1e-30)

    def save_rank_state(step):
        """Collective: every rank writes its own resumable state for `step` to its node-local directory."""
        save_dir = _checkpoint_dir(step)
        os.makedirs(save_dir, exist_ok=True)
        mark_progress(f"saving step {step}")
        # a step directory that is being rewritten must not keep the markers of its previous content
        for _marker in ("COMPLETE", "WEIGHTS_READY"):
            try:
                os.remove(os.path.join(save_dir, _marker))
            except FileNotFoundError:
                pass
        t0 = time.time()
        flat_params = []
        flat_checksums = []
        for unit in fsdp_units:
            flat_param = unit._flat_param
            if not unit._handle.is_sharded(flat_param) or flat_param.dtype != torch.float32:
                raise RuntimeError(f"unexpected FSDP flat parameter state at save: sharded={unit._handle.is_sharded(flat_param)} dtype={flat_param.dtype}")
            flat_checksums.append(float(flat_param.detach().double().sum()))
            flat_params.append(flat_param.detach().to("cpu", copy=True))
        optim_state = optimizer.state_dict()
        moment_checksums = _moment_checksums(optim_state["state"].values())
        optim_state = {
            "state": {i: {k: (v.detach().to("cpu", copy=True) if torch.is_tensor(v) else v) for k, v in s.items()}
                      for i, s in optim_state["state"].items()},
            "param_groups": optim_state["param_groups"],
        }
        payload = {
            "format": 2,
            "global_step": int(step),
            "rank": accelerator.process_index,
            "world_size": accelerator.num_processes,
            "flat_params": flat_params,
            "optimizer": optim_state,
            "lr_scheduler": lr_scheduler.state_dict(),
            "torch_rng": torch_rng.get_state() if torch_rng is not None else None,
            "cuda_rng": torch.cuda.get_rng_state(accelerator.device),
            "cpu_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "dataset_fingerprint": dataset_fingerprint,
            "steps_per_epoch": len(batch_sampler),
            "seed": args.seed,
            "group_by_type": args.group_by_type,
            "flat_checksums": flat_checksums,
            "moment_checksums": moment_checksums,
        }
        tmp_path = os.path.join(save_dir, rank_file_name + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, os.path.join(save_dir, rank_file_name))
        del payload, flat_params, optim_state
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            # all ranks of this node have written their file: mark the checkpoint complete, then prune
            with open(os.path.join(save_dir, "COMPLETE"), "w") as f:
                json.dump({"global_step": int(step), "time": time.strftime("%F %T"), "world_size": accelerator.num_processes}, f)
            if args.checkpoints_total_limit is not None:
                done = [s for s in _local_complete_steps() if s <= step]
                keep = set(done[-args.checkpoints_total_limit:])
                for d in sorted(os.listdir(args.output_dir)):
                    if not d.startswith("checkpoint-"):
                        continue
                    try:
                        s = int(d.split("-")[1])
                    except ValueError:
                        continue
                    if s not in keep and s < step:
                        shutil.rmtree(os.path.join(args.output_dir, d), ignore_errors=True)
                        print(f"[CKPT] pruned {d} on node of rank {accelerator.process_index}", flush=True)
        accelerator.wait_for_everyone()
        mark_progress(f"saved step {step}")
        accelerator.print(f"[CKPT] step {step}: resumable state saved by all ranks in {time.time() - t0:.1f}s -> {save_dir}")

    def _write_weights_background(state_dict, save_dir, step):
        # runs in a thread on rank 0 only: report a failure and leave no partial file (the resumable state of this step
        # is already complete, and the next checkpoint exports again)
        try:
            _write_weights(state_dict, save_dir, step)
        except Exception as e:  # noqa: BLE001
            print(f"[CKPT] ERROR: writing the consolidated weights of step {step} failed: {type(e).__name__}: {e}", flush=True)
            try:
                os.remove(os.path.join(save_dir, "diffusion_pytorch_model.safetensors.tmp"))
            except OSError:
                pass

    def _write_weights(state_dict, save_dir, step):
        from safetensors.torch import save_file
        t0 = time.time()
        state_dict = {k: v.to(dtype=weight_dtype) for k, v in state_dict.items()}
        tmp_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors.tmp")
        save_file(state_dict, tmp_path, metadata={"format": "pt"})
        os.replace(tmp_path, os.path.join(save_dir, "diffusion_pytorch_model.safetensors"))
        config_src = os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer'), "config.json")
        if os.path.isfile(config_src):
            shutil.copyfile(config_src, os.path.join(save_dir, "config.json"))
        with open(os.path.join(save_dir, "WEIGHTS_READY"), "w") as f:
            json.dump({"global_step": int(step), "time": time.strftime("%F %T"), "tensors": len(state_dict)}, f)
        print(f"[CKPT] step {step}: consolidated weights written in {time.time() - t0:.1f}s -> {save_dir}/diffusion_pytorch_model.safetensors", flush=True)

    def export_weights(step, background=True):
        """Collective: gather the full fp32 state dict to rank 0 (as the original did) and write it as bf16 safetensors."""
        save_dir = _checkpoint_dir(step)
        os.makedirs(save_dir, exist_ok=True)
        for th in export_threads:
            th.join()
        export_threads.clear()
        t0 = time.time()
        mark_progress(f"exporting weights of step {step}")
        accelerate_state_dict = accelerator.get_state_dict(transformer3d, unwrap=True)
        if accelerator.is_main_process:
            accelerator.print(f"[CKPT] step {step}: full state dict gathered in {time.time() - t0:.1f}s")
            if background:
                th = threading.Thread(target=_write_weights_background, args=(accelerate_state_dict, save_dir, step), daemon=False)
                th.start()
                export_threads.append(th)
            else:
                _write_weights(accelerate_state_dict, save_dir, step)
        del accelerate_state_dict
        accelerator.wait_for_everyone()
        mark_progress(f"exported weights of step {step}")

    resume_recheck = {}

    def load_rank_state(step):
        path = os.path.join(_checkpoint_dir(step), rank_file_name)
        mark_progress(f"loading step {step}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["world_size"] != accelerator.num_processes or payload["rank"] != accelerator.process_index:
            raise RuntimeError(f"{path} was written by rank {payload['rank']} of {payload['world_size']}, cannot load on rank {accelerator.process_index} of {accelerator.num_processes}")
        if payload["dataset_fingerprint"] != dataset_fingerprint or payload["steps_per_epoch"] != len(batch_sampler) \
                or payload["seed"] != args.seed or payload["group_by_type"] != args.group_by_type:
            raise RuntimeError(f"{path}: dataset / sampler settings differ from the run that wrote the checkpoint; the data position would be wrong")
        if len(payload["flat_params"]) != len(fsdp_units):
            raise RuntimeError(f"{path}: {len(payload['flat_params'])} FSDP units saved, model has {len(fsdp_units)}")
        with torch.no_grad():
            for unit, saved in zip(fsdp_units, payload["flat_params"]):
                flat_param = unit._flat_param
                if flat_param.shape != saved.shape or flat_param.dtype != saved.dtype:
                    raise RuntimeError(f"{path}: flat parameter mismatch {tuple(saved.shape)}/{saved.dtype} vs {tuple(flat_param.shape)}/{flat_param.dtype}")
                flat_param.data.copy_(saved.to(flat_param.device))
                # With use_orig_params=True the model's original parameters are views into the flat parameter. After
                # accelerate's fp32 upcast they still point at the OLD bf16 storage (the stage-1 weights), and FSDP's
                # first pre-forward would write such stale views back over the flat parameter, silently undoing this
                # load. Re-point them at the fp32 flat parameter now (the call FSDP itself makes after sharding).
                unit._handle._use_sharded_views()
                if unit._handle._writeback_orig_params():
                    raise RuntimeError(f"{path}: FSDP still had a pending parameter write-back after the load")
                if any(p.dtype != torch.float32 for p in flat_param._params):
                    raise RuntimeError(f"{path}: original parameters are not fp32 views after the load")
        # Optimizer state is assigned directly (Optimizer.load_state_dict casts the moments to the dtype of the
        # parameters, which must not be relied on here); the moments stay fp32 as saved.
        inner = optimizer.optimizer
        params = [p for g in inner.param_groups for p in g["params"]]
        saved_opt = payload["optimizer"]
        saved_ids = [i for g in saved_opt["param_groups"] for i in g["params"]]
        if len(saved_ids) != len(params) or len(saved_opt["param_groups"]) != len(inner.param_groups):
            raise RuntimeError(f"{path}: optimizer layout mismatch ({len(saved_ids)} saved parameters vs {len(params)})")
        inner.state.clear()
        n_state = 0
        for i, p in zip(saved_ids, params):
            if i not in saved_opt["state"]:
                continue
            st = {}
            for k, v in saved_opt["state"][i].items():
                if torch.is_tensor(v) and k != "step":
                    if tuple(v.shape) != tuple(p.shape):
                        raise RuntimeError(f"{path}: optimizer state shape {tuple(v.shape)} vs parameter shard {tuple(p.shape)}")
                    st[k] = v.to(accelerator.device)
                else:
                    st[k] = v
            inner.state[p] = st
            n_state += 1
        for g, sg in zip(inner.param_groups, saved_opt["param_groups"]):
            for k, v in sg.items():
                if k != "params":
                    g[k] = v
        lr_scheduler.load_state_dict(payload["lr_scheduler"])
        # self check: what is now on the GPU must be exactly what was saved, in fp32
        if "flat_checksums" in payload:
            for unit, want in zip(fsdp_units, payload["flat_checksums"]):
                got = float(unit._flat_param.detach().double().sum())
                if not _close(got, want):
                    raise RuntimeError(f"{path}: model shard checksum mismatch after load ({got} vs {want})")
            got_m = _moment_checksums(inner.state.values())
            want_m = payload["moment_checksums"]
            if got_m["count"] != want_m["count"] or not _close(got_m["exp_avg"], want_m["exp_avg"]) or not _close(got_m["exp_avg_sq"], want_m["exp_avg_sq"]):
                raise RuntimeError(f"{path}: AdamW moment checksum mismatch after load ({got_m} vs {want_m})")
            # compared once more after the first forward/backward of the resumed run (see the training loop)
            resume_recheck["flat_checksums"] = list(payload["flat_checksums"])
        else:
            print(f"[RESUME] rank {accelerator.process_index}: {path} has no checksums (written by an older version); not verified", flush=True)
        bad_dtype = [k for st in inner.state.values() for k, v in st.items() if torch.is_tensor(v) and k != "step" and v.dtype != torch.float32]
        if bad_dtype:
            raise RuntimeError(f"{path}: AdamW moments are not fp32 after load: {bad_dtype[:3]}")
        if torch_rng is not None and payload["torch_rng"] is not None:
            torch_rng.set_state(payload["torch_rng"])
        torch.cuda.set_rng_state(payload["cuda_rng"], accelerator.device)
        torch.set_rng_state(payload["cpu_rng"])
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
        print(f"[RESUME] rank {accelerator.process_index}: loaded {path} ({len(fsdp_units)} FSDP units, optimizer state for {n_state} parameter shards; "
              f"model-shard and AdamW-moment checksums match, moments fp32, lr {inner.param_groups[0]['lr']:.3e})", flush=True)
        return int(payload["global_step"])

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    # Potentially load in the weights and states from a previous save
    # [FAST] "latest" = the newest step that is complete on EVERY rank (each node has its own directory).
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            resume_step = int(os.path.basename(os.path.normpath(args.resume_from_checkpoint)).split("-")[1])
            local_steps = [resume_step] if resume_step in _local_complete_steps() else []
        else:
            local_steps = _local_complete_steps()[-16:]
        _mine = torch.full((1, 16), -1, dtype=torch.int64, device=accelerator.device)
        if local_steps:
            _mine[0, :len(local_steps)] = torch.tensor(local_steps, dtype=torch.int64, device=accelerator.device)
        _all = accelerator.gather(_mine)
        _common = None
        for row in _all.tolist():
            have = {s for s in row if s >= 0}
            _common = have if _common is None else (_common & have)
        resume_step = max(_common) if _common else None
        _any_have = any(s >= 0 for row in _all.tolist() for s in row)

        if resume_step is None:
            if args.resume_from_checkpoint != "latest":
                raise RuntimeError(f"requested checkpoint {args.resume_from_checkpoint} is not complete on every rank")
            if _any_have and not args.allow_fresh_start:
                # every rank takes this branch (the decision comes from the gathered table)
                raise RuntimeError(
                    "some ranks hold complete checkpoints but no step is complete on every rank (rows are ranks, -1 = none): "
                    f"{[[s for s in row if s >= 0] for row in _all.tolist()]}. Restore the missing rank files of one common "
                    "step, or pass --allow_fresh_start to train from step 0.")
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist on any rank. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint-{resume_step}")
            global_step = load_rank_state(resume_step)
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                # checkpoints after the resume point belong to an abandoned continuation: keep them out of "latest" and pruning
                for d in sorted(os.listdir(args.output_dir)):
                    if d.startswith("checkpoint-") and d.split("-")[1].isdigit() and int(d.split("-")[1]) > global_step:
                        os.rename(os.path.join(args.output_dir, d), os.path.join(args.output_dir, f"stale-{d}-{time.strftime('%Y%m%d_%H%M%S')}"))
                        print(f"[RESUME] set aside {d} (later than the resume point) on the node of rank {accelerator.process_index}", flush=True)
            accelerator.print(f"[RESUME] all ranks restored step {global_step}: epoch {first_epoch}, step {global_step % num_update_steps_per_epoch} of {num_update_steps_per_epoch} in that epoch")
    accelerator.wait_for_everyone()

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.multi_stream and args.train_mode != "normal":
        # create extra cuda streams to speedup inpaint vae computation
        vae_stream_1 = torch.cuda.Stream()
        vae_stream_2 = torch.cuda.Stream()
    else:
        vae_stream_1 = None
        vae_stream_2 = None

    idx_sampling = QuietDiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)
    # Keep this call between the resume block and the first forward: FSDP's _apply re-registers the original
    # parameters as views of the (fp32, possibly just loaded) flat parameters. load_rank_state does the same explicitly.
    transformer3d = transformer3d.to(accelerator.device)
    n_replaced_items = 0

    # [FAST] bookkeeping for the progress line (rank 0) and the graceful stop file
    stop_file = os.path.join(args.output_dir, "STOP")
    # [FAST] `touch <output_dir>/SAVE_NOW` on either node: at the next progress line the run writes a normal checkpoint
    # (rank states + consolidated weights) at that step and keeps training; the file is renamed to SAVE_NOW.done-<step>.
    save_now_file = os.path.join(args.output_dir, "SAVE_NOW")
    save_requested = False
    steps_per_epoch = len(batch_sampler)
    step_time_avg = {False: None, True: None}        # running mean of the step time, by kind (video / image)
    main_loss_window = []                            # the last 100 values of the main loss, for the smoothed TensorBoard curve
    timing = {}
    stop_requested = False
    last_saved_step = initial_global_step if args.resume_from_checkpoint else -1
    train_start_time = time.time()
    torch.cuda.reset_peak_memory_stats(accelerator.device)

    def _sync_time():
        torch.cuda.synchronize(accelerator.device)
        return time.perf_counter()

    for epoch in range(first_epoch, args.num_train_epochs):
        if global_step >= args.max_train_steps:
            break

        train_loss_log = 0.0
        train_loss_img_log = 0.0
        # [FAST] deterministic data position: (epoch, step) instead of a per-epoch generator plus a saved offset
        start_step_in_epoch = (global_step % steps_per_epoch) if epoch == first_epoch else 0
        batch_sampler.set_position(epoch, start_step_in_epoch)
        if accelerator.is_main_process:
            image_step_flags = np.array([batch_sampler.step_is_image(epoch, s) for s in range(steps_per_epoch)], dtype=bool)
            print(f"[DATA] epoch {epoch}: starting at step {start_step_in_epoch} of {steps_per_epoch} "
                  f"({int(image_step_flags.sum())} image steps, {int((~image_step_flags).sum())} video steps)", flush=True)
        t_iter = time.perf_counter()
        for step, batch in enumerate(train_dataloader, start=start_step_in_epoch):
            # the step whose progress line is printed is the one that is timed phase by phase (with GPU syncs)
            profile_step = ((global_step + 1) % args.log_every == 0) or global_step == initial_global_step
            t_data = time.perf_counter() - t_iter
            # [FAST] data-position self check: the batch must be exactly the items planned for this rank and step
            expected_index = batch_sampler.rank_batch(epoch, step)
            if batch["index"].tolist() != expected_index:
                raise RuntimeError(f"[rank {accelerator.process_index}] data position error at epoch {epoch} step {step}: "
                                   f"got items {batch['index'].tolist()}, planned {expected_index}")
            step_is_image = bool(batch["is_image"][0])
            n_replaced_items += int((batch["used_index"] != batch["index"]).sum())   # items replaced after a load failure
            if global_step == initial_global_step:
                print(f"[DATA] rank {accelerator.process_index}: position check OK at global step {global_step} "
                      f"(epoch {epoch}, step {step}, items {expected_index}, {'image' if step_is_image else 'video'})", flush=True)

            with accelerator.accumulate(transformer3d):
                # Convert images to latent space
                # [FAST] move first, cast on the GPU (same values as casting on the CPU first)
                tar_pixel_values = batch["tar_pixel_values"].to(accelerator.device, non_blocking=True).to(weight_dtype)
                src_pixel_values = batch["src_pixel_values"].to(accelerator.device, non_blocking=True).to(weight_dtype)
                tar_image_tensor = batch["tar_image_tensor"].to(accelerator.device, non_blocking=True).to(weight_dtype)
                ref_image_tensor = batch["ref_image_tensor"].to(accelerator.device, non_blocking=True).to(weight_dtype)
                instruction = batch["instruction"]

                if profile_step:
                    t0 = _sync_time()

                # [FAST] start copying swapped encoders to the GPU; the VAE encodes meanwhile
                if qwen_swap is not None:
                    qwen_swap.load_async()
                if t5_swap is not None:
                    t5_swap.load_async()

                with torch.no_grad():
                    # This way is quicker when batch grows up
                    def _batch_encode_vae(pixel_values):
                        # A 4D image tensor [B, C, H, W] gets a temporal dimension F=1
                        if pixel_values.dim() == 4:
                            pixel_values = pixel_values.unsqueeze(1)
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i : i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim = 0)

                    tar_latents = _batch_encode_vae(tar_pixel_values)
                    src_latents = _batch_encode_vae(src_pixel_values)
                    tar_img_latents = _batch_encode_vae(tar_image_tensor)
                    ref_img_latents = _batch_encode_vae(ref_image_tensor)

                if profile_step:
                    t1 = _sync_time()
                    timing["vae"] = t1 - t0

                # ====== Get prompt embeddings based on encoder mode ======
                if args.qwen_encoder_mode is not None:
                    # Use Qwen encoder - obtain encoder_hidden_states directly
                    if qwen_swap is not None:
                        qwen_swap.wait()

                    # elif args.qwen_encoder_mode == "edit":
                    ref_img_raw = batch["ref_image_raw"].cpu().numpy()
                    images_for_model = [[Image.fromarray(img).convert("RGB")] for img in ref_img_raw]
                    raw_videos_frist_frame = batch["src_first_frame_raw"].cpu().numpy()
                    src_first_frame_for_model = [[Image.fromarray(img).convert("RGB")] for img in raw_videos_frist_frame]

                    edit_embeds_list = []
                    for i in range(len(batch['text'])):
                        emb_i, _ = qwen_encoder.get_prompt_embeddings(
                            prompts=[""],
                            prompt2sys=instruction[i],
                            images=[images_for_model[i]],
                            second_image=[src_first_frame_for_model[i]],
                            device=tar_latents.device,
                            dtype=weight_dtype,
                            mode="edit",
                        )
                        edit_embeds_list.append(emb_i)
                    prompt_embeds_qwen = torch.cat(edit_embeds_list, dim=0).to(accelerator.device).to(weight_dtype)

                    ref_prompt = ["" for _ in batch['text']]
                    ref_prompt_embeds_qwen, _ = qwen_encoder.get_prompt_embeddings(
                        prompts=ref_prompt,
                        images=images_for_model,
                        second_image=src_first_frame_for_model,
                        device=tar_latents.device,
                        dtype=weight_dtype,
                        mode="ref_edit",
                    )
                    ref_prompt_embeds_qwen = ref_prompt_embeds_qwen.to(accelerator.device)
                    ref_prompt_embeds_qwen = ref_prompt_embeds_qwen.to(weight_dtype)

                    if qwen_swap is not None:
                        qwen_swap.unload()

                if profile_step:
                    t2 = _sync_time()
                    timing["qwen"] = t2 - t1

                if t5_swap is not None:
                    t5_swap.wait()
                with torch.no_grad():
                    prompt_ids = tokenizer(
                        batch['text'],
                        padding="max_length",
                        max_length=args.tokenizer_max_length,
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors="pt"
                    )
                    text_input_ids = prompt_ids.input_ids
                    prompt_attention_mask = prompt_ids.attention_mask
                    seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                    prompt_embeds = text_encoder(text_input_ids.to(tar_latents.device), attention_mask=prompt_attention_mask.to(tar_latents.device))[0]
                    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
                if t5_swap is not None:
                    t5_swap.unload()

                # ==========================================================
                # GPU memory optimization: release the raw high-dimensional tensors immediately after the text and VAE latents are computed
                # ==========================================================
                del tar_pixel_values, src_pixel_values, tar_image_tensor, ref_image_tensor
                del images_for_model, src_first_frame_for_model

                if profile_step:
                    t3 = _sync_time()
                    timing["t5"] = t3 - t2

                bsz, channel, num_frames, height, width = tar_latents.size()
                noise = torch.randn(tar_latents.size(), device=tar_latents.device, generator=torch_rng, dtype=weight_dtype)

                noise2 = torch.randn(tar_img_latents.size(), device=tar_img_latents.device, generator=torch_rng, dtype=weight_dtype)

                if not args.uniform_sampling:
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                else:
                    # Sample a random timestep for each image
                    # timesteps = generate_timestep_with_lognorm(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    # timesteps = torch.randint(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    indices = idx_sampling(bsz, generator=torch_rng, device=tar_latents.device)
                    indices = indices.long().cpu()
                timesteps = noise_scheduler.timesteps[indices].to(device=tar_latents.device)

                def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

                    sigma = sigmas[step_indices].flatten()
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=tar_latents.ndim, dtype=tar_latents.dtype)
                noisy_latents = (1.0 - sigmas) * tar_latents + sigmas * noise
                noisy_latents = noisy_latents.to(accelerator.device)


                noisy_latents2 = (1.0 - sigmas) * tar_img_latents + sigmas * noise2
                noisy_latents2 = noisy_latents2.to(accelerator.device)

                # Add noise
                target = noise - tar_latents

                target2 = noise2 - tar_img_latents

                target_shape = (vae.latent_channels, num_frames, width, height)


                # Predict the noise residual
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    noise_pred,noise_pred_img = transformer3d(
                        tar_latents=noisy_latents,
                        tar_img_latents=noisy_latents2,
                        src_vid_latents=src_latents,
                        ref_img_latents=ref_img_latents,

                        context=prompt_embeds,
                        context_qwen=prompt_embeds_qwen,
                        ref_context_qwen=ref_prompt_embeds_qwen,
                        t=timesteps,

                    )

                if profile_step:
                    t4 = _sync_time()
                    timing["fwd"] = t4 - t3

                def custom_mse_loss(noise_pred, noise_pred_img, target, target2, weighting=None, threshold=50):
                    noise_pred = noise_pred.float()
                    noise_pred_img = noise_pred_img.float()
                    target = target.float()
                    target2 = target2.float()
                    diff = noise_pred - target
                    diff_img = noise_pred_img - target2
                    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
                    mse_loss_img = F.mse_loss(noise_pred_img, target2, reduction='none')
                    mask = (diff.abs() <= threshold).float()
                    masked_loss = mse_loss * mask
                    mask2 = (diff_img.abs() <= threshold).float()
                    masked_loss_img = mse_loss_img * mask2
                    if weighting is not None:
                        masked_loss = masked_loss * weighting
                        masked_loss_img = masked_loss_img * weighting
                    final_loss = masked_loss.mean()
                    final_loss_img = masked_loss_img.mean()
                    return final_loss,final_loss_img

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss,loss_img = custom_mse_loss(noise_pred.float(), noise_pred_img.float(), target.float(), target2.float(), weighting.float())
                loss = loss.mean()
                loss_img = loss_img.mean()

                if args.motion_sub_loss and noise_pred.size()[2] > 2:
                    gt_sub_noise = noise_pred[:, :, 1:].float() - noise_pred[:, :, :-1].float()
                    pre_sub_noise = target[:, :, 1:].float() - target[:, :, :-1].float()
                    sub_loss = F.mse_loss(gt_sub_noise, pre_sub_noise, reduction="mean")
                    loss = loss * (1 - args.motion_sub_loss_ratio) + sub_loss * args.motion_sub_loss_ratio

                ref_weight = 0.3
                loss_total = (1-ref_weight)*loss + ref_weight*loss_img
                accelerator.backward(loss_total)

                if resume_recheck:
                    # [FAST] first forward/backward after a resume: the shards must still be exactly what was loaded
                    for unit, want in zip(fsdp_units, resume_recheck.pop("flat_checksums")):
                        got = float(unit._flat_param.detach().double().sum())
                        if not _close(got, want):
                            raise RuntimeError(f"[rank {accelerator.process_index}] the first forward/backward after the resume changed "
                                               f"the loaded model shards ({got} vs {want})")
                    print(f"[RESUME] rank {accelerator.process_index}: model shards still equal the checkpoint after the first forward/backward", flush=True)

                if profile_step:
                    t5 = _sync_time()
                    timing["bwd"] = t5 - t4

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.detach().repeat(args.train_batch_size)).mean()
                avg_loss_img = accelerator.gather(loss_img.detach().repeat(args.train_batch_size)).mean()
                train_loss_log += avg_loss.item() / args.gradient_accumulation_steps
                train_loss_img_log += avg_loss_img.item() / args.gradient_accumulation_steps
                train_loss_total_log = (1-ref_weight)*train_loss_log + ref_weight*train_loss_img_log
                # Backpropagate

                if accelerator.sync_gradients:

                    if not args.use_deepspeed and not args.use_fsdp:
                        trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                        trainable_params_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                        max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                        if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                            actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                        else:
                            actual_max_grad_norm = max_grad_norm
                    else:
                        actual_max_grad_norm = args.max_grad_norm

                    if not args.use_deepspeed and not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        if trainable_params_total_norm > 1 and global_step > args.abnormal_norm_clip_start:
                            for name, param in transformer3d.named_parameters():
                                if param.requires_grad:
                                    writer.add_scalar(f'gradients/before_clip_norm/{name}', param.grad.norm(), global_step=global_step)

                    norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)
                    # if not args.use_deepspeed and not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                    if accelerator.is_main_process:
                        if norm_sum is not None:
                            writer.add_scalar(f'gradients/norm_sum', norm_sum, global_step=global_step)
                        if actual_max_grad_norm is not None:
                            writer.add_scalar(f'gradients/actual_max_grad_norm', actual_max_grad_norm, global_step=global_step)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if profile_step:
                    t6 = _sync_time()
                    timing["opt"] = t6 - t5

            # Checks if the accelerator has performed an optimization step behind the scenes
            logs = {"total_loss": train_loss_total_log, "video_loss": train_loss_log, "image_loss": train_loss_img_log, "lr": lr_scheduler.get_last_lr()[0]}
            if accelerator.sync_gradients:

                if args.use_ema:
                    ema_transformer3d.step(transformer3d.parameters())
                progress_bar.update(1)
                global_step += 1
                mark_progress(f"step {global_step}")
                step_seconds = time.perf_counter() - t_iter   # includes the wait for the batch
                # [FAST] one tracker call per step instead of three (same tags as the original, plus the step kind)
                # main_loss/* is the loss that is optimised (0.7 * video + 0.3 * image, mean over all ranks): per step
                # and as the mean of the last 100 steps (video and image steps have different levels, so the per-step
                # curve is noisy).
                main_loss_window.append(train_loss_total_log)
                del main_loss_window[:-100]
                accelerator.log({
                    "main_loss/total": train_loss_total_log,
                    "main_loss/total_avg100": sum(main_loss_window) / len(main_loss_window),
                    "loss/train_loss_vid": train_loss_log,
                    "loss/train_loss_img": train_loss_img_log,
                    "Total_Loss/train_loss_total": train_loss_total_log,
                    ("loss_by_kind/image_step_total" if step_is_image else "loss_by_kind/video_step_total"): train_loss_total_log,
                    "data/is_image_step": float(step_is_image),
                    "time/step_seconds": step_seconds,
                }, step=global_step)

                # [FAST] progress line, memory statistics and the graceful stop file (one small collective per log_every steps)
                prev = step_time_avg[step_is_image]
                step_time_avg[step_is_image] = step_seconds if prev is None else 0.9 * prev + 0.1 * step_seconds
                if global_step % args.log_every == 0 or global_step == initial_global_step + 1:
                    stats = torch.tensor([
                        1.0 if os.path.exists(stop_file) else 0.0,
                        torch.cuda.max_memory_allocated(accelerator.device) / 2**30,
                        torch.cuda.max_memory_reserved(accelerator.device) / 2**30,
                        step_seconds,
                        float(n_replaced_items),
                        1.0 if os.path.exists(save_now_file) else 0.0,
                    ], device=accelerator.device, dtype=torch.float64)
                    if accelerator.num_processes > 1:
                        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)
                    stop_requested = bool(stats[0] > 0)
                    save_requested = bool(stats[5] > 0)      # the same value on every rank (it comes from the reduce)
                    if accelerator.is_main_process:
                        remaining_in_epoch = image_step_flags[step + 1:]
                        epochs_left = args.num_train_epochs - epoch - 1
                        n_img_left = int(remaining_in_epoch.sum()) + epochs_left * int(image_step_flags.sum())
                        n_vid_left = int((~remaining_in_epoch).sum()) + epochs_left * int((~image_step_flags).sum())
                        left_total = args.max_train_steps - global_step
                        scale = left_total / max(n_img_left + n_vid_left, 1)
                        t_vid = step_time_avg[False] if step_time_avg[False] is not None else step_time_avg[True]
                        t_img = step_time_avg[True] if step_time_avg[True] is not None else step_time_avg[False]
                        eta = (n_vid_left * t_vid + n_img_left * t_img) * scale
                        # "vae" also contains the wait for the host-to-GPU copies of swapped encoders (they overlap)
                        phase = " ".join(f"{'vae+copy' if k == 'vae' else k} {timing[k]:.2f}" for k in ("vae", "qwen", "t5", "fwd", "bwd", "opt") if k in timing)
                        print(f"[TRAIN] step {global_step}/{args.max_train_steps} epoch {epoch} ({step + 1}/{steps_per_epoch}) "
                              f"{'image' if step_is_image else 'video'} | loss total {train_loss_total_log:.4f} vid {train_loss_log:.4f} img {train_loss_img_log:.4f} "
                              f"| lr {lr_scheduler.get_last_lr()[0]:.2e} | {step_seconds:.2f} s/step (data wait {t_data:.2f}; profiled: {phase}) "
                              f"| avg video {t_vid:.2f}s image {t_img:.2f}s | peak mem alloc {stats[1].item():.1f} reserved {stats[2].item():.1f} GiB "
                              f"| replaced items (worst rank) {int(stats[4].item())} "
                              f"| elapsed {(time.time() - train_start_time) / 3600:.2f} h | ETA {eta / 3600:.1f} h", flush=True)
                        accelerator.log({
                            "mem/peak_allocated_gib": stats[1].item(), "mem/peak_reserved_gib": stats[2].item(),
                            "data/replaced_items_worst_rank": stats[4].item(),
                            **{f"time/{k}": v for k, v in timing.items()},
                        }, step=global_step)

                train_loss_log = 0.0
                train_loss_img_log = 0.0
                train_loss_total_log = 0.0
                if global_step % args.checkpointing_steps == 0 or stop_requested or save_requested or global_step >= args.max_train_steps:
                    # [FAST] node-local resumable state on every rank + consolidated weights on rank 0
                    gc.collect()
                    save_rank_state(global_step)
                    last_saved_step = global_step
                    final = stop_requested or global_step >= args.max_train_steps
                    export_weights(global_step, background=not final)
                    if save_requested:
                        # the request has been honoured: consume it on the node that holds the file
                        if accelerator.is_local_main_process and os.path.exists(save_now_file):
                            os.replace(save_now_file, save_now_file + f".done-{global_step}")
                            print(f"[CKPT] on-demand checkpoint written at step {global_step} (SAVE_NOW consumed on the node of rank {accelerator.process_index})", flush=True)
                        save_requested = False

            # logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            t_iter = time.perf_counter()

            if global_step >= args.max_train_steps or stop_requested:
                break

        # [FAST] leave the epoch loop as well (the original only left the inner loop and ran extra steps)
        if global_step >= args.max_train_steps or stop_requested:
            break

    # [FAST] the original script never saved after the loop; here the last step is always saved (above), and
    # the export thread is joined so the consolidated weights are complete before the process exits.
    for th in export_threads:
        th.join()
    # the consolidated weights of the last saved step must exist (a previous run may have died while exporting them)
    _need_export = torch.zeros(1, device=accelerator.device)
    if accelerator.is_main_process and last_saved_step >= 0 and not os.path.isfile(os.path.join(_checkpoint_dir(last_saved_step), "WEIGHTS_READY")):
        _need_export += 1
    if accelerator.num_processes > 1:
        torch.distributed.all_reduce(_need_export, op=torch.distributed.ReduceOp.MAX)
    if bool(_need_export[0] > 0):
        accelerator.print(f"[CKPT] consolidated weights of step {last_saved_step} are missing; exporting them now")
        export_weights(last_saved_step, background=False)
    if stop_requested and accelerator.is_local_main_process and os.path.exists(stop_file):
        # the stop request has been honoured: a relaunch must be able to continue
        os.replace(stop_file, stop_file + f".consumed-{global_step}")
    if accelerator.is_main_process:
        final_dir = _checkpoint_dir(last_saved_step)
        with open(os.path.join(args.output_dir, "FINAL.json"), "w") as f:
            json.dump({
                "global_step": int(global_step), "max_train_steps": int(args.max_train_steps),
                "steps_per_epoch": int(steps_per_epoch), "num_train_epochs": int(args.num_train_epochs),
                "items": int(len(train_dataset)), "video_segments": int(len(train_dataset.video_item_ids)),
                "single_frame_images": int(len(train_dataset.image_item_ids)),
                "stopped_by_stop_file": bool(stop_requested),
                "finished": bool(global_step >= args.max_train_steps),
                "checkpoint": final_dir,
                "weights": os.path.join(final_dir, "diffusion_pytorch_model.safetensors"),
                "time": time.strftime("%F %T"),
            }, f, indent=2)
        print(f"[DONE] training ended at step {global_step}/{args.max_train_steps} "
              f"({'stop file' if stop_requested else 'finished'}); final weights: {final_dir}/diffusion_pytorch_model.safetensors", flush=True)

    accelerator.end_training()

if __name__ == "__main__":
    main()
