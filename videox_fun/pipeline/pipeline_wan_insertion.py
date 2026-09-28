import copy
import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from PIL import Image

from ..models import (AutoencoderKLWan, AutoTokenizer,
                              WanT5EncoderModel, WanTransformer3DModel)
from ..utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                get_sampling_sigmas)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class WanPipelineOutput(BaseOutput):
    videos: torch.Tensor


class WanPipeline_insertion_loop(DiffusionPipeline):
    r"""
    Pipeline for video-insertion generation using Wan with dual-stream denoising
    and closed-loop Qwen feedback.

    Every step denoises tar_latents (video stream) and tar_img_latents (image stream).
    At each eligible step (sigma in [denoise_lower, denoise_upper], step index a multiple
    of qwen_reembed_interval, not the last step):
      1. Predict the image stream's clean latent x0 = z_t - sigma * v_pred and decode it
      2. Pass the decoded image to the callback (it is always saved)
      3. At up to max_loop evenly spaced eligible steps, the callback re-encodes it with
         Qwen; the new Qwen embeddings and ref_img_latents = x0 are used for the
         remaining steps. The denoising trajectory itself is not restarted.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: WanTransformer3DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
            transformer=transformer, scheduler=scheduler,
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)

    # ------------------------------------------------------------------
    # T5 prompt encoding
    # ------------------------------------------------------------------
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    # ------------------------------------------------------------------
    # Latent helpers
    # ------------------------------------------------------------------
    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = frames.cpu().float().numpy()
        return frames

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    # ------------------------------------------------------------------
    # Main __call__
    # ------------------------------------------------------------------
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # ----- insertion-specific inputs (pre-encoded) -----
        src_vid_latents: Optional[torch.FloatTensor] = None,   # [B, C, F_lat, H_lat, W_lat]
        ref_img_latents: Optional[torch.FloatTensor] = None,   # [B, C, 1,     H_lat, W_lat]
        qwen_embed: Optional[torch.FloatTensor] = None,        # [B, L, D]
        ref_qwen_embed: Optional[torch.FloatTensor] = None,    # [B, L, D]
        # ----- loop feedback -----
        # callback(decoded_pil, step_idx, do_reembed) -> (qwen_embed, ref_qwen_embed) or (None, None)
        qwen_reembed_callback: Optional[Callable] = None,
        qwen_reembed_interval: int = 1,
        denoise_lower: float = 0.0,   # sigma range lower bound (0=clean, 1=noise)
        denoise_upper: float = 1.0,   # sigma range upper bound
        max_loop: int = 5,            # max Qwen re-encoding calls (evenly spaced within range)
        # ----- standard diffusion params -----
        height: int = 480,
        width: int = 832,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        comfyui_progressbar: bool = False,
        shift: int = 1,
    ) -> Union[WanPipelineOutput, Tuple]:
        r"""
        Insertion pipeline with closed-loop Qwen feedback.

        Feedback parameters:
          - qwen_reembed_callback: callable(decoded_pil, step_idx, do_reembed)
              -> (qwen_embed, ref_qwen_embed), or (None, None) when do_reembed is False.
              Called at every eligible step with the decoded x0 estimate of the image stream.
              If None, no feedback is applied.
          - qwen_reembed_interval: int (default 1)
              Only step indices that are multiples of this are eligible; <= 0 disables the loop.
          - denoise_lower / denoise_upper: sigma window of eligible steps (1 = noise, 0 = clean).
          - max_loop: maximum number of evenly spaced eligible steps that re-encode with Qwen;
              0 only saves the decoded estimates, a negative value re-encodes at every eligible step.

        Examples:

        Returns:
            WanPipelineOutput with .videos tensor.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        # 1. Check inputs
        self.check_inputs(
            prompt, height, width, negative_prompt,
            callback_on_step_end_tensor_inputs, prompt_embeds, negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype
        do_classifier_free_guidance = guidance_scale > 1.0

        # ==================================================================
        # 3. Encode T5 prompt
        # ==================================================================
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, negative_prompt, do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        if do_classifier_free_guidance:
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds   # list concat
        else:
            in_prompt_embeds = prompt_embeds

        # ==================================================================
        # 4. Prepare Qwen embeddings
        # ==================================================================
        qwen_embed = qwen_embed.to(device=device, dtype=weight_dtype)
        ref_qwen_embed = ref_qwen_embed.to(device=device, dtype=weight_dtype)

        if do_classifier_free_guidance:
            # Qwen embeddings always present for both uncond/cond branches.
            # CFG difference is only in T5 text embeddings.
            in_qwen_embed = torch.cat([qwen_embed, qwen_embed], dim=0)
            in_ref_qwen_embed = torch.cat([ref_qwen_embed, ref_qwen_embed], dim=0)
        else:
            in_qwen_embed = qwen_embed
            in_ref_qwen_embed = ref_qwen_embed

        # ==================================================================
        # 5. Move condition latents to device
        # ==================================================================
        src_vid_latents = src_vid_latents.to(device=device, dtype=weight_dtype)
        ref_img_latents = ref_img_latents.to(device=device, dtype=weight_dtype)

        # ==================================================================
        # 6. Prepare timesteps
        # ==================================================================
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(self.scheduler, device=device, sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        scheduler_img = copy.deepcopy(self.scheduler)

        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 1)

        # ==================================================================
        # 7. Prepare noise latents - target video
        # ==================================================================
        latent_channels = self.transformer.config.in_channels
        tar_latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels, num_frames, height, width,
            weight_dtype, device, generator, latents,
        )

        # ==================================================================
        # 8. Prepare noise latents - target image (single frame)
        # ==================================================================
        tar_img_shape = (
            batch_size * num_videos_per_prompt,
            latent_channels,
            1,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )
        tar_img_latents = randn_tensor(tar_img_shape, generator=generator, device=device, dtype=weight_dtype)
        if hasattr(self.scheduler, "init_noise_sigma"):
            tar_img_latents = tar_img_latents * self.scheduler.init_noise_sigma

        if comfyui_progressbar:
            pbar.update(1)

        # ==================================================================
        # 9. Extra step kwargs
        # ==================================================================
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # ==================================================================
        # 10. Pre-compute loop schedule
        # ==================================================================
        # eligible_steps: sigma in range, not last step, respects interval
        #   -> ALL get x0 decoded & saved
        # reembed_steps: evenly-spaced subset of eligible, capped by max_loop
        #   -> ALSO call Qwen re-encoding & update ref_img_latents
        eligible_steps = []
        if qwen_reembed_callback is not None:
            for step_i in range(len(timesteps) - 1):           # exclude last step
                sigma_i = self.scheduler.sigmas[step_i].item()
                if (denoise_lower <= sigma_i <= denoise_upper
                        and qwen_reembed_interval > 0
                        and step_i % qwen_reembed_interval == 0):
                    eligible_steps.append(step_i)

        if max_loop == 0:
            reembed_steps = set()                          # 0 = save only, no Qwen
        elif max_loop > 0 and len(eligible_steps) > max_loop:
            pick_idx = np.round(np.linspace(0, len(eligible_steps) - 1, max_loop)).astype(int)
            reembed_steps = set(eligible_steps[j] for j in pick_idx)
        else:
            reembed_steps = set(eligible_steps)            # all fit or no limit
        eligible_steps_set = set(eligible_steps)

        if eligible_steps:
            print(f"  [loop] eligible steps ({len(eligible_steps)}): {eligible_steps}")
            print(f"  [loop] reembed  steps ({len(reembed_steps)}): {sorted(reembed_steps)}")

        # ==================================================================
        # 11. Denoising loop
        # ==================================================================
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.transformer.current_steps = i

                if self.interrupt:
                    continue

                # --- CFG: double all inputs along batch dim ---
                if do_classifier_free_guidance:
                    tar_input       = torch.cat([tar_latents] * 2)
                    tar_img_input   = torch.cat([tar_img_latents] * 2)
                    src_input       = torch.cat([src_vid_latents] * 2)
                    ref_input       = torch.cat([ref_img_latents] * 2)
                else:
                    tar_input       = tar_latents
                    tar_img_input   = tar_img_latents
                    src_input       = src_vid_latents
                    ref_input       = ref_img_latents

                timestep = t.expand(tar_input.shape[0])

                # --- Transformer forward ---
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    noise_pred_vid, noise_pred_img = self.transformer(
                        tar_latents=tar_input,
                        tar_img_latents=tar_img_input,
                        src_vid_latents=src_input,
                        ref_img_latents=ref_input,
                        t=timestep,
                        context=in_prompt_embeds,
                        context_qwen=in_qwen_embed,
                        ref_context_qwen=in_ref_qwen_embed,
                    )

                # --- CFG combine: video only ---
                # Image branch does NOT use CFG; CFG = T5 text difference only.
                if do_classifier_free_guidance:
                    uncond_vid, cond_vid = noise_pred_vid.chunk(2)
                    noise_pred_vid = uncond_vid + self.guidance_scale * (cond_vid - uncond_vid)

                    # Image: no CFG, take the conditional (second) half only
                    _, noise_pred_img = noise_pred_img.chunk(2)

                # --- Save z_t before scheduler step (for x0 prediction) ---
                tar_img_latents_pre_step = tar_img_latents

                # --- Scheduler step: target video ---
                tar_latents = self.scheduler.step(
                    noise_pred_vid, t, tar_latents, **extra_step_kwargs, return_dict=False
                )[0]

                # --- Scheduler step: target image ---
                tar_img_latents = scheduler_img.step(
                    noise_pred_img, t, tar_img_latents, **extra_step_kwargs, return_dict=False
                )[0]

                # ==============================================================
                # Loop feedback (pre-computed schedule):
                #   eligible_steps_set -> decode x0, save image (ALL)
                #   reembed_steps      -> also Qwen re-encode + update conds
                # ==============================================================
                if i in eligible_steps_set:
                    current_sigma = self.scheduler.sigmas[i].item()
                    do_reembed = (i in reembed_steps)

                    # One-step predict clean x0 via flow matching:
                    #   z_t = (1 - sigma) * x0 + sigma * noise
                    #   v_pred = model_output = noise - x0
                    #   => x0 = z_t - sigma * v_pred
                    sigma_t = torch.tensor(current_sigma, device=device, dtype=weight_dtype)
                    x0_pred_img = tar_img_latents_pre_step - sigma_t * noise_pred_img

                    # Decode predicted clean x0 to pixel space
                    decoded = self.vae.decode(x0_pred_img.to(self.vae.dtype)).sample
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)

                    if decoded.ndim == 5:
                        frame = decoded[0, :, 0]   # [C, H, W]
                    else:
                        frame = decoded[0]          # [C, H, W]
                    frame_np = (frame * 255).to(torch.uint8).cpu().permute(1, 2, 0).numpy()
                    decoded_pil = Image.fromarray(frame_np)

                    del decoded, frame

                    # Callback: always saves image; only Qwen re-encodes when do_reembed
                    new_qwen_embed, new_ref_qwen_embed = qwen_reembed_callback(
                        decoded_pil, i, do_reembed)

                    if do_reembed:
                        new_qwen_embed = new_qwen_embed.to(device=device, dtype=weight_dtype)
                        new_ref_qwen_embed = new_ref_qwen_embed.to(device=device, dtype=weight_dtype)

                        # Rebuild CFG-doubled Qwen embeddings (replicated, not zeroed)
                        if do_classifier_free_guidance:
                            in_qwen_embed = torch.cat(
                                [new_qwen_embed, new_qwen_embed], dim=0)
                            in_ref_qwen_embed = torch.cat(
                                [new_ref_qwen_embed, new_ref_qwen_embed], dim=0)
                        else:
                            in_qwen_embed = new_qwen_embed
                            in_ref_qwen_embed = new_ref_qwen_embed

                        # Update ref_img_latents with predicted clean x0 latent
                        ref_img_latents = x0_pred_img.detach().clone()

                        print(f"  [loop] step {i}, sigma={current_sigma:.3f}: "
                              f"x0 + Qwen re-encoded")
                    else:
                        print(f"  [loop] step {i}, sigma={current_sigma:.3f}: "
                              f"x0 saved (skip Qwen)")

                # --- Callbacks ---
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    tar_latents = callback_outputs.pop("latents", tar_latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # --- Progress ---
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if comfyui_progressbar:
                    pbar.update(1)

        # ==================================================================
        # 11. Decode target video latents -> pixel space
        # ==================================================================
        if output_type == "numpy":
            video = self.decode_latents(tar_latents)
        elif output_type != "latent":
            video = self.decode_latents(tar_latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = tar_latents

        self.maybe_free_model_hooks()

        if not return_dict:
            video = torch.from_numpy(video) if isinstance(video, np.ndarray) else video

        return WanPipelineOutput(videos=video)
