import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, WanT5EncoderModel, AutoTokenizer)
from videox_fun.models.wan_transformer3d_insertion_new import WanTransformer3DModel
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline.pipeline_wan_insertion import WanPipeline_insertion_loop
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import filter_kwargs, save_videos_grid
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from qwen_encoder import MLLMInContext, MLLMInContextConfig
if not hasattr(torch, "int1"):
    torch.int1 = torch.int8

# ==================== Helper Functions ====================

def get_first_frame_as_pil(video_path: str) -> Image.Image:
    """Extract the first frame of a video as a PIL Image (RGB)."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read video: {video_path}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def load_video_frames(video_path: str, sample_size, video_length: int) -> torch.Tensor:
    """
    Load video frames, resize, normalize to [-1, 1].
    Matches training InsertionDataset's src_pixel_values preprocessing.

    Returns:
        [F, C, H, W] float32 tensor in [-1, 1]
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (sample_size[1], sample_size[0]), interpolation=cv2.INTER_LINEAR)
        frames.append(frame_resized)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Cannot read video: {video_path}")

    # Pad with last frame or truncate — same as InsertionDataset
    if len(frames) < video_length:
        last_frame = frames[-1]
        frames.extend([last_frame] * (video_length - len(frames)))
    else:
        frames = frames[:video_length]

    frames_np = np.array(frames)  # [F, H, W, C] uint8
    tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous().float() / 255.0  # [F, C, H, W]
    # Normalize to [-1, 1] — same as training video_transforms (Normalize(0.5, 0.5))
    tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(tensor)
    return tensor


def load_reference_image(image_path: str, sample_size) -> torch.Tensor:
    """
    Load image, resize, normalize to [-1, 1].
    Matches training InsertionDataset's ref_image_tensor preprocessing.

    Returns:
        [1, C, H, W] float32 tensor in [-1, 1]  (1 = single frame dimension)
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (sample_size[1], sample_size[0]), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).contiguous().float() / 255.0  # [C, H, W]
    tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(tensor)
    return tensor.unsqueeze(0)  # [1, C, H, W] — frame dim


# ==================== GPU Memory Mode ====================
GPU_memory_mode     = "bf16"

# ==================== Low VRAM Mode ====================
# VAE encode -> (VAE offloaded) -> Qwen encode -> pipeline with model CPU offload (T5 -> Transformer -> VAE).
# Qwen stays on the GPU for the whole denoising loop (feedback re-encoding), so Qwen3-VL-8B and the
# 14B transformer are on the GPU at the same time; Qwen is moved to the CPU after each sample.
low_vram            = True

# ==================== Multi GPUs Config ====================
ulysses_degree      = 1
ring_degree         = 1
fsdp_dit            = False
fsdp_text_encoder   = False
compile_dit         = False

# ==================== TeaCache Config ====================
enable_teacache     = True
teacache_threshold  = 0.10
num_skip_start_steps = 5
teacache_offload    = False

# ==================== Other Optimizations ====================
cfg_skip_ratio      = 0
enable_riflex       = False
riflex_k            = 6

# ==================== Model Paths ====================
config_path         = os.path.join(project_roots[2], "config", "wan2.1", "wan_civitai.yaml")
qwen_encoder_path   = "PATH/Qwen3-VL-8B-Instruct"
shift               = 1.0
transformer_path    = "PATH/checkpoint-N"   # stage-2 checkpoint dir written by finetune.sh (<OUTPUT>/checkpoint-N); copy config.json from the stage-1 weights into it
save_path           = "./samples"
pretrained_model_name_or_path = "PATH/Wan2.1-T2V-14B"   # official Wan2.1-T2V-14B (VAE, T5 encoder, tokenizer)
model_name          = pretrained_model_name_or_path

# ==================== Video Insertion Inputs ====================
# One entry per sample. `instructions` are encoded by Qwen3-VL; `descriptions` (caption of the
# expected result) are encoded by T5 and fall back to the instruction when empty.

condition_video_paths = [
"PATH/0.mp4",
"PATH/1.mp4",
]
reference_image_paths = [
"PATH/0.png",
"PATH/1.png",
]
instructions = [
"Insert the object from the reference image onto the table in the video.",
"Insert the object from the reference image next to the person in the video.",
]
descriptions = [
"",
"",
]

use_t5   = True
use_qwen = True

# ==================== Generation Parameters ====================
sample_size         = [480, 832]  # [height, width]
video_length        = 33
fps                 = 16
weight_dtype        = torch.bfloat16
negative_prompt     = "Missing object, distortion"
guidance_scale      = 6.5
seed                = 43
num_inference_steps = 50
lora_weight         = 0.55


# ==================== Loop Config ====================
# Step interval of the feedback loop. A step i is eligible when i % qwen_reembed_interval == 0,
# its sigma lies in [denoise_lower, denoise_upper] and it is not the last step. Every eligible step
# decodes the image stream's x0 estimate (saved as step_XX.png); up to max_loop of them also
# re-encode with Qwen. <= 0 disables the loop.
qwen_reembed_interval = 10

# Sigma range for Qwen re-encoding.
# sigma ∈ [0,1]: 1 = pure noise, 0 = clean image.
# Only re-encode when denoise_lower <= sigma <= denoise_upper.
# Outside this range, the previous Qwen embeddings are reused.
denoise_upper          = 0.95  # skip very noisy steps (sigma > 0.95)
denoise_lower          = 0.4  # skip near-clean steps (sigma < 0.4)

# Max number of Qwen re-encoding calls within the sigma range.
# If there are more eligible steps than max_loop, evenly space them.
max_loop               = 4

# ==================== Setup ====================
device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

# Qwen encoder (load to CPU first)
qwen_config = MLLMInContextConfig(mllm_id=qwen_encoder_path, mode="edit", crop_system_tokens=True, crop_vision_tokens=False)
qwen_encoder = MLLMInContext(qwen_config).eval()

# Transformer config
transformer_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
transformer_kwargs['use_qwen_encoder'] = use_qwen
transformer_kwargs['qwen_hidden_size'] = qwen_encoder.mllm_hidden_size
transformer_kwargs['use_t5'] = use_t5

# Load Transformer
transformer = WanTransformer3DModel.from_pretrained(
    transformer_path,
    transformer_additional_kwargs=transformer_kwargs,
    low_cpu_mem_usage=False,
    torch_dtype=weight_dtype,
    Debug=True,
)
transformer.eval()

# Load VAE
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)
vae.eval()

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Scheduler
scheduler = FlowMatchEulerDiscreteScheduler(
    **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Text encoder (T5)
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder.eval()

# Create Pipeline (using loop variant)
pipeline = WanPipeline_insertion_loop(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)

# Multi-GPU setup
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("FSDP DIT enabled")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("FSDP Text Encoder enabled")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    print("Compile enabled")

# ==================== GPU Memory Mode (only for non-low-vram) ====================
if not low_vram:
    if GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)

# TeaCache setup
coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"TeaCache enabled: threshold={teacache_threshold}, skip_start_steps={num_skip_start_steps}")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold,
        num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )

if cfg_skip_ratio > 0:
    print(f"CFG skip ratio: {cfg_skip_ratio}")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)

generator = torch.Generator(device=device).manual_seed(seed)

# if lora_path is not None:
#     pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

# ==================== Save Results ====================
def save_results(cond_video_path, ref_image_path, loop_images=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)

    # Create a folder for this result
    result_folder = os.path.join(save_path, prefix)
    os.makedirs(result_folder, exist_ok=True)

    if video_length == 1:
        output_path = os.path.join(result_folder, "output.png")
        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(output_path)
        print(f"Saved image to: {output_path}")
    else:
        output_path = os.path.join(result_folder, "output.mp4")
        save_videos_grid(sample, output_path, fps=fps)
        print(f"Saved video to: {output_path}")

    # Copy input video and reference image
    shutil.copy2(cond_video_path, os.path.join(result_folder, "input.mp4"))
    shutil.copy2(ref_image_path, os.path.join(result_folder, "ref.png"))
    print(f"Copied input video and ref image to: {result_folder}/")

    # Save intermediate loop-decoded images
    if loop_images:
        for step_idx, img in loop_images:
            img.save(os.path.join(result_folder, f"step_{step_idx:02d}.png"))
        print(f"Saved {len(loop_images)} loop images to: {result_folder}/")



# ==================== Inference ====================
for idx in range(len(descriptions)):
    description = descriptions[idx]
    instruction = instructions[idx]
    condition_video_path = condition_video_paths[idx]
    reference_image_path = reference_image_paths[idx]
    with torch.no_grad():
        # Adjust video_length for VAE temporal compression
        video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
        latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

        if enable_riflex:
            pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)

        print(f"  Output size: {sample_size[0]}x{sample_size[1]}, {video_length} frames")

        # ==========================================================================
        # Step 1: VAE Encode — condition video & reference image
        # ==========================================================================
        if low_vram:
            text_encoder.to("cpu")
            qwen_encoder.to("cpu")
            torch.cuda.empty_cache()
            vae.to(device)
            print("[low_vram] VAE → GPU")

        # Condition video: [F,C,H,W] → [1,F,C,H,W] → rearrange → [1,C,F,H,W]
        condition_video_tensor = load_video_frames(condition_video_path, sample_size, video_length)
        condition_video_tensor = rearrange(condition_video_tensor.unsqueeze(0), "b f c h w -> b c f h w")
        condition_video_tensor = condition_video_tensor.to(device=device, dtype=weight_dtype)

        # Reference image: [1,C,H,W] → [1,1,C,H,W] → rearrange → [1,C,1,H,W]
        reference_image_tensor = load_reference_image(reference_image_path, sample_size)
        reference_image_tensor = rearrange(reference_image_tensor.unsqueeze(0), "b f c h w -> b c f h w")
        reference_image_tensor = reference_image_tensor.to(device=device, dtype=weight_dtype)

        src_vid_latents = vae.encode(condition_video_tensor)[0].sample()   # [1, C_lat, F_lat, H_lat, W_lat]
        ref_img_latents = vae.encode(reference_image_tensor)[0].sample()   # [1, C_lat, 1,     H_lat, W_lat]

        del condition_video_tensor, reference_image_tensor
        print(f"  src_vid_latents shape: {src_vid_latents.shape}")
        print(f"  ref_img_latents shape: {ref_img_latents.shape}")

        # ==========================================================================
        # Step 2: Qwen Encode — initial embeddings (edit + ref_edit)
        # ==========================================================================
        if low_vram:
            vae.to("cpu")
            torch.cuda.empty_cache()
            print("[low_vram] VAE → CPU, Qwen → GPU")
        qwen_encoder.to(torch.device(device))

        # Prepare images for Qwen
        ref_pil = Image.open(reference_image_path).convert("RGB")
        ref_pil_resized = ref_pil.resize((sample_size[1], sample_size[0]), Image.LANCZOS)
        images_for_model = [[ref_pil_resized]]

        src_first_pil = get_first_frame_as_pil(condition_video_path)
        src_first_pil_resized = src_first_pil.resize((sample_size[1], sample_size[0]), Image.LANCZOS)
        src_first_frame_for_model = [[src_first_pil_resized]]

        # 2a) Edit embedding
        prompt_embeds_qwen, _ = qwen_encoder.get_prompt_embeddings(
            prompts=[""],
            prompt2sys=[instruction],
            images=images_for_model,
            second_image=src_first_frame_for_model,
            device=torch.device(device),
            dtype=weight_dtype,
            mode="edit",
        )
        prompt_embeds_qwen = prompt_embeds_qwen.to(device=device, dtype=weight_dtype)

        # 2b) Ref-edit embedding
        ref_prompt_embeds_qwen, _ = qwen_encoder.get_prompt_embeddings(
            prompts=[""],
            images=images_for_model,
            second_image=src_first_frame_for_model,
            device=torch.device(device),
            dtype=weight_dtype,
            mode="ref_edit",
        )
        ref_prompt_embeds_qwen = ref_prompt_embeds_qwen.to(device=device, dtype=weight_dtype)

        del images_for_model
        print(f"  qwen_embed shape:      {prompt_embeds_qwen.shape}")
        print(f"  ref_qwen_embed shape:  {ref_prompt_embeds_qwen.shape}")

        # ==========================================================================
        # Step 3: Keep Qwen on GPU for the feedback loop
        #   Qwen is not offloaded here because the pipeline calls it during denoising
        #   through qwen_reembed_fn.
        #
        #   NOTE: This requires enough GPU memory for Qwen + Transformer
        #   (with model_cpu_offload, only one of {Transformer, VAE} is on GPU
        #    at a time, plus Qwen stays resident).
        # ==========================================================================
        # qwen_encoder stays on GPU — do NOT offload
        print("[loop] Qwen stays on GPU for the feedback loop")

        # ==========================================================================
        # Step 3b: Define Qwen re-embedding callback
        #   Called at every eligible step (see Loop Config) with the decoded x0 estimate
        #   of the image stream; the image is always saved, Qwen re-encodes only at re-encode steps.
        #   Replaces the `images` parameter (reference image position) in both
        #   Qwen calls; `second_image` (source video first frame) stays unchanged.
        # ==========================================================================
        loop_decoded_images = []   # collects (step_idx, PIL.Image) for saving

        def qwen_reembed_fn(decoded_pil, step_idx, do_reembed=True):
            """Save decoded image; optionally re-encode with Qwen."""
            loop_decoded_images.append((step_idx, decoded_pil.copy()))

            if not do_reembed:
                return None, None

            decoded_pil_resized = decoded_pil.resize(
                (sample_size[1], sample_size[0]), Image.LANCZOS
            )
            loop_images = [[decoded_pil_resized]]

            # Edit embedding (instruction + decoded image + source first frame)
            new_qwen, _ = qwen_encoder.get_prompt_embeddings(
                prompts=[""],
                prompt2sys=[instruction],
                images=loop_images,
                second_image=src_first_frame_for_model,
                device=torch.device(device),
                dtype=weight_dtype,
                mode="edit",
            )

            # Ref-edit embedding (decoded image + source first frame, no instruction)
            new_ref_qwen, _ = qwen_encoder.get_prompt_embeddings(
                prompts=[""],
                images=loop_images,
                second_image=src_first_frame_for_model,
                device=torch.device(device),
                dtype=weight_dtype,
                mode="ref_edit",
            )

            return (
                new_qwen.to(device=device, dtype=weight_dtype),
                new_ref_qwen.to(device=device, dtype=weight_dtype),
            )

        # ==========================================================================
        # Step 4: Enable model_cpu_offload for pipeline
        # ==========================================================================
        if low_vram:
            pipeline.enable_model_cpu_offload(device=device)
            print("[low_vram] pipeline.enable_model_cpu_offload()")

        # ==========================================================================
        # Step 5: Run Pipeline (with loop callback)
        # ==========================================================================
        t5_prompt = description if description else instruction

        sample = pipeline(
            prompt=t5_prompt,
            negative_prompt=negative_prompt,
            src_vid_latents=src_vid_latents,
            ref_img_latents=ref_img_latents,
            height=sample_size[0],
            width=sample_size[1],
            num_frames=video_length,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            shift=shift,
            qwen_embed=prompt_embeds_qwen,
            ref_qwen_embed=ref_prompt_embeds_qwen,
            # Loop-specific parameters
            qwen_reembed_callback=qwen_reembed_fn,
            qwen_reembed_interval=qwen_reembed_interval,
            denoise_upper=denoise_upper,
            denoise_lower=denoise_lower,
            max_loop=max_loop,
        ).videos

        # ==========================================================================
        # Step 6: Offload Qwen after pipeline is done
        # ==========================================================================
        qwen_encoder.to("cpu")
        torch.cuda.empty_cache()
        print("[loop] Qwen → CPU (pipeline done)")

        if ulysses_degree * ring_degree > 1:
            import torch.distributed as dist
            if dist.get_rank() == 0:
                save_results(condition_video_path, reference_image_path, loop_images=loop_decoded_images)
        else:
            save_results(condition_video_path, reference_image_path, loop_images=loop_decoded_images)
