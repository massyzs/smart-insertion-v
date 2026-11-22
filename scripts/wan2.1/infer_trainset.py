"""Closed-loop feedback inference on a list of samples, driven by arguments.

This is the flow of scripts/wan2.1/inference.py (same preprocessing, same pipeline call, same generation and
feedback-loop parameters) with the module-level configuration block replaced by command-line arguments, so
that several processes can run side by side, one per GPU:

    CUDA_VISIBLE_DEVICES=3 python infer_trainset.py --transformer_path <ckpt dir> ... --shard 3 --num_shards 8

A process handles the samples whose position i in --samples_json satisfies i % num_shards == shard.
Each sample is a dict with absolute paths: {"name", "kind" ("video" | "image"), "input_video", "ref_img",
"prompt", "description", and optionally "gt_video", "gt_ref_img"}. Image samples run with one frame.

Output per sample, in <save_path>/<position>_<name>/:
    output.mp4 (or output.png), the copied inputs, step_XX.png (decoded feedback estimates),
    compare.mp4 / compare.png (source | output | ground truth) and meta.json.
"""
import argparse
import json
import os
import shutil
import sys
import time

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

from videox_fun.models import (AutoencoderKLWan, WanT5EncoderModel, AutoTokenizer)
from videox_fun.models.wan_transformer3d_insertion_new import WanTransformer3DModel
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline.pipeline_wan_insertion import WanPipeline_insertion_loop
from videox_fun.utils.utils import filter_kwargs, save_videos_grid
from qwen_encoder import MLLMInContext, MLLMInContextConfig
if not hasattr(torch, "int1"):
    torch.int1 = torch.int8


# ==================== Helper Functions (as in inference.py) ====================

def read_frames_rgb(path: str, max_frames: int):
    """First `max_frames` frames of a video (or the single frame of a still) as RGB uint8 arrays."""
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) == 0:
        # regular image files that OpenCV's video reader does not open
        frames = [np.array(Image.open(path).convert("RGB"))]
    return frames


def get_first_frame_as_pil(video_path: str) -> Image.Image:
    """Extract the first frame of a video as a PIL Image (RGB)."""
    return Image.fromarray(read_frames_rgb(video_path, 1)[0])


def load_video_frames(video_path: str, sample_size, video_length: int) -> torch.Tensor:
    """
    Load video frames, resize, normalize to [-1, 1].

    Returns:
        [F, C, H, W] float32 tensor in [-1, 1]
    """
    frames = [cv2.resize(f, (sample_size[1], sample_size[0]), interpolation=cv2.INTER_LINEAR)
              for f in read_frames_rgb(video_path, video_length)]

    # Pad with last frame or truncate
    if len(frames) < video_length:
        last_frame = frames[-1]
        frames.extend([last_frame] * (video_length - len(frames)))
    else:
        frames = frames[:video_length]

    frames_np = np.array(frames)  # [F, H, W, C] uint8
    tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous().float() / 255.0  # [F, C, H, W]
    tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(tensor)
    return tensor


def load_reference_image(image_path: str, sample_size) -> torch.Tensor:
    """
    Load image, resize, normalize to [-1, 1].

    Returns:
        [1, C, H, W] float32 tensor in [-1, 1]  (1 = single frame dimension)
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (sample_size[1], sample_size[0]), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).contiguous().float() / 255.0  # [C, H, W]
    tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(tensor)
    return tensor.unsqueeze(0)  # [1, C, H, W] — frame dim


def check_transformer_weights(transformer, transformer_path):
    """from_pretrained loads non-strictly and only prints mismatches; make sure nothing stayed at its random init."""
    from safetensors import safe_open
    weights_file = os.path.join(transformer_path, "diffusion_pytorch_model.safetensors")
    with safe_open(weights_file, framework="pt") as f:
        file_shapes = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    model_shapes = {k: tuple(v.shape) for k, v in transformer.state_dict().items()}
    missing = sorted(set(model_shapes) - set(file_shapes))
    unexpected = sorted(set(file_shapes) - set(model_shapes))
    mismatched = sorted(k for k in set(model_shapes) & set(file_shapes) if model_shapes[k] != file_shapes[k])
    if missing or unexpected or mismatched:
        raise RuntimeError(f"{weights_file} does not match the model: {len(missing)} missing, {len(unexpected)} unexpected, "
                           f"{len(mismatched)} shape mismatches (e.g. {(missing + unexpected + mismatched)[:5]})")
    print(f"[weights] {weights_file}: all {len(model_shapes)} tensors match the model", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--transformer_path", required=True, help="stage-2 checkpoint dir with config.json + diffusion_pytorch_model.safetensors")
    p.add_argument("--pretrained_model_name_or_path", required=True, help="official Wan2.1-T2V-14B (VAE, T5 encoder, tokenizer)")
    p.add_argument("--qwen_encoder_path", required=True)
    p.add_argument("--samples_json", required=True)
    p.add_argument("--save_path", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--min_free_gpu_gib", type=float, default=60.0)
    p.add_argument("--skip_done", action="store_true", help="skip samples whose result folder already holds a DONE marker")
    # generation parameters: defaults of inference.py
    p.add_argument("--video_length", type=int, default=33)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--negative_prompt", type=str, default="Missing object, distortion")
    p.add_argument("--guidance_scale", type=float, default=6.5)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--qwen_reembed_interval", type=int, default=10)
    p.add_argument("--denoise_upper", type=float, default=0.95)
    p.add_argument("--denoise_lower", type=float, default=0.4)
    p.add_argument("--max_loop", type=int, default=4)
    p.add_argument("--no_low_vram", dest="low_vram", action="store_false",
                   help="keep every model on the GPU (inference.py default is low_vram: model CPU offload)")
    p.set_defaults(low_vram=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    weight_dtype = torch.bfloat16
    sample_size = [480, 832]  # [height, width]
    shift = 1.0
    use_t5 = True
    use_qwen = True
    enable_teacache = True
    teacache_threshold = 0.10
    num_skip_start_steps = 5
    teacache_offload = False
    low_vram = args.low_vram

    with open(args.samples_json, "r", encoding="utf-8") as f:
        all_samples = json.load(f)
    my_samples = [(i, s) for i, s in enumerate(all_samples) if i % args.num_shards == args.shard]
    if args.skip_done:
        my_samples = [(i, s) for i, s in my_samples
                      if not os.path.isfile(os.path.join(args.save_path, f"{i:02d}_{s.get('name', 'sample')}", "DONE"))]
    print(f"[shard {args.shard}/{args.num_shards}] {len(my_samples)} of {len(all_samples)} samples", flush=True)
    if not my_samples:
        return

    # Create the CUDA context first (the model load below is minutes of CPU-only work) and require a free card.
    torch.zeros(1, device=device)
    free_gib = torch.cuda.mem_get_info()[0] / 2**30
    if free_gib < args.min_free_gpu_gib:
        raise RuntimeError(f"only {free_gib:.1f} GiB free on the GPU (need {args.min_free_gpu_gib}); is another job holding it?")

    config_path = os.path.join(project_roots[2], "config", "wan2.1", "wan_civitai.yaml")
    config = OmegaConf.load(config_path)

    # Qwen encoder (load to CPU first)
    qwen_config = MLLMInContextConfig(mllm_id=args.qwen_encoder_path, mode="edit", crop_system_tokens=True, crop_vision_tokens=False)
    qwen_encoder = MLLMInContext(qwen_config).eval()

    # Transformer config
    transformer_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
    transformer_kwargs['use_qwen_encoder'] = use_qwen
    transformer_kwargs['qwen_hidden_size'] = qwen_encoder.mllm_hidden_size
    transformer_kwargs['use_t5'] = use_t5

    # Load Transformer
    transformer = WanTransformer3DModel.from_pretrained(
        args.transformer_path,
        transformer_additional_kwargs=transformer_kwargs,
        low_cpu_mem_usage=False,
        torch_dtype=weight_dtype,
        Debug=True,
    )
    transformer.eval()
    check_transformer_weights(transformer, args.transformer_path)

    # Load VAE
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)
    vae.eval()

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Text encoder (T5)
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
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

    if not low_vram:
        pipeline.to(device=device)

    # TeaCache setup (as in inference.py)
    coefficients = get_teacache_coefficients(args.pretrained_model_name_or_path) if enable_teacache else None
    if coefficients is not None:
        print(f"TeaCache enabled: threshold={teacache_threshold}, skip_start_steps={num_skip_start_steps}")
        pipeline.transformer.enable_teacache(
            coefficients, args.num_inference_steps, teacache_threshold,
            num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
        )

    os.makedirs(args.save_path, exist_ok=True)

    for position, item in my_samples:
        t_start = time.time()
        description = item.get("description", "") or ""
        instruction = item["prompt"]
        condition_video_path = item["input_video"]
        reference_image_path = item["ref_img"]
        is_image = item.get("kind", "video") == "image"
        video_length = 1 if is_image else args.video_length
        # one generator per sample, so a sample's noise does not depend on its position or on the sharding
        generator = torch.Generator(device=device).manual_seed(args.seed)
        result_folder = os.path.join(args.save_path, f"{position:02d}_{item.get('name', 'sample')}")
        os.makedirs(result_folder, exist_ok=True)
        print(f"\n===== sample {position} ({item.get('name')}), {video_length} frame(s) -> {result_folder}", flush=True)

        with torch.no_grad():
            # Adjust video_length for VAE temporal compression
            video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1

            # ==========================================================================
            # Step 1: VAE Encode — condition video & reference image
            # ==========================================================================
            if low_vram:
                text_encoder.to("cpu")
                qwen_encoder.to("cpu")
                torch.cuda.empty_cache()
                vae.to(device)

            # Condition video: [F,C,H,W] → [1,F,C,H,W] → rearrange → [1,C,F,H,W]
            condition_video_frames = load_video_frames(condition_video_path, sample_size, video_length)
            condition_video_tensor = rearrange(condition_video_frames.unsqueeze(0), "b f c h w -> b c f h w")
            condition_video_tensor = condition_video_tensor.to(device=device, dtype=weight_dtype)

            # Reference image: [1,C,H,W] → [1,1,C,H,W] → rearrange → [1,C,1,H,W]
            reference_image_tensor = load_reference_image(reference_image_path, sample_size)
            reference_image_tensor = rearrange(reference_image_tensor.unsqueeze(0), "b f c h w -> b c f h w")
            reference_image_tensor = reference_image_tensor.to(device=device, dtype=weight_dtype)

            src_vid_latents = vae.encode(condition_video_tensor)[0].sample()   # [1, C_lat, F_lat, H_lat, W_lat]
            ref_img_latents = vae.encode(reference_image_tensor)[0].sample()   # [1, C_lat, 1,     H_lat, W_lat]

            del condition_video_tensor, reference_image_tensor

            # ==========================================================================
            # Step 2: Qwen Encode — initial embeddings (edit + ref_edit)
            # ==========================================================================
            if low_vram:
                vae.to("cpu")
                torch.cuda.empty_cache()
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

            # ==========================================================================
            # Step 3: Qwen stays on the GPU for the feedback loop; re-embedding callback
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

            # ==========================================================================
            # Step 5: Run Pipeline (with loop callback)
            # ==========================================================================
            t5_prompt = description if description else instruction

            sample = pipeline(
                prompt=t5_prompt,
                negative_prompt=args.negative_prompt,
                src_vid_latents=src_vid_latents,
                ref_img_latents=ref_img_latents,
                height=sample_size[0],
                width=sample_size[1],
                num_frames=video_length,
                generator=generator,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                shift=shift,
                qwen_embed=prompt_embeds_qwen,
                ref_qwen_embed=ref_prompt_embeds_qwen,
                # Loop-specific parameters
                qwen_reembed_callback=qwen_reembed_fn,
                qwen_reembed_interval=args.qwen_reembed_interval,
                denoise_upper=args.denoise_upper,
                denoise_lower=args.denoise_lower,
                max_loop=args.max_loop,
            ).videos

            # ==========================================================================
            # Step 6: Offload Qwen after pipeline is done
            # ==========================================================================
            qwen_encoder.to("cpu")
            torch.cuda.empty_cache()

        # ==================== Save Results ====================
        sample = sample.float().cpu()                                   # [1, C, F, H, W] in [0, 1]
        if not bool(torch.isfinite(sample).all()) or float(sample.std()) < 1e-4:
            raise RuntimeError(f"sample {position}: output is not finite or is constant (std {float(sample.std()):.2e})")
        source = (condition_video_frames * 0.5 + 0.5).clamp(0, 1).permute(1, 0, 2, 3).unsqueeze(0)   # [1, C, F, H, W]
        panels = [source, sample]
        if item.get("gt_video"):
            gt = load_video_frames(item["gt_video"], sample_size, video_length)
            panels.append((gt * 0.5 + 0.5).clamp(0, 1).permute(1, 0, 2, 3).unsqueeze(0))
        compare = torch.cat(panels, dim=4)                              # side by side: source | output | ground truth

        def to_pil(frame_chw):
            return Image.fromarray((frame_chw.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))

        if video_length == 1:
            to_pil(sample[0, :, 0]).save(os.path.join(result_folder, "output.png"))
            to_pil(compare[0, :, 0]).save(os.path.join(result_folder, "compare.png"))
        else:
            save_videos_grid(sample, os.path.join(result_folder, "output.mp4"), fps=args.fps)
            save_videos_grid(compare, os.path.join(result_folder, "compare.mp4"), fps=args.fps)
            # first / middle / last frame of the comparison, for a quick look without a video player
            Image.fromarray(np.concatenate([np.array(to_pil(compare[0, :, t])) for t in (0, video_length // 2, video_length - 1)], axis=0)) \
                .save(os.path.join(result_folder, "compare.png"))

        # Copy the inputs (keeping their extensions) and the ground truth
        for key in ("input_video", "ref_img", "gt_video", "gt_ref_img"):
            if item.get(key):
                shutil.copy2(item[key], os.path.join(result_folder, key + os.path.splitext(item[key])[1]))

        # Save intermediate loop-decoded images
        for step_idx, img in loop_decoded_images:
            img.save(os.path.join(result_folder, f"step_{step_idx:02d}.png"))

        seconds = time.time() - t_start
        with open(os.path.join(result_folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({**item, "position": position, "frames": video_length, "seed": args.seed,
                       "num_inference_steps": args.num_inference_steps, "guidance_scale": args.guidance_scale,
                       "feedback_steps": [s for s, _ in loop_decoded_images], "transformer_path": args.transformer_path,
                       "seconds": round(seconds, 1), "output_mean": float(sample.mean()), "output_std": float(sample.std())},
                      f, ensure_ascii=False, indent=2)
        with open(os.path.join(result_folder, "DONE"), "w") as f:
            f.write(time.strftime("%F %T"))
        print(f"[done] sample {position} in {seconds:.0f}s -> {result_folder}", flush=True)


if __name__ == "__main__":
    main()
