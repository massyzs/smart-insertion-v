# Smart-Insertion-V

Official training and inference code for
**Smart-Insertion-V: Photorealistic Video Insertion via a Closed-Loop Feedback Dual-Stream Framework**
(SIGGRAPH Asia 2026).

Smart-Insertion-V inserts the object shown in a reference image into a source video, following a text
instruction. It is built on the [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) video
diffusion transformer and uses [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
as a vision-language guidance encoder next to the T5 text encoder.

## Released resources

| Resource | Hugging Face link |
| --- | --- |
| Stage-1 (pre-training) checkpoint | [https://huggingface.co/Massyzs/smart-insertion-v-stage1](https://huggingface.co/Massyzs/smart-insertion-v-stage1) |
| Smart-Insertion-V dataset | [https://huggingface.co/datasets/Massyzs/smart-insertion-v-dataset](https://huggingface.co/datasets/Massyzs/smart-insertion-v-dataset) |

Both Hugging Face repositories are gated: request access on the repository page, then log in with
`hf auth login` (or set `HF_TOKEN`) before downloading.

**Release policy.** Due to policy restrictions, the fine-tuned (stage-2) model is not released. Instead, we
release the stage-1 pre-training checkpoint, which is the most compute-intensive part of training,
together with the full training code and the Smart-Insertion-V dataset. Starting from the stage-1 checkpoint,
you can reproduce the fine-tuned model by running stage 2 on the released dataset.

## Getting started

1. [Install](#installation) the environment.
2. [Download](#model-weights) the base models and the stage-1 checkpoint.
3. [Download](#dataset) the dataset and build the frame-count cache.
4. [Train stage 2](#stage-2-insertion-fine-tuning) with `scripts/wan2.1/finetune.sh`.
5. [Run inference](#inference) with the closed-loop feedback script `scripts/wan2.1/inference.py`.

All commands below are run from the repository root, and models and data are downloaded there.

## Repository layout

| Path | Content |
| --- | --- |
| `scripts/wan2.1/pretrain.sh`, `pretrain.py` | Stage 1: connects Qwen3-VL to Wan2.1-T2V-14B |
| `scripts/wan2.1/finetune.sh`, `finetune.py` | Stage 2: end-to-end insertion fine-tuning |
| `scripts/wan2.1/inference.py` | Closed-loop feedback inference on your own videos and reference images |
| `scripts/wan2.1/qwen_encoder.py` | Qwen3-VL guidance encoder |
| `scripts/wan2.1/probe_video_lengths.py` | Builds the frame-count cache required by stage 2 |
| `videox_fun/` | Models, pipelines, datasets and utilities |
| `config/wan2.1/wan_civitai.yaml` | Model config shared by all scripts |

## Installation

```bash
conda create -n smart-insertion-v python=3.10 -y
conda activate smart-insertion-v
pip install -r requirements.txt
```

Reference environment: Python 3.10, PyTorch 2.8, transformers 5.12, diffusers 0.39 and accelerate 1.14.
Qwen3-VL needs `transformers>=4.57`. Optional packages such as `flash-attn` are listed at the end of
`requirements.txt`.

## Model weights

Download the base models:

```bash
hf download Wan-AI/Wan2.1-T2V-14B --local-dir Wan2.1-T2V-14B
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir Qwen3-VL-8B-Instruct
```

### Stage-1 weights

The stage-1 checkpoint is released, so stage 1 can be skipped. Stage 2 expects a directory laid out like
Wan2.1-T2V-14B. Build it with symbolic links:

```bash
hf download Massyzs/smart-insertion-v-stage1 --local-dir smart-insertion-v-stage1

mkdir -p Wan2.1-T2V-14B-stage1
ln -s "$PWD"/Wan2.1-T2V-14B/{Wan2.1_VAE.pth,models_t5_umt5-xxl-enc-bf16.pth,google} Wan2.1-T2V-14B-stage1/
ln -s "$PWD"/smart-insertion-v-stage1/{config.json,diffusion_pytorch_model.safetensors} Wan2.1-T2V-14B-stage1/
```

`Wan2.1-T2V-14B-stage1` is the `MODEL_NAME` used by `finetune.sh`.

## Dataset

The dataset has three subsets: `video1/`, `video2/` and `image/`. The released recipe trains on the two
video subsets only. Download and extract them as described on the
[dataset card](https://huggingface.co/datasets/Massyzs/smart-insertion-v-dataset). The two video subsets are
about 750 GB of archives and need about as much space again once extracted.

```bash
hf download Massyzs/smart-insertion-v-dataset --repo-type dataset --local-dir smart-insertion-v-dataset \
    --include "video1/*" "video2/*"
cd smart-insertion-v-dataset
for n in $(ls */*.tar.part-0000 */*/*.tar.part-0000 2>/dev/null | sed 's#\.tar\.part-0000##'); do
    cat "$n".tar.part-* | tar -xf -
done
```

Every sample has the keys `input_video`, `gt_video`, `ref_img`, `gt_ref_img`, `prompt` (the editing
instruction, encoded by Qwen3-VL) and `description` (the caption of the target video, encoded by T5).
`ref_img` is empty for a small number of samples; the stage-2 loader skips them.

The training code resolves relative paths against the parent of the folder that holds the index file.
Move each index into an `info/` folder so that its paths resolve correctly:

```bash
for s in video1 video2; do
    mkdir -p "$s/info" && mv "$s/info.json" "$s/info/"
done
cd ..
```

Stage 2 splits long clips into fixed-length segments using a frame-count cache. Build it once:

```bash
python scripts/wan2.1/probe_video_lengths.py \
    --ann_path smart-insertion-v-dataset/video1/info smart-insertion-v-dataset/video2/info \
    --output insertion_cache.json
```

## Training

Both launch scripts use FSDP through `accelerate`. The example configuration is 2 nodes with 8 GPUs each.
Set `WORLD_SIZE` in the script to your number of nodes. Before launching, replace every `PATH/...`
placeholder in the script (models, dataset folders, frame-count cache, `OUTPUT`) with an absolute path.
The scripts change into `scripts/wan2.1` before starting, so relative paths do not work.

All nodes must see the models, the dataset, the frame-count cache and `OUTPUT` at the same absolute paths.
`OUTPUT` must be on a filesystem shared by all nodes: checkpoints are sharded across ranks, and training
resumes from the latest checkpoint automatically.

Start the script on every node with the IP address of node 0:

```bash
# node 0
MASTER_ADDR=<ip-of-node-0> bash scripts/wan2.1/finetune.sh
# node 1
MASTER_ADDR=<ip-of-node-0> NODE_RANK=1 bash scripts/wan2.1/finetune.sh
```

Node 0 is detected automatically when its IP equals `MASTER_ADDR`; the other nodes need `NODE_RANK`.
`FORCE_RANK` overrides the detected rank.

### Stage 1: pre-training (optional, the checkpoint is released)

`scripts/wan2.1/pretrain.sh` trains a new MLP, `qwen_project_in`, that maps Qwen3-VL hidden states into the
transformer width, together with the `cross_attn` and `norm3` layers of every block. The projected Qwen tokens
are concatenated with the T5 text tokens as cross-attention context. Qwen3-VL sees the first frame of each clip.
The stage-1 training corpus is not released. To run stage 1, provide your own text-to-video data as JSON
files with `file_path` (absolute path), `text` and `dense_text`; the format is documented at the end of
`pretrain.sh`.

### Stage 2: insertion fine-tuning

The fine-tuned model is not released, so this step is required before inference.
`scripts/wan2.1/finetune.sh` fine-tunes the whole transformer on the Smart-Insertion-V dataset, starting
from the stage-1 checkpoint. Set `MODEL_NAME` to the absolute path of the `Wan2.1-T2V-14B-stage1` directory,
list the dataset index folders in `INSERT_DATASET_JSON_FOLDER` and point `INSERTION_CACHE` at the
frame-count cache.
Checkpoints are written to `<OUTPUT>/checkpoint-N/`.

## Inference

Inference needs a stage-2 checkpoint that you train with `finetune.sh` (see [Release policy](#released-resources)).
The stage-1 checkpoint loads without error but is not trained for insertion.

`scripts/wan2.1/inference.py` runs the closed-loop feedback inference. During denoising, the image stream's
one-step estimate of the clean result (x0) is decoded and fed back to Qwen3-VL in place of the reference image.
The new Qwen3-VL guidance and the x0 latent then replace the conditions for the remaining steps. The denoising
trajectory itself is not restarted.

Edit the configuration block at the top of the script:

- `transformer_path`: a stage-2 checkpoint directory (`<OUTPUT>/checkpoint-N`). Copy `config.json` from the
  stage-1 weights into it.
- `pretrained_model_name_or_path`: the official Wan2.1-T2V-14B directory.
- `qwen_encoder_path`: the Qwen3-VL-8B-Instruct directory.
- `condition_video_paths`, `reference_image_paths`, `instructions`, `descriptions`: one entry per sample.
  `instructions` go to Qwen3-VL; `descriptions` go to T5 and fall back to the instruction when empty.
- Feedback loop (`Loop Config`): a denoising step is eligible when its sigma lies in
  [`denoise_lower`, `denoise_upper`] (default 0.4 to 0.95) and its index is a multiple of
  `qwen_reembed_interval` (default 10). At most `max_loop` (default 4) evenly spaced eligible steps re-encode
  with Qwen3-VL. `max_loop = 0` disables the feedback and only saves the intermediate estimates.

Then run:

```bash
python scripts/wan2.1/inference.py
```

Each sample gets its own folder under `./samples` (relative to the current directory), for example
`./samples/00000001/`. It contains `output.mp4`, copies of the inputs (`input.mp4`, `ref.png`) and the
decoded intermediate estimates (`step_XX.png`). The first 33 frames of each source video are edited at
832x480, matching training. The script runs on a single GPU. Qwen3-VL stays on the GPU during denoising
for the feedback loop, while the other models are offloaded to the CPU when they are not in use.

## Acknowledgements

This code builds on [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun),
[Wan2.1](https://github.com/Wan-Video/Wan2.1) and [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL).

## License

This project is released under the Apache 2.0 license. See [LICENSE](LICENSE).
