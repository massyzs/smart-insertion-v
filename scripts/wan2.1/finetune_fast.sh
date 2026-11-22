#!/usr/bin/env bash
# =============================================================================
# Stage 2 (insertion fine-tuning) with finetune_fast.py, the faster stage-2 trainer.
# Same recipe as finetune.sh: stage-1 initialization, all transformer parameters, AdamW at lr 1e-5, gradient
# clipping 0.5, bf16, FSDP FULL_SHARD, 33-frame clips at 832x480. In addition:
#   * the `image` subset of the dataset is used, as single-frame samples (every global step holds samples of
#     one kind, all video clips or all single-frame images);
#   * decoding runs in dataloader workers, and the frozen encoders stay on the GPU or in pinned host memory
#     instead of being moved between CPU and GPU several times per step;
#   * every rank writes its own checkpoint file, so OUTPUT can be node-local (no shared filesystem needed);
#     a restart resumes from the newest checkpoint that is complete on every node, with the optimizer state,
#     the RNG states and the exact data position;
#   * optional FlashAttention-3 and a faster, numerically equivalent model file.
#
# Run one copy on every node, node 0 first:
#     MASTER_ADDR=<ip-of-node-0> bash scripts/wan2.1/finetune_fast.sh 0     # node 0
#     MASTER_ADDR=<ip-of-node-0> bash scripts/wan2.1/finetune_fast.sh 1     # node 1 (2, 3, ... for more nodes)
# Clean stop: `touch $OUTPUT/STOP` on any node. State and weights are saved, then the run exits; the file is renamed
#   to STOP.consumed-<step>, so starting the script again continues the run.
# Extra checkpoint while training continues: `touch $OUTPUT/SAVE_NOW` on any node.
# Consolidated weights for inference: <OUTPUT>/checkpoint-N/diffusion_pytorch_model.safetensors (+ config.json),
#   written on node 0.
# =============================================================================
set -eo pipefail

if [ $# -lt 1 ]; then
  echo "usage: MASTER_ADDR=<ip-of-node-0> bash $0 <NODE_RANK>   (0 on node 0, 1, 2, ... on the other nodes)"
  exit 1
fi
NODE_RANK=$1
if [ -z "$MASTER_ADDR" ]; then
  echo "set MASTER_ADDR to the IP address of node 0"
  exit 1
fi

# ---------------------------- Config ----------------------------
NUM_NODES=2                                  # example: 2 nodes x 8 GPUs
GPUS_PER_NODE=8
MASTER_PORT=21569

MODEL_NAME=PATH/Wan2.1-T2V-14B-stage1        # Wan2.1-T2V-14B layout with the stage-1 transformer (README: "Stage-1 weights")
QWEN_MODEL_NAME=PATH/Qwen3-VL-8B-Instruct
DATASET=PATH/smart-insertion-v-dataset       # holds video1/info, video2/info and image/info (README: "Dataset")
INSERTION_CACHE=PATH/insertion_cache.json    # frame-count cache of the two video subsets (probe_video_lengths.py)
OUTPUT=PATH/output_stage2_fast               # checkpoint-N/, logs/ and TensorBoard events; node-local is fine
FA3_SITE=PATH/fa3_site                       # directory holding flash_attn_interface.py (only used with USE_FA3=1)

video_w=832
video_h=480
FRAME_NUM=33
NUM_TRAIN_EPOCHS=1
CHECKPOINTING_STEPS=500
CHECKPOINTS_TOTAL_LIMIT=5
DATALOADER_NUM_WORKERS=4
# Placement of the frozen encoders on 80 GB GPUs (33 frames, 832x480): Qwen3-VL resident + T5 swapped peaks at
# about 64-66 GiB allocated; swapping both lowers the peak to about 50 GiB but adds about 1 s per step.
QWEN_PLACEMENT=resident      # resident = always on the GPU; swap = pinned host memory, on the GPU only for its two calls per step
T5_PLACEMENT=swap
FAST_MODEL=1                 # 1 = videox_fun/models/wan_transformer3d_insertion_fast.py (same weights and maths, fewer
                             #     elementwise passes per block), 0 = the original model file
FAST_OPTS=all                # switches of the fast model file (see its docstring); ignored with FAST_MODEL=0
USE_FA3=0                    # 1 = FlashAttention-3 from $FA3_SITE (the run refuses to start without it), 0 = FlashAttention-2

# ---------------------------- Environment ----------------------------
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p "$OUTPUT/logs"
LOG_FILE=$OUTPUT/logs/node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[launcher] node $NODE_RANK of $NUM_NODES, master $MASTER_ADDR:$MASTER_PORT, output $OUTPUT"
if [ -f "$OUTPUT/STOP" ]; then
  echo "[launcher] $OUTPUT/STOP exists (a clean stop was requested); remove it to continue training"
  exit 1
fi

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=4
export NCCL_SOCKET_IFNAME=eth0               # network interface used by NCCL; adjust to your machine (e.g. ib0, bond0)
export GLOO_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export SI_FAST_MODEL=$FAST_MODEL             # read by finetune_fast.py (model file) and by the fast model file (switches)
export SI_FAST_OPTS=$FAST_OPTS
FA3_ARGS=""
if [ "$USE_FA3" = "1" ]; then
  if [ ! -f "$FA3_SITE/flash_attn_interface.py" ]; then
    echo "[launcher] USE_FA3=1 but $FA3_SITE/flash_attn_interface.py does not exist"
    exit 1
  fi
  export PYTHONPATH=$FA3_SITE:$PYTHONPATH    # videox_fun/models/attention_utils.py uses FlashAttention-3 when it can be imported
  FA3_ARGS="--require_fa3"
fi

# ---------------------------- Train ----------------------------
accelerate launch \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $NODE_RANK \
  --num_machines $NUM_NODES \
  --num_processes $((NUM_NODES * GPUS_PER_NODE)) \
  --use_fsdp \
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock \
  --fsdp_sharding_strategy "FULL_SHARD" \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_backward_prefetch "BACKWARD_PRE" \
  --fsdp_forward_prefetch true \
  --fsdp_cpu_ram_efficient_loading False \
  --mixed_precision="bf16" \
   finetune_fast.py \
  --config_path="../../config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --qwen_encoder_path=$QWEN_MODEL_NAME \
  --qwen_encoder_mode="edit" \
  --pretrain_dataset $DATASET/video1/info $DATASET/video2/info $DATASET/image/info \
  --insertion_cache $INSERTION_CACHE \
  --video_sample_size_h=$video_h \
  --video_sample_size_w=$video_w \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=$FRAME_NUM \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=$CHECKPOINTS_TOTAL_LIMIT \
  --learning_rate=1e-05 \
  --high_lr=1e-5 \
  --low_lr=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir=$OUTPUT \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=1e-2 \
  --adam_epsilon=1e-8 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.5 \
  --uniform_sampling \
  --train_mode="normal" \
  --trainable_modules "."  \
  --use_t5 \
  --qwen_placement $QWEN_PLACEMENT \
  --t5_placement $T5_PLACEMENT \
  $FA3_ARGS \
  --tracker_project_name stage2_fast \
  --resume_from_checkpoint "latest"
echo "[launcher] $(date '+%F %T') training finished on node $NODE_RANK"
