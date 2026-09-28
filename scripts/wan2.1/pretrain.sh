#!/bin/bash
# Stage 1: pre-training. Adapts Wan2.1-T2V-14B to Qwen3-VL-8B conditioning (T5 text context is kept).
# Qwen3-VL encodes the first frame of each clip (i2v mode). Trainable: qwen_project_in (high lr)
# plus cross_attn / norm3 of every block (low lr); both groups use lr 2e-5 here. The released stage-1
# weights (Massyzs/smart-insertion-v-stage1 on Hugging Face) use these hyper-parameters; the number
# of nodes below is only an example.

# Activate your Python environment before running this script.
cd "$(dirname "$(readlink -f "$0")")"
# Set up error handling (pipefail: a failed training run is not hidden by tee)
set -eo pipefail


# NCCL configuration (distributed communication optimization)
# Increase socket concurrency
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=8

export NCCL_P2P_LEVEL=NVL

export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-20}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_NVLS_ENABLE=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_SOCKET_IFNAME=eth0   # network interface used by NCCL; adjust to your machine (e.g. ib0, bond0)
export NCCL_ALGO=RING
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTHONPATH=$(pwd)

export WORLD_SIZE=2   # number of nodes (example: 2 nodes x 8 GPUs); each node uses 8 GPUs (see --num_processes below)

# Master node IP: set MASTER_ADDR on every node (and NODE_RANK=1,2,... on the worker nodes)
if [ -n "$MASTER_ADDR" ]; then
    echo "Using master IP specified by environment variable: $MASTER_ADDR"
else
    echo "Master IP not found; please set MASTER_ADDR to the IP of node 0"
    exit 1
fi

export MASTER_PORT=21569

echo "Distributed configuration:"
echo "Master IP (MASTER_ADDR): $MASTER_ADDR"
echo "Local IP: $(hostname -I | awk '{print $1}')"
echo "Node rank (NODE_RANK): ${NODE_RANK:-0}"

# Determine this node's rank: FORCE_RANK > master IP match > NODE_RANK
LOCAL_IP=$(hostname -I | awk '{print $1}')
if [ -n "$FORCE_RANK" ]; then
    export RANK=$FORCE_RANK
    echo "Using manually specified RANK=$RANK"
elif [ "$LOCAL_IP" = "$MASTER_ADDR" ]; then
    export RANK=0
    echo "Detected this machine as the master node (RANK=0)"
elif [ -n "$NODE_RANK" ]; then
    # Node index starts from 0: the master node is 0, worker nodes are 1,2,3...
    export RANK=$NODE_RANK
    echo "Using NODE_RANK as RANK=$RANK"
else
    echo "This node's IP ($LOCAL_IP) is not MASTER_ADDR; please set NODE_RANK (1..$((WORLD_SIZE - 1)) on worker nodes)"
    exit 1
fi
if [ "$RANK" -lt 0 ] || [ "$RANK" -ge "$WORLD_SIZE" ]; then
    echo "RANK=$RANK is out of range for WORLD_SIZE=$WORLD_SIZE"
    exit 1
fi


# Show final configuration
echo "   Final distributed configuration:"
echo "   WORLD_SIZE: $WORLD_SIZE"
echo "   RANK: $RANK"
echo "   MASTER_ADDR: $MASTER_ADDR"
echo "   MASTER_PORT: $MASTER_PORT"


# Role-specific wait strategy
if [ "$RANK" = "0" ]; then
    echo "Master node waits 2 seconds for the other nodes to get ready..."
    sleep 2
else
    echo "Worker node waits 4 seconds for the master node to start first..."
    sleep 4
fi

echo "Starting distributed training..."

export video_w=832
export video_h=480

export MODEL_NAME="PATH/Wan2.1-T2V-14B"   # official Wan2.1-T2V-14B checkpoint
export PRETRAIN_DATASET_JSON_FOLDER="PATH/pretrain_json"   # JSON annotations (see the format notes at the end)
export QWEN_MODEL_NAME="PATH/Qwen3-VL-8B-Instruct"
export OUTPUT="PATH/output_stage1"   # checkpoints and logs are written here (must be shared by all nodes)
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT"
LOG_FILE="${OUTPUT}/train_rank${RANK}_$(date +%F_%H-%M-%S).log"

accelerate launch \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${RANK} \
  --num_machines ${WORLD_SIZE} \
  --num_processes $((${WORLD_SIZE} * 8)) \
  --use_fsdp \
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock \
  --fsdp_sharding_strategy "FULL_SHARD" \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_backward_prefetch "BACKWARD_PRE" \
  --fsdp_cpu_ram_efficient_loading False \
  --mixed_precision="bf16" \
   pretrain.py \
  --config_path="../../config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --qwen_encoder_path=$QWEN_MODEL_NAME \
  --qwen_encoder_mode="i2v" \
  --pretrain_dataset=$PRETRAIN_DATASET_JSON_FOLDER \
  --video_sample_size_h=$video_h \
  --video_sample_size_w=$video_w \
  --token_sample_size=640 \
  --video_sample_stride=2 \
  --video_sample_n_frames=49 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=0 \
  --num_train_epochs=10 \
  --checkpointing_steps=50 \
  --checkpoints_total_limit=30 \
  --learning_rate=2e-05 \
  --high_lr=2e-5 \
  --low_lr=2e-5 \
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
  --low_vram \
  --train_mode="normal" \
  --trainable_modules "project_in"  \
  --trainable_modules_low_learning_rate "cross_attn" "norm3" \
  --use_t5 \
  --resume_from_checkpoint "latest" \
  2>&1 | tee "$LOG_FILE"
echo "Training finished"
# Optional: add --check_json to the arguments above to drop samples whose media files are missing
# (the result is cached in $OUTPUT/check_json.json).

# ====== Dataset JSON format (PretrainVideoDataset) ======
#
# --pretrain_dataset accepts one or more JSON files and/or directories of JSON files:
#    a) Single JSON file: --pretrain_dataset /path/to/data.json
#    b) JSON directory:   --pretrain_dataset /path/to/json_dir/
#    c) Multiple inputs:  --pretrain_dataset /path/to/dir1/ /path/to/file.json /path/to/dir2/
#
# A JSON file holds one sample ({}) or a list of samples ([{}, {}, ...]):
#    [
#        {
#            "file_path": "/abs/path/to/video1.mp4",   # absolute path
#            "text": "A short caption",
#            "dense_text": "A detailed caption of the whole clip"
#        },
#        ...
#    ]
# The dense caption is used for 90% of the samples and the short caption for the rest.
# In i2v mode the first frame of each clip is also given to the Qwen3-VL encoder.

# ====== Trainable module notes ======
# --trainable_modules and --trainable_modules_low_learning_rate take keywords that are matched as
# substrings against transformer parameter names (see _match in pretrain.py).
#   - a parameter matching a low-lr keyword goes to the low-lr group (lr = --low_lr)
#   - otherwise a parameter matching a high-lr keyword goes to the high-lr group (lr = --high_lr)
#   - --learning_rate only sets the optimizer default and does not override these two group values
# Examples: "project_in" (the qwen_project_in layer), "cross_attn", "norm3"; "." matches every parameter.
#
# --qwen_encoder_mode argument:
#   "i2v"  : first frame + empty prompt (used by this stage)
#   "t2v"  : empty prompt, no image (Qwen contributes only its template tokens)
#   "edit" : reference image + source first frame + instruction (used by finetune.sh)
