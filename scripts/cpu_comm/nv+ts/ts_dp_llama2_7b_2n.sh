#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=1
export NCCL_SOCKET_IFNAME=ens21f0
export GLOO_SOCKET_IFNAME=ens21f0

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=10.21.18.42
MASTER_PORT=34567
NNODES=2
NODE_RANK=1

# CHECKPOINT_PATH=/data2/share/llama-dataset/cp
# TENSORBOARD_LOGS_PATH=/data2/share/llama-dataset/tb
TOKENIZER_PATH=/data2/nfs/llama-dataset/tokenizer.model
DATA_PATH=/data2/nfs/llama-dataset/RedPajama-Data-1T-Sample/RedPajama-Data-1T-Sample

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

LLAMA_MODEL_ARGS=(
    --num-layers 12
    --hidden-size 4096
    --ffn-hidden-size 11008
    --num-attention-heads 32
    --seq-length 4096
    --max-position-embeddings 4096
    --num-query-groups 8
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model $TOKENIZER_PATH
    --swiglu
    --use-flash-attn
    --normalization RMSNorm
    --position-embedding-type rope
    --disable-bias-linear
)

MIXED_PRETRAIN_ARGS=(
    # 开启CPU绕转
    --hetero-use-cpu-communication
    --use-tp-pp-dp-mapping
    --micro-batch-size-per-dp 1 2 1 1
    # --num-micro-batches-per-dp 1 10 1 10
)

TRAINING_ARGS=(
    # --micro-batch-size 1
    --global-batch-size 30
    --train-iters 30
    --weight-decay 1e-2
    --use-distributed-optimizer
    --clip-grad 1.0
    --fp16
    --lr 0.00015
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction .01
    --lr-decay-iters 320000
    --adam-beta1 0.9
    --adam-beta2 0.95
    --attention-dropout 0
    --hidden-dropout 0
    --untie-embeddings-and-output-weights
    --sequence-parallel
    --distributed-backend nccl
)

MODEL_PARALLEL_ARGS=(
    --transformer-impl local
    --use-legacy-models
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 2
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    # --save-interval 10000
    # --eval-interval 1000
    # --save $CHECKPOINT_PATH
    # --load $CHECKPOINT_PATH
    # --eval-iters 10
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH
)

cmd="
torchrun ${DISTRIBUTED_ARGS[@]} ../../../pretrain_gpt.py \
    ${LLAMA_MODEL_ARGS[@]} \
    ${MIXED_PRETRAIN_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
"

echo $cmd
eval $cmd


