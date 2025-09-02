#!/bin/bash

# Runs the "70B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

export GLOO_SOCKET_IFNAME=ens21f0
export NCCL_SOCKET_IFNAME=ens21f0

#export GLOO_SOCKET_IFNAME=ibs20
#export NCCL_SOCKET_IFNAME=ibs20
#export NCCL_IB_DISABLE=0
#export NCCL_IB_CUDA_SUPPORT=1
#export NCCL_IB_GID_INDEX=4
#export NCCL_IB_RETRY_CNT=7
#
#export OMP_NUM_THREADS=4
#export NCCL_NET_SHARED_BUFFERS=0
#export NCCL_ALGO=Ring
#export NCCL_P2P_NET_CHUNKSIZE=1048576
#export NCCL_CHUNK_SIZE=1048576
#export NCCL_BUFFSIZE=8388608
#export NCCL_MAX_NCHANNELS=1
#export NCCL_MIN_NCHANNELS=1
#export NCCL_MAX_P2P_NCHANNELS=1
#export NCCL_PROTO=Simple
#export NCCL_P2P_LL_THRESHOLD=0
#export NCCL_NET_PLUGIN=none
#export NCCL_SHM_DISABLE=1
#
#export IXCCL_MIX_NV=1
#export IXCCL_FUSED_ENABLE=0

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=10.21.18.44
MASTER_PORT=34567
NNODES=2
NODE_RANK=1

# CHECKPOINT_PATH=/data2/share/llama-dataset/cp
# TENSORBOARD_LOGS_PATH=/data2/share/llama-dataset/tb
TOKENIZER_PATH=/data2/nfs/llama-dataset/tokenizer.model
DATA_PATH=/data2/nfs/llama-dataset/RedPajama-Data-1T-Sample/RedPajama-Data-1T-Sample


# 异构训练参数
HETERO_ARGS=(
    --distributed-timeout-minutes 1
    --hetero-use-cpu-communication
    --enable-hetero
    --hetero-process-meshes 2 1 1 4 1   2 1 4 4 1
    --hetero-device-types H100 BIV150S
    --hetero-current-device-type BIV150S
    --num-experts 8
)

# 分布式参数
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    # --log_dir ${LOG_DIR}/details
    # --rdzv_backend static
    # --rdzv_id default
    # --rdzv_endpoint localhost:46435
)

# 模型训练参数
TRAINING_ARGS=(
    # 模型结构
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 4
    --num-layers 32
    --hidden-size 1024
    --num-attention-heads 32
    --seq-length 4096
    --max-position-embeddings 4096
    --norm-epsilon 1e-05
    --use-rotary-position-embeddings
    --no-position-embedding
    --swiglu
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    --disable-bias-linear

    # 优化器
    --use-distributed-optimizer
    --adam-beta1 0.9
    --adam-beta2 0.95
    --weight-decay 0.1
    --clip-grad 1.0

    # 学习率
    --lr 0.00015
    --min-lr 1.5e-05
    --lr-warmup-samples 500
    --lr-decay-style cosine

    # 并行配置
     --sequence-parallel
    --transformer-impl transformer_engine
    --use-mcore-models

    # 内存优化
    --use-flash-attn
    # --use-legacy-models
    # --recompute-granularity full
    # --recompute-method uniform
    # --recompute-num-layers 1

    # 精度设置
    --bf16
    --attention-softmax-in-fp32
    --accumulate-allreduce-grads-in-fp32

    # 数据参数
    --global-batch-size 32
    --micro-batch-size 2
    --train-samples 100000
    --seed 42

    # 初始化参数
    --init-method-std 0.0165
    --attention-dropout 0.0
    --hidden-dropout 0.0
)

# 数据参数
DATA_ARGS=(
    --data-path ${DATA_PATH}
    --split 1
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_PATH}
    --vocab-size 64000
    --make-vocab-size-divisible-by 64
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
    ${HETERO_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
"

echo $cmd
eval $cmd
