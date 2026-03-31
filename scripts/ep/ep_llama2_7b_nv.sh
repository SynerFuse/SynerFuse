#!/bin/bash

# Runs the "7B" parameter model
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMPI_MCA_btl_tcp_if_include=ens3f0
export NCCL_SOCKET_IFNAME=ens3f0
export GLOO_SOCKET_IFNAME=ens3f0
export NCCL_IB_HCA=mlx5_0
source /opt/dtk/env.sh
export LD_LIBRARY_PATH=/root/nccl/build/lib:$LD_LIBRARY_PATH


export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_NET_GDR_READ=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_TIMEOUT=22
export NCCL_IBEXT_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=4
export NCCL_MIN_P2P_NCHANNELS=4
export NCCL_MAX_P2P_NCHANNELS=4
export NCCL_NCHANNELS_PER_PEER=1
export NCCL_PROTO=Simple
#export NCCL_PROTO=LL
export NCCL_ALGO=RING
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export HIP_VISIBLE_DEVICES=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=10.21.18.44
MASTER_PORT=60006
NNODES=1
NODE_RANK=0

#CHECKPOINT_PATH=/data2/share/llama-dataset/cp
#TENSORBOARD_LOGS_PATH=/data2/share/llama-dataset/tb
TOKENIZER_PATH=/data2/nfs/llama-dataset/tokenizer.model
DATA_PATH=/data2/nfs/llama-dataset/merged-1t/merged-1t

# 7 B
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008
NUM_LAYERS=16
NUM_HEADS=32
SEQ_LENGTH=4096

TRAIN_STEPS=20
# LR=3e-4
# MIN_LR=3e-5
# LR_WARMUP_STEPS=1
# WEIGHT_DECAY=0.1
# GRAD_CLIP=1




MBS=1


PP=2

TP=2

GBS=30

EP=2


# result_file="llama7b_training"
# DATE=$(date +%y%m%d%H%M%S)
# LOG_NAME="7b_pp_28_2_2"
# LOG_DIR=./${result_file}/${LOG_NAME}/log_${DATE}_nv
# mkdir -p ${LOG_DIR}


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

LLAMA_MODEL_ARGS=(
    --micro-batch-size ${MBS}
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads ${NUM_HEADS}
    --seq-length ${SEQ_LENGTH}
    --max-position-embeddings ${SEQ_LENGTH}
    --num-query-groups 8
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model $TOKENIZER_PATH
    --swiglu
    --use-flash-attn
    --use-rotary-position-embeddings
    --no-position-embedding
    --disable-bias-linear
)

MIXED_PRETRAIN_ARGS=(
    --enable-hetero
    --hetero-process-meshes    2 1 2 2 1   2 1 2 2 1
    --hetero-device-types H100 H100
    --hetero-current-device-type H100
    --num-experts 2
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
)

TRAINING_ARGS=(
    --global-batch-size ${GBS}
    --train-iters ${TRAIN_STEPS}
    --weight-decay 1e-2
    --use-distributed-optimizer
    --clip-grad 1.0
    # --fp16
    --bf16
    --attention-softmax-in-fp32
    --lr 0.00015
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction .01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --attention-dropout 0
    --hidden-dropout 0
    --untie-embeddings-and-output-weights
    --sequence-parallel
    --distributed-backend nccl
    --initial-loss-scale 65536
    --min-loss-scale 1.0
    --loss-scale-window 1024
    --transformer-impl transformer_engine
    # --use-legacy-models
    # --use-tp-pp-dp-mapping
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --split 1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --timing-log-level 1
)

INITIALIZATION_ARGS=(
    --init-method-std 0.02
    --seed 1234
)

SCRIPT_FILE=/data2/nfs/zhurui/cmcc/SynerFuse_with_tp_cp/pretrain_llama.py

cmd="
torchrun ${DISTRIBUTED_ARGS[@]} ${SCRIPT_FILE} \
    ${LLAMA_MODEL_ARGS[@]} \
    ${MIXED_PRETRAIN_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
"

echo $cmd
eval $cmd

