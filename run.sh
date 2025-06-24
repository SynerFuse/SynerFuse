#!/bin/bash

# Runs the "70B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6007
NNODES=1
NODE_RANK=0

CHECKPOINT_PATH=/data2/share/llama-dataset/cp
TENSORBOARD_LOGS_PATH=/data2/share/llama-dataset/tb
TOKENIZER_PATH=/data2/nfs/llama-dataset/tokenizer.model
DATA_PATH=/data2/nfs/llama-dataset/merged-1t/merged-1t

# 7 B
HIDDEN_SIZE=4096 
FFN_HIDDEN_SIZE=11008
NUM_LAYERS=8
NUM_HEADS=32 
SEQ_LENGTH=4096

TRAIN_STEPS=5
# LR=3e-4
# MIN_LR=3e-5
# LR_WARMUP_STEPS=1
# WEIGHT_DECAY=0.1
# GRAD_CLIP=1

TP=2
PP=2
MBS=2

GBS=128


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

HETERO_ARGS=(
    # hetero pp config
    --hetero-pipeline-stages 1 2 1 6
    
    # Hetero dp config
    # --use-tp-pp-dp-mapping 
    # --micro-batch-size-per-dp 1 2 1 6 
    # --num-micro-batches-per-dp 1 1 1 1
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

SCRIPT_FILE=$(pwd)/pretrain_gpt.py

cmd="
torchrun ${DISTRIBUTED_ARGS[@]} ${SCRIPT_FILE} \
    ${LLAMA_MODEL_ARGS[@]} \
    ${HETERO_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
"

echo $cmd
eval $cmd
