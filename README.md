# SynerFuse
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10803/badge)](https://www.bestpractices.dev/projects/10803)
## Latest News
- [2025/6]SynerFuse publishes framework of Release 1.0, which can provide heterogeneous pipeline parallelism and heterogeneous data parallelism capabilities.
- [2025/5] China Mobile has established the repository for SynerFuse and obtained the OpenSSF certification badge.
## SynerFuse Overview
Currently, the “resource wall” between different GPUs makes it difficult to build one heterogeneous resource pool for Large-scale models training. Heterogeneous distributed training becomes a pressing challenge for the industry to solve. We brought up a cross-architecture unified heterogeneous training framework SynerFuse to deal with the problem.

SynerFuse enables multiple LLMs deployed and trained on multiple types of GPUs. The Inhomogeneous Task Distribution(ITD) algorithm for heterogeneous training task splitting is innovatively proposed, which supports heterogeneous data parallelism(DP) and heterogeneous pipeline parallelism(PP), and realizes the adaptive adjustment of parameters such as size and number of micro batches in DP, stages and layers in PP on heterogeneous GPUs.

We’ve verified our capability on Nvidia and other 4 types of GPUs. The acceleration ratio reached 95%, loss converges to 1.8 and PPL curve converged normally.
## Quick Start

### Environment Preparation

SynerFuse is recommended to run in a containerized environment based on NVIDIA's NGC PyTorch image, which ensures compatibility with GPU-accelerated libraries.

- Ubuntu 20.04 / 22.04
- Python 3.10+
- PyTorch 2.4.0+ (with GPU support)
- NVIDIA GPUs with a recent driver (the NGC container includes CUDA and cuDNN)

### Setup

We recommend using NGC's PyTorch container (e.g., [nvcr.io/nvidia/pytorch:24.02-py3](https://nvcr.io/nvidia/pytorch:24.02-py3)) for setup.

1. Create and start the container:

   ```bash
   docker run -itd \
     --name synerfuse \
     --gpus all \
     --network=host \
     --ipc=host \
     --privileged \
     nvcr.io/nvidia/pytorch:24.07-py3 \
     bash
   ```

2. Access the container:

   ```bash
   docker exec -it synerfuse bash
   ```

3. Clone the repository:

   ```shell
   git clone https://github.com/SynerFuse/SynerFuse
   ```

4. Install dependencies:

   ```bash
   pip install sentencepiece
   ```

### Prepare Data

Before running training tasks, prepare the following data files:

1. **Tokenizer Model**: Download or prepare a tokenizer model file (e.g., `tokenizer.model`).
2. **Training Dataset**: Prepare training data in Megatron-LM format (`.bin` and `.idx` files).

Update the following paths in `run.sh` according to your data location:

```bash
TOKENIZER_PATH=/path/to/your/tokenizer.model
DATA_PATH=/path/to/your/dataset
```

### Configure Paths

Configure directories for checkpoints and TensorBoard logs in `run.sh`:

```bash
CHECKPOINT_PATH=/path/to/your/checkpoints
TENSORBOARD_LOGS_PATH=/path/to/your/tensorboard_logs
```

Make sure these directories exist inside the container, for example:

```bash
mkdir -p /workspace/checkpoints
mkdir -p /workspace/tensorboard_logs
```

### Run a Task

SynerFuse provides a simple bash script for running pre-training tasks. To start distributed heterogeneous training:

```bash
cd SynerFuse
bash run.sh
```

The script launches a distributed training job with default configurations. 

Monitor training progress through the console output and (optionally) TensorBoard if enabled.

### Heterogeneous Parallelism Configuration

#### Heterogeneous Pipeline Parallelism

The `--hetero-pipeline-stages` parameter configures different layer counts for different pipeline stages.

**Format**: `n0 layers_0_0 layers_0_1 ... n1 layers_1_0 layers_1_1 ...`

- `n0`: Number of devices in the 0-th heterogeneous stage, followed by layer counts for each device in that stage (`layers_0_0`, `layers_0_1`, ...)
- `n1`: Number of devices in the 1-st heterogeneous stage, followed by layer counts for each device in that stage (`layers_1_0`, `layers_1_1`, ...)
- Additional stages follow the same pattern: `n2 layers_2_0 layers_2_1 ...`

**Constraints**:

$$
\sum_{i=0}^{k-1} n_i  = \text{pipeline-model-parallel-size}
$$

$$
\text{num-layers} = \sum_{i=0}^{k-1} \sum_{j=0}^{n_i-1} \text{layers}_{i,j}
$$

Where:

- $k$ is the number of heterogeneous stages
- $n_i$ is the number of devices in the i-th stage
- $\text{layers}_{i,j}$ is the layer count for the j-th device in the i-th stage

This parameter enables assigning more layers to more powerful GPUs and fewer layers to less powerful GPUs in a heterogeneous GPU cluster.

**Example**: For a pipeline with 2 stages and 8 layers in total:

```bash
--num-layers 8
--pipeline-model-parallel-size 2
--hetero-pipeline-stages 1 6 1 2
```

This configuration specifies:

- Stage 0: 1 device with 6 layers (`layers_0_0 = 6`)
- Stage 1: 1 device with 2 layers (`layers_1_0 = 2`)

#### Heterogeneous Data Parallelism

Configure heterogeneous DP using three parameters:

- `--use-tp-pp-dp-mapping`: Changes communication group order to enable heterogeneous DP
- `--micro-batch-size-per-dp`: Sets micro-batch size for different DP groups
- `--num-micro-batches-per-dp`: Sets number of micro-batches for different DP groups

**Format for `--micro-batch-size-per-dp`**: `n0 mbs0 n1 mbs1 ...`

- `n0, n1, ...`: Number of consecutive devices within a DP group
- `mbs0, mbs1, ...`: Micro-batch size for the corresponding device group

**Format for `--num-micro-batches-per-dp`**: `n0 nmb0 n1 nmb1 ...`

- `n0, n1, ...`: Number of consecutive devices within a DP group
- `nmb0, nmb1, ...`: Number of micro-batches for the corresponding device group

**Constraints**:

$$
\sum_{i} n_i = \text{data-parallel-size}
$$

$$
\text{global-batch-size} = \sum_{i} n_i \times \text{mbs}_i \times \text{num-mbs}_i
$$

**Example**: For data-parallel-size=2 and global-batch-size=32:

```shell
--global-batch-size 32
--use-tp-pp-dp-mapping 
--micro-batch-size-per-dp 1 6 1 2
--num-micro-batches-per-dp 1 4 1 4
```

This configuration creates two DP groups: one device with micro-batch size 6 and 4 micro-batches (6×4=24), and another device with micro-batch size 2 and 4 micro-batches (2×4=8), summing to 32.

### Customize Training

Modify variables in `run.sh` to customize training:

- **Model Size**: Adjust `HIDDEN_SIZE`, `NUM_LAYERS`, `NUM_HEADS`, etc.
- **Training Steps**: Set `TRAIN_STEPS` to control iterations.
- **Batch Sizes**: Modify `MBS` (micro-batch size) and `GBS` (global batch size).
- **Parallelism**: Adjust `TP` (tensor parallel size) and `PP` (pipeline parallel size).
- **Heterogeneous Configuration**: Modify `--hetero-pipeline-stages` and related parameters.

### Notes

- Ensure available GPUs match `CUDA_VISIBLE_DEVICES` and `GPUS_PER_NODE` configurations.
- The training script uses `torchrun` for distributed training.
- Check console output for training progress and potential errors.
- TensorBoard logs are saved to `TENSORBOARD_LOGS_PATH` if configured.
