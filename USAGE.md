# SynerFuse

[](https://www.bestpractices.dev/projects/10699)

## Heterogeneous Configuration Methods

### Heterogeneous DP Configuration

#### \--use-tp-pp-dp-mapping 

Changes the communication group order to enable heterogeneous DP.

#### \--micro-batch-size-per-dp

Sets the micro-batch size for different data parallel groups.

  - Format: `n0 mbs0 n1 mbs1 ...`

      - `n0, n1, ...`: Number of consecutive devices within a data parallel group.
      - `mbs0, mbs1, ...`: Micro-batch size for the corresponding device group.

  - Constraints:


$$
\sum_{i} n_i = \text{data-parallel-size}
$$

$$
\text{global-batch-size} = \sum_{i} n_i \times \text{mbs}_i \times \text{num-mbs}_i
$$

#### \--num-micro-batches-per-dp

Sets the number of micro-batches for different data parallel groups.

  - Format: `n0 nmb0 n1 nmb1 ...`

      - `n0, n1, ...`: Number of consecutive devices within a data parallel group.
      - `nmb0, nmb1, ...`: Number of micro-batches for the corresponding device group.

  - Constraints:

$$\sum_{i} n_i = \text{data-parallel-size}$$

$$\text{global-batch-size} = \sum_{i} n_i \times \text{mbs}_i \times \text{num-mbs}_i$$

### Heterogeneous PP Configuration

#### --hetero-pipeline-stages

Used to configure different numbers of layers for different stages.

- Format: `n0 layers_0_0 layers_0_1 ... n1 layers_1_0 layers_1_1 ...`

  - `n0` represents the number of devices in the 0th heterogeneous stage, followed by the number of layers for each layer in that stage;
  - `n1` represents the number of devices in the 1st heterogeneous stage, followed by the number of layers for each layer in that stage, and so on.
