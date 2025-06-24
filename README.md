# SynerFuse
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10699/badge)](https://www.bestpractices.dev/projects/10699)
## 异构配置方法

### 异构DP配置

#### --use-tp-pp-dp-mapping 

改变通信组顺序，让DP走异构。

#### --micro-batch-size-per-dp

为不同的数据并行组设置微批次大小。

- 格式：`n0 mbs0 n1 mbs1 ...`

  - `n0, n1, ...`：数据并行组内连续设备个数。
  - `mbs0, mbs1, ...`：对应设备组的微批次大小。

- 约束：
  $$
  \sum_{i} n_i = \text{data-parallel-size}
  $$

  $$
  \text{GBS} \mod \left( \sum_{i} n_i \times \text{mbs}_i \right) = 0
  $$

#### --num-micro-batches-per-dp

为不同的数据并行组设置微批次数量。

- 格式：`n0 nmb0 n1 nmb1 ...`

  - `n0, n1, ...`：数据并行组内连续设备个数。
  - `nmb0, nmb1, ...`：对应设备组的微批次数量。

- 约束：
  $$
  \sum_{i} n_i = \text{data-parallel-size}
  $$

  $$
  \text{global\_batch\_size} = \sum_{i} n_i \times \text{mbs}_i \times \text{num\_mbs}_i
  $$

---

### 异构PP配置

#### --hetero-pipeline-stages

用于给不同的阶段配置不同的层数。

- 格式：`n0 layers_0_0 layers_0_1 ... n1 layers_1_0 layers_1_1 ...`

  - `n0` 表示第 0 个异构阶段的设备数量，后续跟随该阶段各层的层数；
  - `n1` 表示第 1 个异构阶段的设备数量，后续跟随该阶段各层的层数，以此类推。