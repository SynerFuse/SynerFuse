# SynerFuse
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10699/badge)](https://www.bestpractices.dev/projects/10699)
## Latest News
- [2025/6]SynerFuse publishes framework of Release 1.0, which can provide heterogeneous pipeline parallelism and heterogeneous data parallelism capabilities.
- [2025/5] China Mobile has established the repository for SynerFuse and obtained the OpenSSF certification badge.
## SynerFuse Overview
Currently, the “resource wall” between different GPUs makes it difficult to build one heterogeneous resource pool for Large-scale models training. Heterogeneous distributed training becomes a pressing challenge for the industry to solve. We brought up a cross-architecture unified heterogeneous training framework SynerFuse to deal with the problem.

SynerFuse enables multiple LLMs deployed and trained on multiple types of GPUs. The Inhomogeneous Task Distribution(ITD) algorithm for heterogeneous training task splitting is innovatively proposed, which supports heterogeneous data parallelism(DP) and heterogeneous pipeline parallelism(PP), and realizes the adaptive adjustment of parameters such as size and number of micro batches in DP, stages and layers in PP on heterogeneous GPUs.

We’ve verified our capability on Nvidia and other 4 types of GPUs. The acceleration ratio reached 95%, loss converges to 1.8 and PPL curve converged normally.
