# Multi-agent Dynamic Parameter Sharing (MADPS)

[![AAMAS 2024](https://img.shields.io/badge/AAMAS-2024-blue)](https://aamas2024-conference.auckland.ac.nz/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **Measuring Policy Distance for Multi-Agent Reinforcement Learning** (AAMAS 2024).

**Authors:** Tianyi Hu, Zhiqiang Pu, Xiaolin Ai, Tenghai Qiu, Jianqiang Yi  
**Affiliations:** Institute of Automation, Chinese Academy of Sciences; National Key Laboratory of Cognition and Decision Intelligence for Complex Systems; University of Chinese Academy of Sciences

---

## Abstract

This paper proposes **MADPS** (Multi-agent Dynamic Parameter Sharing), a method for measuring *policy distance* in multi-agent reinforcement learning (MARL) and utilizing it for dynamic parameter sharing.

- **Policy Distance Measurement:** We train a Conditional VAE to learn conditional representations of agents' decisions, then compute multi-agent policy distance matrices (Bhattacharyya, Hellinger, Wasserstein) between agents.
- **Dynamic Parameter Sharing:** Based on the policy distance matrix, we automatically adjust the parameter sharing scheme among agents, enabling more flexible and interpretable multi-agent learning.

---

## 🚀 Extended Work: HetDPS (AAMAS 2026)

> **本工作的全面扩充版已发表于 AAMAS 2026！**
>
> 我们的后续工作 [**Heterogeneity in Multi-Agent Reinforcement Learning**](https://arxiv.org/pdf/2512.22941) 在本文基础上进行了系统性拓展：
>
> - **更丰富的异构类型：** 不仅限于策略异构（policy heterogeneity），还系统讨论了观测异构、转移异构、目标异构等多种异构距离的量化方法
> - **升级版算法 HetDPS：** 提出 Meta-Heterogeneity Distance 与 Heterogeneity-based Dynamic Parameter Sharing，具有更好的可解释性与更少的任务相关超参数
> - **更完整的理论框架：** 从*定义*、*量化*、*利用*三个维度系统阐述 MARL 中的异构性
>
> 欢迎阅读 [论文](https://arxiv.org/pdf/2512.22941) 获取完整理论与方法。

---

## Installation

### Requirements

We recommend using a Conda virtual environment:

```bash
conda create --name madps --file requirements.txt
conda activate madps
```

> **Note:** We will update the docker approach soon. Some libraries may have dependency order issues; if automatic installation fails, try installing them manually.

### Supported MARL Environments

- **PettingZoo MPE** (Multi-agent Spread): [PettingZoo Version](https://github.com/semitable/PettingZoo)

> **Note:** In our research, we have upgraded the task's difficulty level. Replace the file at `pettingzoo/mpe/scenarios/large_spread.py` with `large_spread_example.py` from this repository to access the updated task.

---

## Quick Start

Run MADPS on the Particle-based Multi-agent Spreading environment:

```bash
python ac_NF.py with env_name='pettingzoo:pz-mpe-large-spread-v1' time_limit=50
```

Run on different scenario versions (v1–v6 correspond to `15a_3c`, `30a_3c`, `30a_5c`, `30a_5c_super`, `15a_3c_shuffle`, `30a_3c_shuffle`):

```bash
python ac_NF.py with env_name='pettingzoo:pz-mpe-large-spread-v2' time_limit=50
python ac_NF.py with env_name='pettingzoo:pz-mpe-large-spread-v3' time_limit=50
# ... v4, v5, v6
```

---

## Repository Structure

| File | Description |
|------|-------------|
| `ac_NF.py` | Main training loop: multi-agent environment construction, sampling, actor-critic training |
| `MADPS_NF.py` | Policy distance computation (MAPD) and dynamic parameter sharing (MADPS) |
| `model_NF.py` | Neural networks: `MADPSNet`, `MultiAgentFCNetwork`, `Policy`, `ConditionalVAE` |
| `large_spread_example.py` | Upgraded large-spread scenario for PettingZoo |
| `wrappers.py` | Environment wrappers for MARL benchmarks |

### Core Components

**1. Policy Distance Computing & MADPS (`MADPS_NF.py`)**

- `compute_fusions` — MAPD + MADPS: trains Conditional VAE, computes policy distance matrix (dij), adjusts parameter sharing
- `calculate_N_Gaussians_BD` — Bhattacharyya distance (parallel, PyTorch)
- `calculate_N_Gaussians_Hellinger_through_BD` — Hellinger distance
- `calculate_N_Gaussians_WD` — Wasserstein distance

**2. Neural Network Models (`model_NF.py`)**

- `MADPSNet` — Multi-agent network with dynamic and hierarchical parameter sharing
- `MultiAgentFCNetwork` — SePS-style parameter sharing (replication)
- `Policy` — Multi-agent policy models
- `ConditionalVAE` — CVAE for conditional representations of agents' decisions

**3. Training Framework (`ac_NF.py`)**

- Multi-agent environment construction and maintenance
- Environment sampling and sample pool construction
- Neural network evaluation and training

---

## Paper

[Measuring Policy Distance for Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2401.11257.pdf) (arXiv)

> **Note:** Since there is no appendix on the AAMAS official link, we provide the appendix in this repository (`Appendices(657).pdf`).

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{hu2024MAPD,
  title={Measuring Policy Distance for Multi-Agent Reinforcement Learning},
  author={Hu, Tianyi and Pu, Zhiqiang and Ai, Xiaolin and Qiu, Tenghai and Yi, Jianqiang},
  booktitle={Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2024)},
  pages={834--842},
  year={2024}
}
```

---

## Contact

- **First Author:** hutianyi2021@ia.ac.cn (Tianyi Hu)
- **Corresponding Author:** zhiqiang.pu@ia.ac.cn (Zhiqiang Pu)
