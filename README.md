# Multi-agent Dynamic Parameter Sharing (MADPS)

[![AAMAS 2024](https://img.shields.io/badge/AAMAS-2024-blue)](https://aamas2024-conference.auckland.ac.nz/)
[![AAMAS 2026](https://img.shields.io/badge/AAMAS-2026-orange)](https://arxiv.org/abs/2512.22941)
[![arXiv](https://img.shields.io/badge/arXiv-2401.11257-red)](https://arxiv.org/abs/2401.11257)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

---

## 🏛️ 关于本仓库 | About This Repository

本仓库由 **[CASIA-Collect-AI](https://github.com/CASIA-Collect-AI)** 收录维护，作为多智能体强化学习优质论文代码的集合。

📌 **原始仓库（推荐访问）：** [Harry67Hu/MADPS](https://github.com/Harry67Hu/MADPS)
⭐ **如果本工作对你有帮助，请前往原始仓库点 Star 支持作者！**

> CASIA-Collect-AI 是中国科学院自动化研究所 AI 团队维护的开源代码收录平台，专注于收录和整理 MARL、LLM、机器人等领域的高质量研究代码。

---

Official implementation of **Measuring Policy Distance for Multi-Agent Reinforcement Learning** (AAMAS 2024).

**Authors:** Tianyi Hu, Zhiqiang Pu, Xiaolin Ai, Tenghai Qiu, Jianqiang Yi
**Affiliations:** Institute of Automation, Chinese Academy of Sciences; National Key Laboratory of Cognition and Decision Intelligence for Complex Systems; University of Chinese Academy of Sciences

---

## Abstract

This paper proposes **MADPS** (Multi-agent Dynamic Parameter Sharing), a method for measuring *policy distance* in multi-agent reinforcement learning (MARL) and utilizing it for dynamic parameter sharing.

- **Policy Distance Measurement:** We train a Conditional VAE to learn conditional representations of agents' decisions, then compute multi-agent policy distance matrices between agents.
- **Dynamic Parameter Sharing:** Based on the policy distance matrix, we automatically adjust the parameter sharing scheme among agents, enabling more flexible and interpretable multi-agent learning.

---

## 📖 论文深度解读

### 核心问题：多智能体参数共享的困境

在多智能体强化学习中，**参数共享** 是提升样本效率的核心技术，但传统方法存在明显缺陷：

- **全参数共享（FPS）**：强迫所有智能体使用同一网络。当智能体有不同目标时（如不同颜色智能体需向不同方向运动），冲突的经验导致网络剧烈波动——在稀疏奖励场景中甚至完全失效（reward ≈ 0）。
- **选择性参数共享（SePS）**：基于预学习表示做固定聚类，方案在训练中不再更新，无法捕捉策略的动态演化；且每个任务都需要手动调整超参数（聚类数 K），泛化性差。

**根本缺口**：缺乏通用指标来量化智能体间策略差异，无从判断"哪些智能体应该共享参数"。

---

### 方法核心：MAPD + MADPS

#### 第一步：用 Conditional VAE 学习策略表示

直接比较两个智能体的动作分布存在困难——不同智能体可能面对不同观测空间或拥有异构动作类型。MADPS 训练一个 **CVAE** 将策略映射到统一的隐空间：

- **编码器**：输入 `(动作分布 π_i(a|o), 观测 o)`，输出隐变量后验 `q(z|π_i, o)`
- **解码器**：从 `z` 和 `o` 重建动作分布
- **训练直接使用策略分布**，避免从环境采样的低效问题

#### 第二步：计算多智能体策略距离（MAPD）

在隐空间中使用 **Wasserstein 距离** 对隐分布求积分：

```
d_ij = ∫ W[p_i(z|o), p_j(z|o)] do
```

将高斯分布、双峰分布、离散分布等各种形态统一到可比较的隐空间，实现跨智能体的通用策略距离度量。选用 Wasserstein 而非 KL 散度的原因：在分布无重叠时 Wasserstein 距离仍有意义，对 MARL 中常见的模式分离场景更加鲁棒。

#### 第三步：MADPS 动态参数共享

基于策略距离矩阵，每隔 T 步执行合并/分离操作：

- **合并（Fusion）**：`d_ij < ε₁` → 参数合并以提升效率
- **分离（Division）**：`d_ij > ε₂` → 参数解耦以维持多样性
- **设计约束**：`ε₂ ≥ 2ε₁`（由三角不等式推导，避免频繁震荡）

参数共享方案随训练进程**自适应调整**，无需手动设定聚类数。

---

### 实验分析

**测试环境：PettingZoo MPE Large-Spread（6个难度变体 v1–v6）**

| 变体 | 智能体数 | 地标数 | 特点 |
|------|---------|-------|------|
| v1 | 15 | 3 | 基础版本 |
| v2 | 30 | 3 | 大规模 |
| v3–v6 | 30 | 5 | 异构/乱序观测 |

**关键发现：**
- FPS 在稀疏奖励任务上**完全失败**（异构目标导致梯度冲突）
- MADPS 在全部 6 个变体上**一致超越所有 baseline**，且只需 1 组超参数，无需任务特定调整
- SePS 在异构地图上表现不稳定，需逐任务手动调整 K

---

### 🚀 拓展工作：HetDPS（AAMAS 2026）

> 完整拓展工作已发表于 AAMAS 2026！见 [arXiv:2512.22941](https://arxiv.org/abs/2512.22941)

MADPS 仅利用了**策略异质性**。HetDPS 将其扩展至完整的异质性框架：

**五种异质性类型：**
1. **观测异质性**：智能体对全局信息感知方式的差异
2. **响应转移异质性**：环境因素对各智能体状态影响的不同
3. **效果转移异质性**：智能体动作对整体系统影响的差异
4. **目标异质性**：奖励函数的差异
5. **策略异质性**：基于观测的决策方式差异（MADPS 仅覆盖此项）

**HetDPS 通过 Meta-Heterogeneity Distance 综合量化各维度异质性**，消除了任务特定超参数，在 MPE 和 SMAC 环境上均达到最优或可比性能。

---

## Installation

```bash
conda create --name madps --file requirements.txt
conda activate madps
```

Replace `pettingzoo/mpe/scenarios/large_spread.py` with `large_spread_example.py` from this repo to access the upgraded task.

---

## Quick Start

```bash
python ac_NF.py with env_name='pettingzoo:pz-mpe-large-spread-v1' time_limit=50
python ac_NF.py with env_name='pettingzoo:pz-mpe-large-spread-v2' time_limit=50
# v3–v6 similarly
```

---

## Citation

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
