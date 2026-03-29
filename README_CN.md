# 多智能体动态参数共享（MADPS）

[English](README.md) | [中文](README_CN.md)

[![AAMAS 2024](https://img.shields.io/badge/AAMAS-2024-blue)](https://aamas2024-conference.auckland.ac.nz/)
[![AAMAS 2026](https://img.shields.io/badge/AAMAS-2026-orange)](https://arxiv.org/abs/2512.22941)
[![arXiv](https://img.shields.io/badge/arXiv-2401.11257-red)](https://arxiv.org/abs/2401.11257)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

---

## 🏛️ 关于本仓库

本仓库由 **[CASIA-Collect-AI](https://github.com/CASIA-Collect-AI)** 收录维护，作为高质量 MARL 研究代码的精选集合。

📌 **原始仓库（推荐访问）：** [Harry67Hu/MADPS](https://github.com/Harry67Hu/MADPS)
⭐ **如果本工作对你有帮助，请前往原始仓库点 Star 支持作者！**

> **团队：** 中国科学院自动化研究所 飞行器智能技术团队（群体智能团队-蒲志强）
> CASIA-Collect-AI 收录和维护 MARL、LLM 和机器人领域的高质量开源研究代码。

---

**《面向多智能体强化学习的策略距离度量》** 官方实现（AAMAS 2024）

**作者：** 胡天一, 蒲志强, 艾晓林, 仇腾海, 易建强
**单位：** 中国科学院自动化研究所；认知与决策智能复杂系统国家重点实验室；中国科学院大学

---

## 摘要

本文提出 **MADPS**（多智能体动态参数共享），一种在多智能体强化学习（MARL）中度量*策略距离*并将其用于动态参数共享的方法。

- **策略距离度量：** 训练条件变分自编码器（CVAE）学习智能体决策的条件表示，进而计算智能体间的多智能体策略距离矩阵。
- **动态参数共享：** 基于策略距离矩阵，自动调整智能体间的参数共享方案，实现更灵活、可解释的多智能体学习。

<div align="center"><img src="imgs/fig_overview.png" width="620"></div>

*本文贡献与多智能体强化学习的关系（粗斜体为核心贡献）*

---

## 📖 论文深度解读

### 核心问题：MARL 中的参数共享困境

**参数共享**是提升 MARL 样本效率的核心技术，但传统方法存在明显缺陷：

- **完全参数共享（FPS）：** 强制所有智能体使用同一网络。当智能体目标不同时（如不同颜色的智能体需向不同方向移动），梯度冲突导致网络震荡，在稀疏奖励下（reward≈0）甚至完全失败。
- **选择性参数共享（SePS）：** 基于预学习表示进行固定聚类。训练期间聚类方案冻结，错过了策略的动态演化；每个新任务都需要手动调整超参数（聚类数 K），泛化能力有限。

**根本缺口：** 缺乏量化智能体间策略差异的通用度量，无法判断哪些智能体*应该*共享参数。

---

### 方法：MAPD + MADPS

#### 第一步：用条件 VAE 学习策略表示

直接比较智能体间的动作分布存在困难——不同智能体可能面临不同的观测空间或异构动作类型。MADPS 训练一个 **CVAE** 将所有策略映射到统一的隐空间：

- **编码器：** 输入 `(动作分布 π_i(a|o), 观测 o)` → 隐后验 `q(z|π_i, o)`
- **解码器：** 从 `z` 和 `o` 重建动作分布
- 训练直接使用已知策略分布，**避免了环境采样的低效性**

<div align="center"><img src="imgs/fig_cvae.png" width="700"></div>

*通过 CVAE 学习智能体决策的条件表示*

#### 第二步：计算多智能体策略距离（MAPD）

在隐空间中，通过对观测积分的 **Wasserstein 距离** 计算策略距离：

```
d_ij = ∫ W[p_i(z|o), p_j(z|o)] do
```

这将高斯分布、双峰分布和离散分布统一到可比较的隐空间。选择 Wasserstein 而非 KL 散度，是因为即使分布没有重叠时 Wasserstein 仍然有意义——这对 MARL 中常见的模式分离现象至关重要。

#### 第三步：动态参数共享（MADPS）

基于策略距离矩阵，MADPS 每 T 步执行合并/分裂操作：

- **融合：** `d_ij < ε₁` → 合并参数以提升效率
- **分裂：** `d_ij > ε₂` → 解耦参数以保持多样性
- **设计约束：** `ε₂ ≥ 2ε₁`（由三角不等式推导，防止震荡）

<div align="center"><img src="imgs/fig_dynamic_ps.png" width="620"></div>

*动态参数共享的基本思想：策略相似的智能体共享参数；策略差异大的智能体独立学习*

共享方案在整个训练过程中**自适应调整**——无需手动设置聚类数。

---

### 实验结果

**测试环境：** PettingZoo MPE Large-Spread（6个难度变体 v1–v6）

| 变体 | 智能体数 | 地标数 | 特点 |
|------|----------|--------|------|
| v1 | 15 | 3 | 基础 |
| v2 | 30 | 3 | 大规模 |
| v3–v6 | 30 | 5 | 异构 / 打乱观测 |

<div align="center"><img src="imgs/fig_results.png" width="750"></div>

*在多智能体扩散任务和 SMAC 超难任务上与基线方法的性能对比*

**核心发现：**
- **FPS 在稀疏奖励任务上完全失败**（异构目标导致梯度冲突）
- **MADPS 在所有 6 个变体上持续优于所有基线**，仅用一组超参数
- **SePS** 在异构地图上不稳定，且每个任务都需手动调整 K

---

## 🚀 扩展工作：HetDPS（AAMAS 2026）

> 完整扩展版已发表于 AAMAS 2026！参见 [arXiv:2512.22941](https://arxiv.org/abs/2512.22941)

MADPS 仅显式利用了**策略异构性**。HetDPS 将其扩展为完整的异构性框架：

**五种异构性类型：**
1. **观测异构性：** 智能体感知全局信息的方式不同
2. **响应转移异构性：** 环境因素影响每个智能体状态的方式不同
3. **效果转移异构性：** 智能体动作对整体系统影响的方式不同
4. **目标异构性：** 奖励函数不同
5. **策略异构性：** 基于观测的决策不同 *（MADPS 仅覆盖此项）*

<div align="center"><img src="imgs/fig_hetdps_method.png" width="750"></div>

*HetDPS：通过跨五种异构性类型的表示学习度量异构距离*

**HetDPS 核心优势：**
- 引入**元异构距离**，全面量化所有异构性维度
- 消除任务特定超参数（无需设置聚类数 K）
- 通过可视化距离矩阵增强可解释性

<div align="center"><img src="imgs/fig_hetdps_results.png" width="700"></div>

*HetDPS 在基于粒子的多智能体扩散任务上的结果，达到最优或相当的性能*

---

## 安装

```bash
conda create --name madps --file requirements.txt
conda activate madps
```

> 将 `pettingzoo/mpe/scenarios/large_spread.py` 替换为本仓库中的 `large_spread_example.py` 以使用升级后的任务变体。

---

## 快速开始

```bash
python ac_NF.py with env_name='pettingzoo:pz-mpe-large-spread-v1' time_limit=50
python ac_NF.py with env_name='pettingzoo:pz-mpe-large-spread-v2' time_limit=50
# v3–v6 类似
```

---

## 引用

```bibtex
@inproceedings{hu2024MAPD,
  title={Measuring Policy Distance for Multi-Agent Reinforcement Learning},
  author={Hu, Tianyi and Pu, Zhiqiang and Ai, Xiaolin and Qiu, Tenghai and Yi, Jianqiang},
  booktitle={Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2024)},
  pages={834--842},
  year={2024}
}

@article{hu2025HetDPS,
  title={Heterogeneity in Multi-Agent Reinforcement Learning},
  author={Hu, Tianyi and Pu, Zhiqiang and others},
  journal={arXiv preprint arXiv:2512.22941},
  year={2025}
}
```

---

## 联系方式

- **第一作者：** hutianyi2021@ia.ac.cn（胡天一）
- **通讯作者：** zhiqiang.pu@ia.ac.cn（蒲志强）
