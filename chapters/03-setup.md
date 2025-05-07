---
prev-chapter: "关键相关工作"
prev-url: "02-related-works.html"
page-title: 定义与背景
next-chapter: "训练概览"
next-url: "04-optimization.html"
---

# 定义与背景

本章收录了RLHF过程中常用的定义、符号与操作，并简要介绍了语言模型（本书讨论的主要优化对象）的基本知识。

## 语言建模概述

现代大多数语言模型的训练目标，是以自回归（autoregressive）方式学习一系列token（词、子词或字符）序列的联合概率分布。  
自回归的含义是：每个token的预测都依赖于序列中之前的内容。  
给定一个token序列 $x = (x_1, x_2, \ldots, x_T)$，模型会将整个序列的概率分解为一系列条件分布的乘积：

$$P_{\theta}(x) = \prod_{t=1}^{T} P_{\theta}(x_{t} \mid x_{1}, \ldots, x_{t-1}).$$ {#eq:llming}

为了让模型能够准确预测上述概率，训练目标通常是最大化模型对训练数据的似然。  
实际做法是最小化负对数似然（NLL）损失：

$$\mathcal{L}_{\text{LM}}(\theta)=-\,\mathbb{E}_{x \sim \mathcal{D}}\left[\sum_{t=1}^{T}\log P_{\theta}\left(x_t \mid x_{<t}\right)\right]. $$ {#eq:nll}

在实际工程中，通常采用交叉熵损失（cross-entropy loss）来度量每个token的预测，把模型预测与真实token逐一比对。

语言模型的实现形式多种多样。  
现代主流语言模型（如ChatGPT、Claude、Gemini等）大多采用**仅解码器结构的Transformer** [@Vaswani2017AttentionIA]。  
Transformer的核心创新在于充分利用**自注意力机制**（self-attention）[@Bahdanau2014NeuralMT]，使模型能够直接关注上下文中的概念，并学习复杂的映射关系。  
在本书（尤其是第7章奖励模型部分），我们会讨论如何为Transformer添加新的head或修改语言建模（LM）head。  
LM head是一个最终的线性投影层，用于将模型内部的embedding空间映射到分词器空间（即词表/vocabulary）。  
通过不同的head，可以复用模型内部结构，微调输出不同类型的结果。

## 机器学习相关定义

- **Kullback-Leibler (KL)散度（$D_{KL}(P || Q)$）**：又称KL散度，是衡量两个概率分布差异的指标。  
对于定义在同一概率空间$\mathcal{X}$上的离散概率分布$P$和$Q$，KL散度定义为：

$$ D_{KL}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) $$ {#eq:def_kl}

## 自然语言处理相关定义

- **Prompt（提示，$x$）**：输入给语言模型的文本，用于生成回复或补全。
- **Completion（补全，$y$）**：语言模型根据Prompt生成的输出文本。通常记作$y|x$。
- **Chosen Completion（选中补全，$y_c$）**：在多个补全中被选中或偏好的那个，常记为$y_{chosen}$。
- **Rejected Completion（被拒补全，$y_r$）**：在成对偏好场景下被认为不优的补全。
- **Preference Relation（偏好关系，$\succ$）**：表示一个补全优于另一个的符号，例如$y_{chosen} \succ y_{rejected}$。
- **Policy（策略，$\pi$）**：对所有可能补全的概率分布，由参数$\theta$控制，记作$\pi_\theta(y|x)$。

## 强化学习相关定义

- **Reward（奖励，$r$）**：用于衡量某个动作或状态优劣的标量，通常用$r$表示。
- **Action（动作，$a$）**：智能体在环境中做出的决策，常表示为$a \in A$，$A$为动作集合。
- **State（状态，$s$）**：环境当前的配置或情景，通常记为$s \in S$，$S$为状态空间。
- **Trajectory（轨迹，$\tau$）**：智能体经历的一系列状态、动作和奖励，$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, a_T, r_T)$。
- **Trajectory Distribution（轨迹分布，$(\tau|\pi)$）**：在策略$\pi$下轨迹的概率为 $P(\tau|\pi) = p(s_0)\prod_{t=0}^T \pi(a_t|s_t)p(s_{t+1}|s_t,a_t)$，其中$p(s_0)$为初始状态分布，$p(s_{t+1}|s_t,a_t)$为转移概率。
- **Policy（策略，$\pi$）**：在RLHF中也称为“策略模型”。策略是智能体在给定状态下选择动作的规则：$\pi(a|s)$。
- **Discount Factor（折扣因子，$\gamma$）**：$0 \le \gamma < 1$，用于对未来奖励进行指数衰减，权衡即时收益与长期收益，并保证无穷和的收敛。有时不使用折扣（$\gamma=1$）。
- **Value Function（价值函数，$V$）**：估算从某状态出发所能获得的期望累计奖励：$V(s) = \mathbb{E}[\sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s]$。
- **Q-Function（Q函数，$Q$）**：估算在某状态采取特定动作后所能获得的期望累计奖励：$Q(s,a) = \mathbb{E}[\sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s, a_0 = a]$。
- **Advantage Function（优势函数，$A$）**：$A(s,a)$衡量在状态$s$下采取动作$a$相对于平均动作的优势，定义为$A(s,a) = Q(s,a) - V(s)$。优势函数（和价值函数）可依赖于特定策略，记作$A^\pi(s,a)$。
- **策略条件下的取值（Policy-conditioned Values，$[]^{\pi(\cdot)}$）**：在RL推导和实现中，核心内容之一就是收集特定策略下的数据或数值。本书中会在简写（如$V,A,Q,G$）和带策略上标（如$V^\pi,A^\pi,Q^\pi$）间切换。计算期望时，需从特定策略$d_\pi$采样数据$d$。
- **奖励优化的期望值（Expectation of Reward Optimization）**：RL的主要目标是最大化期望累计奖励：

  $$\max_{\theta} \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta}[\sum_{t=0}^{\infty} \gamma^t r_t]$$ {#eq:expect_reward_opt}

  其中$\rho_\pi$为策略$\pi$下的状态分布，$\gamma$为折扣因子。

- **有限视野奖励（Finite Horizon Reward，$J(\pi_\theta)$）**：策略$\pi_\theta$在有限步长$T$下的期望累计奖励定义为：
$J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t r_t \right]$ {#eq:finite_horizon_return}
其中$\tau \sim \pi_\theta$表示按照策略$\pi_\theta$采样的轨迹，$T$为有限步长。

- **On-policy**：在RLHF领域，尤其是RL与直接对齐算法（Direct Alignment Algorithms）之争中，常讨论**on-policy**数据。在RL文献中，on-policy数据指完全由当前智能体生成的数据；而在偏好微调领域，on-policy也可指当前模型版本生成的内容——例如指令微调检查点在偏好微调前生成的数据。相对地，off-policy则指由其他模型在后训练阶段生成的数据。

## RLHF专有定义

- **参考模型（Reference Model，$\pi_\text{ref}$）**：RLHF中保存的一组参数，用于在优化过程中对输出进行正则化。

## 拓展术语表

- **合成数据（Synthetic Data）**：指由另一个AI系统生成的训练数据。可以是模型根据开放式提示生成的文本，也可以是模型对已有内容的重写等。
- **蒸馏（Distillation）**：一种AI训练实践，即用强模型的输出训练新模型。这是一种合成数据，常用于训练更小但效果强劲的模型。大多数模型会在开源权重协议或API服务条款中明确蒸馏相关规则。蒸馏（distillation）一词在ML文献中有更具体的技术定义。
- **（教师-学生）知识蒸馏（Knowledge Distillation）**：指从特定教师模型到学生模型的知识迁移，是蒸馏的具体形式，也是该术语的起源。其做法是：学生网络的损失函数被修改为学习教师模型对多个token/logit的概率分布，而非直接学习某个输出 [@hinton2015distilling]。现代采用知识蒸馏训练的模型如Gemma 2 [@team2024gemma]、Gemma 3等。在语言建模中，next-token损失可修改为 [@agarwal2024policy]，即学生模型$P_\theta$学习教师分布$P_\phi$：

$$\mathcal{L}_{\text{KD}}(\theta) = -\,\mathbb{E}_{x \sim \mathcal{D}}\left[\sum_{t=1}^{T} P_{\phi}(x_t \mid x_{<t}) \log P_{\theta}(x_t \mid x_{<t})\right]. $$ {#eq:knowledge_distillation}

- **上下文学习（In-context Learning, ICL）**：指在语言模型上下文窗口内添加信息，通常是将示例加入prompt。最简单的ICL就是在prompt前添加类似例子，进阶用法则能根据具体场景选择更合适的信息。
- **思维链（Chain of Thought, CoT）**：指引导语言模型以“分步推理”形式解决问题的行为。最初的实现方式是用提示词“Let's think step by step” [@wei2022chain]。