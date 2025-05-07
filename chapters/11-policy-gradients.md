---
prev-chapter: "拒绝采样"
prev-url: "10-rejection-sampling.html"
page-title: 策略梯度算法
next-chapter: "直接对齐算法"
next-url: "12-direct-alignment.html"
---

# 策略梯度算法（Policy Gradient Algorithms）

让RLHF在语言模型领域流行起来的，正是基于策略梯度（policy gradient）的强化学习算法。
这些算法（如PPO、GRPO、REINFORCE）利用最近生成的样本直接更新模型，而不是像传统强化学习那样依赖经验回放缓冲区（replay buffer）。
本节将介绍策略梯度算法的基本原理，以及它们在现代RLHF框架中的应用。

从机器学习角度看，策略梯度是RLHF流程中最复杂的部分之一。
不过，和大多数现代AI模型一样，其最终效果很大程度上仍取决于输入的数据质量。

RLHF常用的算法体系也在不断演变。
ChatGPT时代，大家公认其核心算法是PPO，许多早期工作也以此为基础。
后来，越来越多研究显示REINFORCE风格的算法也很有潜力 [@ahmadian2024back] [@wang2024helpsteer2p]，其优点是无需奖励模型（节省显存和GPU），且值函数估计更简单（无需GAE）。
此外，还有如Group Relative Policy Optimization（GRPO）等新算法，特别适用于推理类任务，但总体上这些算法都可针对具体任务进行调整和优化。
本章将重点介绍策略梯度的基本框架，以及PPO、GRPO和REINFORCE三大算法，因为它们构成了RLHF文献的核心。

相关符号定义请参见“问题设定”章节。

## 策略梯度算法基础

强化学习算法的目标是最大化状态$s \in \mathcal{S}$和动作$a \in \mathcal{A}$轨迹上的未来折扣奖励（见第3章定义）。
智能体的目标（return）是某一时刻$t$下未来折扣奖励的累加和（$\gamma\in [0,1)$为折扣因子，平衡即时与远期收益）：

$$G_t = R_{t+1} + \gamma R_{t+2} + \cdots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}.$$ {#eq:return_definition}

递推形式为：
$$G_{t} = \gamma G_{t+1} + R_{t+1}.$$ {#eq:recursive_return}

据此可定义值函数$V(s)$，即在当前状态下未来回报的期望：

$$V(s) = \mathbb{E}\big[G_t | S_t = s \big].$$ {#eq:value_function}

所有策略梯度算法的目标，都是优化由特定策略$\pi(s|a)$诱导的值函数。

其中$d_\pi(s)$为策略$\pi(s)$诱导的状态平稳分布，优化目标为：
$$
J(\theta)
\;=\;
\sum_{s} d_\pi(s) V_\pi(s),
$$ {#eq:policy_objective}

策略梯度算法的核心是计算当前策略下有限时间期望回报的梯度。
有了期望回报$J$，参数更新为：
$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$ {#eq:policy_update}

关键在于如何计算该梯度。
Schulman等（2015）对策略梯度的多种计算方式做了综述 [@schulman2015high]。
目标是*估算*精确梯度$g := \nabla_\theta \mathbb{E}[\sum_{t=0}^\infty r_t]$，常见形式如：

$$ g = \mathbb{E}\Big[\sum_{t=0}^\infty \Psi_t \nabla_\theta \text{log} \pi_\theta(a_t|s_t) \Big], $$ {#eq:general_gradient}

其中$\Psi_t$可以是（奖励通常带$\gamma$折扣）：

1. $\sum_{t=0}^{\infty} r_t$：整个轨迹的总奖励。
2. $\sum_{t'=t}^{\infty} r_{t'}$：动作$a_t$之后的奖励（即return $G$）。
3. $\sum_{t'=t}^{\infty} r_{t'} - b(s_t)$：带baseline的版本。
4. $Q^{\pi}(s_t, a_t)$：状态-动作值函数。
5. $A^{\pi}(s_t, a_t)$：优势函数，若可准确计算，理论方差最低。
6. $r_t + V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$：TD残差。

*Baseline*的作用是降低策略更新的方差（后文详述）。

对于语言模型，上述部分概念需适当解释。
如确定性策略下，$V(s) = \max_a Q(s,a)$；随机策略下，$V(s) = \mathbb{E}_{a \sim \pi(a|s)}[Q(s,a)]$。
若将$s+a$视为prompt补全，则$Q(s, a) = V(s+a)$，可得另一种优势技巧：

$$A(s,a) = Q(s,a) - V(s) = V(s + a) - V(s) = r + \gamma V(s + a) - V(s)$$ {#eq:advantage_trick}

即结合了奖励、prompt的价值和整个回复的折扣价值。

### Vanilla Policy Gradient

最基础的策略梯度实现是对$J(\theta)$关于策略参数求导：

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) R_t \right]$$ {#eq:vanilla_policy_gradient}

普通策略梯度算法的常见问题是梯度方差大，可通过多种方式缓解。
常用方法是引入*baseline*对值估计进行归一化。
baseline的本质是按状态对下游动作的价值进行归一化（如Advantage即Q值与V值之差）。
最简单的baseline是奖励的批均值或滑动平均。
即使是这些简单baseline，也能让梯度去偏$\mathbb{E}_{a \sim \pi(a|s)}[\nabla_\theta \log \pi_\theta(a|s)] = 0$，提升学习信号。

许多策略梯度算法都基于优势函数的形式：

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t, a_t) \right]$$ {#eq:advantage_policy_gradient}

策略梯度实现的核心是对概率策略求导：

$$\nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$$ {#eq:log_prob_derivative}

由链式法则推导：

$$\nabla_\theta \log x = \frac{1}{x} \nabla_\theta x$$ {#eq:log_chain_rule}

本章后续会用到这些推导。

### REINFORCE

REINFORCE算法名称可能是后加缩写，但其核心思想对现代RL算法极为重要。
其定义见经典论文*Simple statistical gradient-following algorithms for connectionist reinforcement learning* [@williams1992simple]：

> 名称意为"REward Increment = Nonnegative Factor X Offset Reinforcement X Characteristic Eligibility."

即更新规则有三个部分：

1. 非负因子：学习率（如$\alpha$）。
2. Offset Reinforcement：baseline $b$或其他奖励归一化因子，提升稳定性。
3. Characteristic Eligibility：每个token的归属度，现代公式中常为policy的log概率。

因此，更新形式如下：

$$ \Delta_\theta = \alpha(r - b)e $$ {#eq:REINFORCE_BASIC}

用现代符号和广义return $G$，REINFORCE公式为：

$$
\nabla_{\theta}\,J(\theta)
\;=\;
\mathbb{E}_{\tau \sim \pi_{\theta}}\!\Big[
    \sum_{t=0}^{T}
    \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\,(G_t - b)
\Big],
$$ {#eq:REINFORCE_with_baseline}

其中$G_t - b(s_t)$即当前状态下的*优势*，可进一步写为：

$$
\nabla_{\theta}\,J(\theta)
\;=\;
\mathbb{E}_{\tau \sim \pi_{\theta}}\!\Big[
    \sum_{t=0}^{T}
    \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\,A_t
\Big],
$$ {#eq:REINFORCE_with_advantage}

REINFORCE是vanilla策略梯度的具体实现，使用蒙特卡洛方法估算梯度。

REINFORCE可不使用值网络（value network）；值网络仅用于baseline。
而PPO则需要值网络来准确计算优势函数。

#### REINFORCE Leave One Out（RLOO）

RLOO与标准REINFORCE的核心区别在于：RLOO用*同一batch中其他样本*的平均奖励做baseline，而不是全batch均值 [@huang2024putting], [@ahmadian2024back], [@kool2019buy]。

这要求每个prompt生成多个回复，这在RL微调语言模型中很常见。

具体地，RLOO baseline如下（每个prompt采样$K$条轨迹或动作$a_1, ..., a_K$）：

$$
b(s, a_k) = \frac{1}{K-1}\sum_{i=1, i\neq k}^{K} R(s, a_i),
$$ {#eq:RLOO_baseline}

对应的优势为：

$$
A(s, a_k) = R(s, a_k) - b(s, a_k).
$$ {#eq:RLOO_advantage}

也可写为：

$$
A(s, a_k) = \frac{K}{K - 1}\left(R(s, a_k) - \frac{1}{K}\sum_{i=1}^{K} R(s, a_i)\right).
$$ {#eq:RLOO_advantage_alt}

这是一种简单低方差的优势更新，与GRPO非常相似（GRPO后文介绍），只是KL惩罚和步长裁剪的位置不同。
RLOO优势也可与PPO裁剪结合，说明这些算法本质上非常接近。

RLOO等无需值网络的算法，会将序列的优势（或奖励）分配给序列中每个token。
而PPO等用值网络的算法，则为每个token单独分配值，并对最终奖励进行折扣。
例如，带KL惩罚时，RLOO对整个补全求和，PPO等则对每个token单独计算并从奖励中扣除（GRPO则从优势中扣除）。
这些细节和权衡将在后文详述。

### Proximal Policy Optimization（PPO）

PPO [@schulman2017proximal] 是深度RL成功的基石算法之一（如OpenAI的DOTA 5 [@berner2019dota]等）。
其每个样本的损失函数为：

$$J(\theta) = \min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}A, \text{clip} \left( \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right).$$ {#eq:PPO_EQN}

在语言模型中，损失按token计算，本质上等价于自回归预测的概率乘积。
实际实现中通常用*log概率*，计算更高效。

$$ J(\theta) = \frac{1}{|a|} \sum_{t=0}^{|a|} \min\left(\frac{\pi_\theta(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}A_{t}, \text{clip} \left( \frac{\pi_\theta(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}, 1-\varepsilon, 1+\varepsilon \right) A_{t} \right).  $$  {#eq:PPO_EQN_EXPANDED}

这就是PPO的逐token版本，也适用于其他策略梯度方法，后文实现部分会详细介绍。
其中$\frac{1}{|a|}$为常见实现习惯，正式推导中未必出现（见[@liu2025understanding]）。

PPO的核心在于“策略比率”：

$$R(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$$ {#eq:PPO_POL_RATIO}

策略比率控制参数更新，直观易懂。
每个batch的第一步梯度，策略比率为1（通常每batch做1-4步梯度更新）。
如果策略比率大于1且优势为正，说明新策略更倾向于该动作，反之则相反。

不同情况下，损失函数的表达式如下：

- 优势为正，比例超过$1+\varepsilon$，被裁剪为$(1+\varepsilon)A$。
- 优势为正，比例小于$1-\varepsilon$，则取$R(\theta)A$。
- 优势为负，裁剪发生在$R(\theta)<1-\varepsilon$时，目标变为$(1-\varepsilon)A$。

所有这些设计都是为了让正优势的行为更可能发生，并将梯度步长限制在信任域内。
值得注意的是，PPO在信任域内与标准策略梯度等价。

### Group Relative Policy Optimization（GRPO）

GRPO最早出现在DeepSeekMath [@shao2024deepseekmath]，也被DeepSeek-V3 [@liu2024deepseek]、DeepSeek-R1 [@guo2025deepseek]等采用。
GRPO本质上是PPO的变体，但无需单独学习值函数，而是用另一份原始策略模型（或检查点）初始化。
这样有两个好处：

1. 避免用LM主干学习值函数的难题（最佳实践尚未确定）。
2. 节省显存，无需保留另一套模型权重。

GRPO通过简化值估计，将同一prompt下所有token赋予相同的优势（即每个补全每个token的优势一致），用蒙特卡洛估计法采集多个补全和奖励。

GRPO目标函数与PPO类似。对同一问题$s$的$G$个回复$\{a_1, ..., a_G\}$，损失为：

$$J(\theta) = \frac{1}{G}\sum_{i=1}^G \left(\min\left(\frac{\pi_\theta(a_i|s)}{\pi_{\theta_{old}}(a_i|s)}A_i, \text{clip} \left( \frac{\pi_\theta(a_i|s)}{\pi_{\theta_{old}}(a_i|s)}, 1-\varepsilon, 1+\varepsilon \right) A_i \right) - \beta D_{KL}(\pi_\theta||\pi_{ref})\right).$$ {#eq:GRPO}

可展开为逐token损失：

$$ J(\theta) = \frac{1}{G}\sum_{i=1}^G  \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \left( \min\left(\frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\theta_{old}}(a_{i,t}|s_{i,t})}A_{i,t}, \text{clip} \left( \frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\theta_{old}}(a_{i,t}|s_{i,t})}, 1-\varepsilon, 1+\varepsilon \right) A_{i,t} \right) - \beta D_{KL}(\pi_\theta(\cdot|s_{i,t})||\pi_{ref}(\cdot|s_{i,t})) \right)  $$ {#eq:GRPO_token}

GRPO与PPO的主要区别在于，GRPO的标准实现将KL距离直接加到损失项中。
优势计算为：

$$A_i = \frac{r_i - \text{mean}({r_1, r_2, \cdots, r_G})}{\text{std}({r_1, r_2, \cdots, r_G})}.$$ {#eq:GRPO_ADV}

直观上，GRPO是在同一问题内比较多个答案，模型学会更像正确答案、远离错误答案。
这种优势计算简单直接，适合RLHF大批量采样场景。
与PPO、REINFORCE等基于奖励模型打分的RLHF相比，GRPO常用更多每prompt采样数，优势估计有更丰富上下文。

GRPO的优势估计有一定偏差：标准差归一化会让奖励分布极窄（全对/全错）的问题获得更高优势。
[@liu2025understanding]建议去掉标准差项，但这会降低对“全错但有少数正确样本”的问题的学习信号。

@eq:GRPO_ADV适用于结果监督（如标准奖励模型或可验证奖励），过程监督则需不同实现。
此时，GRPO优势为后续推理步骤归一化奖励之和。

GRPO的优势估计也可不带PPO裁剪，直接用于vanilla策略梯度（如REINFORCE），但这不是标准形式。
实际上，GRPO的Dr. GRPO变体 [@liu2025understanding]的优势估计与RLOO仅差一个缩放因子：

$$ \tilde{A}_i = r_i - \text{mean}({r_1, r_2, \cdots, r_G}) = r_i - \frac{1}{G}\sum{j=1}^G r_j $$ {#eq:DrGRPO_ADV}

而RLOO优势为：

$$ A_i^\text{RLOO} = r_i - \frac{1}{G-1}\sum_{j=1, i\neq j}^G r_j $$ {#eq:RLOO_ADV_AGAIN}

两者相差$\frac{G}{G-1}$的缩放（通常可通过归一化消除实际影响）：

$$
\begin{aligned}
\frac{G}{G-1} \tilde{A}_i &= \frac{G}{G-1} \left( r_i - \frac{1}{G}\sum_{j=1}^G r_j \right) \\
&= r_i - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j \\
&= A_i^{\text{RLOO}}
\end{aligned}
$$ {#eq:RLOO_GRPO_EQUIV}

## 实现细节

与早期深度RL文献相比，面向语言模型等大模型的RLHF实现有许多细节需注意。
以下是一些关键点：

- **值网络初始化**：PPO等算法的值网络可用同结构其他模型或随机权重初始化，对性能影响大。
- **奖励/优势归一化与白化**：奖励模型输出可归一化到0~1，白化（whitening）奖励或优势能进一步提升稳定性。
- **KL估计方式**：复杂语言模型下KL精确计算难，常用近似 [@schulman2016klapprox]。
- **KL控制器**：早期PPO实现有动态KL控制器，现代RLHF多用静态KL惩罚，但也可调整。

更多实现细节可参考 [@huang2024n]，算法基础见 [@weng2018PG]。

### 策略梯度基础实现

一个简单的策略梯度实现如下，利用优势函数估计梯度，为后续PPO/GRPO等算法做准备：

```python
pg_loss = -advantages * ratio
```
ratio为新旧策略概率的比值（log概率差的指数）。

理解该公式时，需考虑不同batch中的情况。
我们希望模型越好，损失越小。

- 情况1：优势为正，说明该动作优于期望值，应强化。此时loss为负，模型会提升该token概率。
- 情况2：优势为负，说明动作劣于期望值。此时loss为正，模型会降低该token概率。
- 情况3：优势为零，不更新，loss为零。

### 损失聚合

实际实现时，关键问题是：如何对KL距离和损失进行求和，设计不同的值归属方式。

*本节大多数讨论假设每个token为一个动作，即RL问题为MDP而非bandit。bandit问题下，所有token用同一损失，这也是A-LoL等算法的默认实现。MDP和bandit的差别，实质在于每样本损失的聚合方式。bandit方法对每token分配相同损失，也与DPO等直接对齐算法一致。*

假设有如下变量，batch大小为B，序列长度为L：

```python
advantages # [B, 1]
per_token_probability_ratios # [B, L]
```

用`pg_loss = -advantages * ratio`即可批量计算损失。这样每个补全的所有token共享同一优势（outcome reward场景），乘以每token的概率比。

若用值网络，loss行为差异更大。
outcome reward时，优势每token相同，token概率差异主导学习动力。

GRPO和PPO等算法中，loss通常对补全token求和：

```python
sequence_loss = ((per_token_loss * completion_mask).sum(dim=1) / \
             completion_mask.sum(dim=1)).mean()
```

这类似于`masked_mean`操作。
也可对每token单独平均：

```python
token_loss = ((per_token_loss * completion_mask).sum() / \
            completion_mask.sum())
```

直观上，按序列平均似乎更好，因为我们关注的是*结果*，而不是具体token。
但这会引入微妙偏差。
举例，两个长度不同的序列，优势分别为`a_1`和`a_2`：

```python
seq_1_advs = [a_1, a_1, a_1, a_1, a_1] # 5 tokens
seq_2_advs = [a_2, a_2, a_2, a_2, a_2, a_2, a_2, a_2, a_2, a_2] # 10 tokens
```

若最后一个token决定优势为正，多步梯度后，loss可能为：

```python
seq_1_losses = [1, 1, 1, 1, 10] # 5 tokens
seq_2_losses = [1, 1, 1, 1, 1, 1, 1, 1, 1, 10] # 10 tokens
```

序列平均后：

```
seq_1_loss = 2.8
seq_2_loss = 1.9
```

若对所有token平均，则为2.27。
若序列差异更大，loss差距会更明显。

更完整的例子见下方代码，演示了两条样本（长短不同）下三种loss聚合方式。

```python
from typing import Optional
import torch

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def masked_sum(
        values: torch.Tensor,
        mask: torch.Tensor,
        axis: Optional[bool] = None,
        constant_normalizer: float = 1.0,
    ) -> torch.Tensor:
    if axis is not None:
        return (values * mask).sum(axis=axis) / constant_normalizer
    else:
        return (values * mask).sum() / constant_normalizer

ratio = torch.tensor([
    [1., 1, 1, 1, 1, 1, 1,],
    [1, 1, 1, 1, 1, 1, 1,],
], requires_grad=True)


advs = torch.tensor([
    [2, 2, 2, 2, 2, 2, 2,],
    [2, 2, 2, 2, 2, 2, 2,],
])

masks = torch.tensor([
    [1, 1, 1, 1, 0, 0, 0,],
    [1, 1, 1, 1, 1, 1, 1,],
])

max_gen_len = 7

masked_mean_result = masked_mean(ratio * advs, masks, axis=1)
masked_mean_token_level = masked_mean(ratio, masks, axis=None)
masked_sum_result = masked_sum(ratio * advs, masks, axis=1, constant_normalizer=max_gen_len)

print("masked_mean", masked_mean_result)
print("masked_sum", masked_sum_result)
print("masked_mean_token_level", masked_mean_token_level)

masked_mean_result.mean().backward()
print("ratio.grad", ratio.grad)
ratio.grad.zero_()

masked_sum_result.mean().backward()
print("ratio.grad", ratio.grad)

masked_mean_token_level.mean().backward()
print("ratio.grad", ratio.grad)
```

可以看到，GRPO默认实现（`masked_mean`）下，短序列每token梯度更大，Dr. GRPO和DAPO则更均衡。
如果用梯度累积，短长序列的平衡还会变化。

另一种聚合方式（见[@liu2025understanding]）是每token损失用最大序列长度归一化，这会影响不同batch间的损失对比。

实际应用中，最优方案取决于具体任务和在线学习设置。
RLHF实践中，数值稳定性和损失方差最小的方案往往更受青睐。

### PPO实现示例

PPO有多种实现，核心*损失*计算如下。
值函数的计算也很关键，且有多种实现方式。

注意，这里的参考策略（或旧log概率）指采样时的参数，不一定是reference policy。reference policy仅用于KL惩罚。

```python
# B: Batch Size, L: Sequence Length, G: Num of Generations
# Apply KL penalty to rewards
rewards = rewards - self.beta * per_token_kl  # Shape: (B*G, L)

# Get value predictions
values = value_net(completions)  # Shape: (B*G, L)

# Compute simple advantages
advantages = rewards - values.detach()  # Shape: (B*G, L)

# Normalize advantages (optional but stable)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
advantages = advantages.unsqueeze(1)  # Shape: (B*G, 1)

# Compute probability ratio between new and old policies
ratio = torch.exp(new_per_token_logps - per_token_logps)  # Shape: (B*G, L)

# PPO clipping objective
eps = self.cliprange  # e.g. 0.2
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)

# Simple value function loss
vf_loss = 0.5 * ((rewards - values) ** 2)  # Shape: (B*G, L)

# Combine policy and value losses
per_token_loss = pg_loss_max + self.vf_coef * vf_loss  # Shape: (B*G, L)

# Apply completion mask and compute final loss
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # Scalar

# Compute metrics for logging
with torch.no_grad():
    # Compute clipping fraction
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()
    
    # Compute approximate KL
    approx_kl = 0.5 * ((new_per_token_logps - per_token_logps)**2).mean()
    
    # Compute value loss for logging
    value_loss = vf_loss.mean()
```

PPO的核心在于如何更新策略梯度损失。
关注这三行：

```python
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)
```
`pg_losses1`即vanilla优势策略梯度，PPO在此基础上引入裁剪，使更新步长不至于过大。

如果*不*约束损失，策略梯度会直接将权重更新到新分布。
通过clamp logratio，PPO限制了参数更新幅度。

最后取二者最大值，确保更保守的损失更新。

PPO在学习值函数的同时进行上述操作，虽然更复杂，但这是参数更新的核心逻辑。

#### PPO/GRPO单步梯度（无裁剪）简化

若PPO（或GRPO）每样本只做一次梯度更新，主方程可大幅简化。
此时$\pi_\theta = \pi_{\theta_{old}}$，更新规则变为（$[]_\nabla$表示停止梯度）：

$$J(\theta) = \frac{1}{G}\sum_{i=1}^G \left(\frac{\pi_\theta(a_i|s)}{\left[\pi_{\theta}(a_i|s)\right]_\nabla}A_i - \beta D_{KL}(\pi_\theta||\pi_{ref})\right). $$ {#eq:ppo_1step}

此时可省略第二次策略梯度和裁剪逻辑，优化器更接近标准策略梯度。

### GRPO实现示例

DeepSeekMath论文详细介绍了GRPO与PPO的实现差异 [@shao2024deepseekmath]。
如，GRPO将KL惩罚直接加到损失项，而PPO多在奖励中加KL。
通常KL距离对每个token单独计算，推理训练时每个prompt采样多个补全，每batch有多个prompt，KL距离的shape为[B, L, N]。

综合起来，伪代码如下：

```python
# B: Batch Size, L: Sequence Length, G: Number of Generations
# Compute grouped-wise rewards # Shape: (B,)
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)    

# Normalize the rewards to compute the advantages
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
# Shape: (B*G,)

# Compute advantages
advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
advantages = advantages.unsqueeze(1)
# Shape: (B*G, 1)

# Compute probability ratio between new and old policies
ratio = torch.exp(new_per_token_logps - per_token_logps)  # Shape: (B*G, L)

# PPO clipping objective
eps = self.cliprange  # e.g. 0.2
pg_losses1 = -advantages * ratio  # Shape: (B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # Shape: (B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # Shape: (B*G, L)

# important to GRPO -- PPO applies this in reward traditionally
# Combine with KL penalty
per_token_loss = pg_loss_max + self.beta * per_token_kl  # Shape: (B*G, L)

# Apply completion mask and compute final loss
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # Scalar

# Compute core metric for logging (KL, reward, etc. also logged)
with torch.no_grad():
    # Compute clipping fraction
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()
    
    # Compute approximate KL
    approx_kl = 0.5 * ((new_per_token_logps - per_token_logps)**2).mean()
```

更多解释请见上文PPO部分。

#### RLOO vs. GRPO

RLOO的优势更新与GRPO极为接近，突出两者在本质上的相似性。
RLOO的优势是相对于同一问题下其他补全的奖励。
简明代码（参考[TRL实现](https://github.com/huggingface/trl/blob/bfe20756082488350091352d1cdc19c172e42cd8/trl/trainer/rloo_trainer.py#L433)）：

```python
# rloo_k --> 每prompt补全数 
# rlhf_reward --> 所有补全的奖励，长度B = N x k
rlhf_reward = rlhf_reward.reshape(rloo_k, -1) # 
# 形状: (k, N)，每列j为prompt j的k个奖励

baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
# baseline --> leave-one-out基线奖励，形状: (k, N)
# baseline[i, j]是prompt j除i外样本的平均奖励

advantages = rlhf_reward - baseline
# advantages --> 同形状

advantages = advantages.flatten() # 恢复原始tensor形状
```

其余实现细节与其他策略梯度方法类似。

## 补充话题

要精通策略梯度算法在RLHF中的应用，还有许多细节值得关注。

### 广义优势估计（GAE）

广义优势估计（Generalized Advantage Estimation, GAE）是一种平衡偏差-方差权衡的优势计算方法[@schulman2015high]。单步优势估计具有较大方差，而完整轨迹估计则存在较大偏差。GAE通过结合多步预测和指数加权滑动平均来优化这一权衡。

$n$步优势估计的数学表达式如下（类似于时序差分残差）：

$$
\hat{A}_t^{(n)} = \begin{cases}
r_t + \gamma V(s_{t+1}) - V(s_t), & n = 1 \\
r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t), & n = 2 \\
\vdots \\
\sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n}) - V(s_t), & \text{一般情况}
\end{cases}
$$

其中：
- $\gamma$ 为折扣因子
- $V(s)$ 是状态价值函数
- 最终表达式通过引入衰减因子$\lambda$实现滑动平均：

$$
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
$$

该方法的优势在于能灵活调节偏差与方差的平衡（通过$\lambda$参数），已成为现代深度强化学习的标准组件。具体实现可参考TRL库的[GAE计算模块](https://github.com/huggingface/trl)。