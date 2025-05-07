---
prev-chapter: "策略梯度"
prev-url: "11-policy-gradients.html"
page-title: 直接对齐算法
next-chapter: "宪法AI"
next-url: "13-cai.html"
---

# 直接对齐算法（Direct Alignment Algorithms, DAAs）

直接对齐算法（Direct Alignment Algorithms, DAAs）允许我们在无需训练奖励模型或使用强化学习优化器的情况下，直接优化RLHF目标。
其中最具代表性、并掀起学术界大规模关注的，是直接偏好优化（Direct Preference Optimization, DPO）[@rafailov2024direct]。
DPO的本质是用梯度上升直接求解带约束的RLHF目标。
自2023年5月发布以来，经过社区对数据和超参数（尤其是意外地低学习率）的探索，DPO及其变体被广泛应用于主流模型，如Zephyr-$\beta$（2023年10月）[@tunstall2023zephyr]、Llama 3 Instruct [@dubey2024llama]、Tülu 2 [@ivison2023camels]、Tülu 3 [@lambert2024t]、Nemotron 4 340B [@adler2024nemotron]等。
严格来说，Sequence Likelihood Calibration（SLiC-HF）[@zhao2023slic]更早提出，但因有效性和运气等原因未被广泛采用。

DPO和DAAs最重要的意义，是极大降低了语言模型后训练的技术门槛。

## 直接偏好优化（Direct Preference Optimization, DPO）

下面我们将直观解释DPO的原理，并推导其核心公式。

### DPO的原理

DPO表面上是直接优化策略以求解RLHF目标。
其损失函数本质上是log概率的成对关系。
Bradley-Terry奖励模型推导出的损失函数如下：

$$ \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}}\left[ \log \sigma\left( \beta \log \frac{\pi_{\theta}(y_c \mid x)}{\pi_{\text{ref}}(y_c \mid x)} - \beta \log \frac{\pi_{\theta}(y_r \mid x)}{\pi_{\text{ref}}(y_r \mid x)} \right) \right] $$ {#eq:dpo_core}

这里用到DPO的隐式奖励：

$$r(x, y) = \beta  \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$ {#eq:dpo_reward}

这个奖励来自Bradley-Terry模型下的最优策略推导（见@eq:dpo_opt_policy）。
本质上，DPO的隐式奖励模型用“最优策略下人类偏好数据的概率”替代了外部奖励模型。

观察@eq:dpo_core中的损失，优化目标是让选中回复的log比率大于被拒回复（归一化参考模型）。
实际中，这就是对模型在数据中token序列的log概率求和。
因此，DPO实质上是在拉大选中与被拒回复概率的差距。

有了@eq:dpo_reward中的奖励，我们可以写出损失的梯度，进一步理解机制：

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\beta \mathbb{E}_{(x, y_c, y_r)\sim \mathcal{D}}\left[ \sigma\left(r_{\theta}(x, y_r) - r_{\theta}(x, y_c)\right) \left(\nabla_{\theta}\log \pi(y_c \mid x) - \nabla_{\theta}\log \pi(y_r \mid x)\right) \right] $$ {#eq:dpo_gradient}

直观理解如下：

- $\sigma(\cdot)$中的第一项，为参数更新赋予0到1的权重，当奖励估计错误时（被拒样本更优），权重更大。
- 内部括号$[\cdot]$项提升选中回复$y_c$的概率，降低被拒回复$y_r$的概率。
- $\beta$控制优化中排序与KL距离的平衡。

核心直觉是，DPO“隐式拟合了一个奖励模型，其对应最优策略可解析写出”（归功于梯度上升和ML工具）。
很多人误以为DPO直接训练策略，其实本质上还是在学习一个奖励模型，这也是论文副标题“Your Language Model is Secretly a Reward Model”的由来。

通过隐式奖励建模，DPO针对数据和KL约束$\beta$，给出RLHF目标的最优解。
与策略梯度类RL方法的区别在于，DPO的生成不是在线的，而是离线的，因此$\beta$更易调节，但最优值依赖于具体模型与数据。

每批偏好数据（即大量$y_{chosen} \succ y_{rejected}$对）下，DPO直接向最优解做梯度步，比策略梯度方法要简单得多。

![DPO简洁性梗图，致谢Tom Goldstein。](images/dpo_meme.jpeg){#fig:dpo-meme}

### DPO公式推导

DPO推导分两步：  
1. 推导RLHF目标的最优策略形式；  
2. 用Bradley-Terry模型推导如何从偏好数据获得该解。

#### 1. RLHF最优解推导

RLHF优化目标：

$$ \max_{\pi} \mathbb{E}_{\tau \sim \pi} \left[r_\theta(s_t, a_t)\right] - \beta  \mathcal{D}_{KL}(\pi^{\text{RL}}(\cdot|s_t) \| \pi^{\text{ref}}(\cdot|s_t)).$$ {#eq:rlhf_opt_eq_repeat}

展开KL散度：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[r(x,y)-\beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right] $$ {#eq:dpo_deriv_1}

拆分括号，变成两个期望：

$$ = \max_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}[r(x,y)] - \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_2}

提取$-1$和$\beta$：

$$ = \min_{\pi}\left(-\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}[r(x,y)] + \beta\,\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}\right]\right) $$ {#eq:dpo_deriv_3}

除以$\beta$并合并：

$$ = \min_{\pi}\left(\mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)}\left[ \log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r(x,y) \right]\right) $$ {#eq:dpo_deriv_4}

引入分区函数$Z(x)$：

$$ Z(x) = \sum_y \pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_partition}

$Z(x)$是对参考策略归一化的分区函数，对prompt $x$的所有回复$y$求和。
代入后，优化目标变为：

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\mathbb{E}_{y\sim\pi(y|x)}\left[\log\frac{\pi(y|x)}{\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)} - \log Z(x)\right] $$ {#eq:dpo_deriv_5}

本质上，这等价于KL距离最小化：

$$ \min_{\pi}\mathbb{E}_{x\sim\mathcal{D}}\left[\mathbb{D}_\text{KL} \left(\pi(y|x)||\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) \right) - \log Z(x)\right] $$ {#eq:dpo_deriv_10}

Gibbs不等式告诉我们，最优解$\pi^*$满足两者相等：

$$ \pi^*(y|x) = \pi(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right) $$ {#eq:dpo_opt_policy}

#### 2. Bradley-Terry模型下的DPO目标

回顾第7章奖励建模与第6章偏好数据，Bradley-Terry模型为：

$$p^*(y_1 \succ y_2 \mid x) = \frac{\exp\left(r^*(x,y_1)\right)}{\exp\left(r^*(x,y_1)\right) + \exp\left(r^*(x, y_2)\right)} $$ {#eq:bradley_terry_dpo}

对@eq:dpo_opt_policy取对数并代入，得DPO奖励：

$$r^*(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$ {#eq:dpo_reward_full}

代入Bradley-Terry公式，化简后得：

$$p^*(y_1 \succ y_2 \mid x) = \sigma\left(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} - \beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}\right) $$ {#eq:dpo_loss_deriv3}

这正是DPO的损失函数（见@eq:dpo_core）。

#### 3. Bradley-Terry DPO梯度推导

DPO梯度如@eq:dpo_gradient所示，推导如下：

$$\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\nabla_{\theta}\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[ \log \sigma\left(\beta \log \frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)} - \beta \log \frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right)\right] $$ {#eq:dpo_grad_0}

利用sigmoid和log的求导性质，可化为：

$$ -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\left[\beta\sigma\left(\beta\log\frac{\pi_{\theta}(y_r|x)}{\pi_{\text{ref}}(y_r|x)} - \beta\log\frac{\pi_{\theta}(y_c|x)}{\pi_{\text{ref}}(y_c|x)}\right)\left[\nabla_{\theta}\log\pi(y_c|x)-\nabla_{\theta}\log\pi(y_r|x)\right]\right] $$ {#eq:dpo_grad_3}

## 数值问题、局限与变体

DPO算法已出现多种变体，旨在解决其局限。
例如，DPO在无奖励模型评分的情况下，对每对偏好数据赋予同等权重，忽略了更丰富的标签信息。
为此，相关算法尝试重新平衡优化目标：

- **REBEL**：将奖励模型分数作为选中与被拒回复之间的margin，提升RLHF问题的求解准确性 [@gao2024rebel]。
- **保守DPO（cDPO）与身份偏好优化（IPO）**：假设偏好数据存在噪声，cDPO假定N%数据标注错误 [@rafailov2024direct]，IPO则将偏好概率改为非线性函数，弱化直接标签优化 [@azar2024general]。
- **带偏移的DPO（ODPO）**：要求选中与被拒回复的likelihood差异大于某个阈值，不再一视同仁 [@amini2024direct]。

有些变体通过调整损失函数或内存优化提升学习信号或效率：

- **ORPO（Odds Ratio Policy Optimization）**：直接拉高选中回复概率，并对其加小惩罚，无需参考模型，简化流程 [@hong2024reference]。
- **SimPO（Simple Preference Optimization）**：将DPO中的log概率取平均而非求和，或加长度归一化，提升性能 [@meng2025simpo]。

![DPO中的偏好“位移”问题示意。](images/dpo_displacement.png){#fig:dpo_issue .center}

DPO的一个突出问题是：优化目标仅仅是拉大选中与被拒回复概率的间隔。
数值上，模型会降低两者的概率，但被拒回复降得更多（见@fig:dpo_issue）。
这对泛化的影响尚不明确，有研究认为这会提升未被覆盖行为的概率 [@razin2024unintentional] [@ren2024learning]。
如Cal-DPO [@xiao2024cal]、AlphaPO [@gupta2025alphapo]等方法通过调整优化过程或奖励形状缓解这种**偏好位移**。
实际影响尚不明朗，但这可能是在线RL方法优于DPO的原因之一。

另一个DPO类方法性能上限低于在线RLHF的主要原因，是其训练信号来自其他模型的补全。
在线变体如**Online DPO** [@guo2024direct]，或结合奖励模型重标记的新生成数据的**Discriminator-Guided DPO（D2PO）** [@singhal2024d2po]，通过实时生成新补全并引入偏好信号，缓解了这一问题。

还有许多DAA变体，如Direct Nash Optimization（DNO）[@rosset2024direct]、Binary Classifier Optimization（BCO）[@jung2024binary]等，但目前算法选择远不如初始模型和数据重要 [@lambert2024t] [@zhao2024rainbowpo] [@gorbatovski2025differences]。

## 实现注意事项

DAA如DPO的实现方式与策略梯度优化器有很大不同。
DPO损失函数的典型实现如下 [@rafailov2024direct]：

```python
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps

logits = pi_logratios - ref_logratios  # 即 h_{\pi_\theta}^{y_w,y_l}

losses = -F.logsigmoid(beta * logits)

chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
```

这可以直接用于标准语言模型训练流程（只需加一个参考模型）。

这种方式更简单，也提升了开发体验，但有一些新的注意点：

1. **KL距离为静态**：DPO等算法中，KL距离由$\beta$参数直接设定，作为距离惩罚。这是因为DPO每步梯度都朝着RLHF目标的*最优*解迈进，$\beta$决定了目标解的具体位置。RL方法则每步根据batch和最新数据调整。
2. **缓存log概率**：简单实现中，policy和reference模型同时前向推理，方便损失计算，但会使显存消耗翻倍。可先离线计算参考模型log概率，训练时直接查表，显著降低显存需求。

## DAA与RL：在线与离线数据

本质问题是：我们是否需要强化学习的内部机制（如值函数、策略梯度等）来实现RLHF对齐？
当然，二者各有成熟体系，关键在于理解两者的本质差异和性能表现。

多项研究发现，基于策略梯度和RL的方法在性能上优于DPO及其变体。
这些研究通过控制数据、算法对比训练模型 [@ivison2024unpacking] [@xu2024dpo]，或研究RL优化循环中on-policy数据的作用 [@tajwar2024preference]，都显示DPO略逊一筹。

尽管如此，DAA因其简单性在主流模型中广泛应用。
DAA为训练数据和配置的快速迭代提供了极大便利，而数据往往比算法本身更重要，因此DPO在实际中依然很有价值。

随着以RL为主的推理模型兴起，未来会有更多投资回归RL偏好微调，这将提升RL基础设施的健壮性，并进一步拉开DAA与RLHF在线优化的性能差距。