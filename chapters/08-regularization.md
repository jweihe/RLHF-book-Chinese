---
prev-chapter: "奖励建模"
prev-url: "07-reward-models.html"
page-title: 正则化
next-chapter: "指令微调"
next-url: "09-instruction-tuning.html"
---

# 正则化

在RLHF优化过程中，通常需要引入多种正则化手段，以防止奖励模型出现过度优化（over-optimization）的现象。
在实际中，过度优化常表现为模型输出无意义的文本，例如：推理过程看似合理但答案极其错误、文本重复、频繁切换语言、或出现大量特殊字符等。

目前最常见的正则化方式，是对生成样本的当前策略与参考策略之间施加KL距离惩罚，这一方法被广泛应用于主流RLHF实现中。
除此之外，文献中还出现过许多其他正则化技巧，但往往在新一代模型中被简化或淘汰。
也就是说，除了核心的生成KL距离，其他正则化方法多用于稳定实验设置，随着模型迭代会逐步简化。
尽管如此，理解如何在RLHF中约束优化过程仍然非常重要。

在RLHF框架下，结合奖励模型$r_\theta$，常见的正则化表达如下：

$$ r = r_\theta - \lambda r_{\text{reg.}} $$ {#eq:rl_start}

最常见的实现是：

$$
r = r_\theta - \lambda_{\text{KL}} \mathcal{D}_{\text{KL}} \left( \pi^{\text{RL}}(y \mid x) \, \| \, \pi^{\text{Ref.}}(y \mid x) \right)
$$ {#eq:kl_standard}

## RL优化中的KL距离

关于数学定义，见第5章“问题设定与背景”。
回顾KL距离的定义：

$$ D_{KL}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) $$ {#eq:kl_distance_regularization}

在RLHF中，常用的两个分布分别是新模型版本的分布$P(x)$和参考策略的分布$Q(x)$。

### 参考模型与生成样本

KL惩罚最常见的实现方式，是将训练过程中的生成token与静态参考模型的输出进行比较。
直观上，这意味着你希望训练出来的模型风格尽量接近参考模型。
参考模型通常是指令微调后的模型，也可以是之前某个RL检查点。
用上面的公式，采样模型为$P^{\text{RL}}(x)$，参考模型为$P^{\text{Ref.}}(x)$，如@eq:kl_standard所示。
早在大语言模型流行之前，KL距离就被用于对话智能体 [@jaques2017sequence]，之后KL正则很快成为预训练模型微调的核心技术 [@jaques2020human]。

### 实现示例

在实际工程中，KL距离通常用近似计算 [@schulman2016klapprox]，实现起来非常简单。
根据上述定义，KL求和可转化为对分布$P(X)$的采样期望。
此时，$P(X)$即为当前训练模型的生成分布（而非参考模型）。
KL距离的计算变为：

$$
D_{\text{KL}}(P \,||\, Q) = \mathbb{E}_{x \sim P} \left[ \log P(x) - \log Q(x) \right].
$$ {#eq:kl_expectation}

这种方式尤其适合直接用语言模型训练中常用的log概率实现。

```python
# 步骤1：从策略采样（或生成）序列
generated_tokens = model.generate(inputs)

# 步骤2：在两个模型下对生成序列评分
#    对于自回归语言模型，通常这样处理：
#      inputs_for_scoring = generated_tokens[:, :-1]
#      labels           = generated_tokens[:, 1:]
logits       = model.forward(generated_tokens[:, :-1]).logits
ref_logits   = ref_model.forward(generated_tokens[:, :-1]).logits

# 转换为log概率，并对齐标签以索引logits
logprobs     = F.log_softmax(logits, dim=-1)
ref_logprobs = F.log_softmax(ref_logits, dim=-1)

# 获取实际下一个token的log概率
token_logprobs     = logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
ref_token_logprobs = ref_logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

# 累加（或平均）得到序列log概率，再计算KL：
seq_logprob     = token_logprobs.sum(dim=-1)
ref_seq_logprob = ref_token_logprobs.sum(dim=-1)

kl_approx = seq_logprob - ref_seq_logprob
kl_full   = F.kl_div(ref_logprobs, logprobs, reduction='batchmean')
```

典型实现可参考 [TRL](https://github.com/huggingface/trl/blob/5c21de30ae210e4251ead85517ba8dfe3f210e81/trl/trainer/ppo_trainer.py#L1150) 和 [Hamish Ivison的Jax代码](https://github.com/hamishivi/EasyLM/blob/main/EasyLM/models/llama/llama_train_ppo.py#L278)。

## 预训练梯度

另一种正则化视角，是希望模型保持对某个*数据集*的拟合能力，这在InstructGPT中被用来“修复在公开NLP数据集上的性能回退” [@ouyang2022training]。
为此，可以对RLHF的训练目标进行如下调整：
基于@eq:rl_start，采样RL策略模型的补全$y$和prompt $x$，优化目标为：

$$
\text{objective} (\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi^{\text{RL}}_{\theta}}} \left[ r_{\theta}(x, y) - \lambda r_{\text{reg.}} \right]
$$ {#eq:objective_regularization}

然后，可以为预训练准确率增加额外奖励项：

$$
\text{objective} (\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi^{\text{RL}}_{\theta}}} \left[ r_{\theta}(x, y) - \lambda r_{\text{reg.}} \right] + \gamma \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}} \left[ \log(\pi^{\text{RL}}_{\theta}(x)) \right]
$$ {#eq:objective_pretraining}

近期研究提出用负对数似然项（NLL）平衡直接偏好优化（DPO）的优化过程 [@pang2024iterative]。
鉴于DPO损失的成对特性，类似的损失修正也可用于奖励模型训练，约束模型输出更准确的文本（有实验室未公开的相关传闻）。

优化目标可写作DPO的修正规则：

$$\mathcal{L}_{\text{DPO+NLL}} = \mathcal{L}_{\text{DPO}}(c_i^w, y_i^w, c_i^l, y_i^l \mid x_i) + \alpha \mathcal{L}_{\text{NLL}}(c_i^w, y_i^w \mid x_i)
$$ {#eq:dpo_nll}

$$
= -\log \sigma \left( \beta \log \frac{M_\theta(c_i^w, y_i^w \mid x_i)}{M_t(c_i^w, y_i^w \mid x_i)} - \beta \log \frac{M_\theta(c_i^l, y_i^l \mid x_i)}{M_t(c_i^l, y_i^l \mid x_i)} \right) - \alpha \frac{\log M_\theta(c_i^w, y_i^w \mid x_i)}{|c_i^w| + |y_i^w|}.
$$ {#eq:dpo_nll_expanded}

## 其他正则化方法

在RLHF体系的其他部分，对优化的控制方式则不那么明确。
大多数奖励模型除了标准对比损失外，并无额外正则化。
直接对齐算法则通过$\beta$参数以不同方式控制KL距离（详见“直接对齐算法”章节）。

Llama 2提出了奖励模型训练的margin loss [@touvron2023llama]：

$$
\mathcal{L}(\theta) = - \left[ \log \left( \sigma \left( r_{\theta}(x, y_w) - r_{\theta}(x, y_l) - m(r) \right) \right) \right]
$$ {#eq:margin_loss}

其中$m(r)$为两位标注员打分的数值差异。
这可以通过让标注员用数值量表（如Likert量表）对输出评分，或用定量排序方法实现。

奖励margin在直接对齐相关文献中被大量使用，如Reward weighted DPO、“Reward-aware Preference Optimization”（RPO，奖励感知偏好优化），即在DPO损失基础上将奖励模型分数纳入更新规则 [@adler2024nemotron]，或REBEL [@gao2024rebel]，采用奖励差异加权的回归损失等。