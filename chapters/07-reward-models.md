---
prev-chapter: "偏好数据"
prev-url: "06-preference-data.html"
page-title: 奖励建模
next-chapter: "正则化"
next-url: "08-regularization.html"
---

# 奖励建模

奖励模型（Reward Model）是现代RLHF方法的核心组成部分。
奖励模型在强化学习领域被广泛用作环境奖励的代理 [@sutton2018reinforcement]。
这与逆向强化学习（Inverse Reinforcement Learning）密切相关，即通过智能体的行为轨迹来近似其奖励函数 [@ng2000algorithms]，以及其他深度强化学习方向。
奖励模型的现代形式，最早被提出用于研究价值对齐（value alignment）问题 [@leike2018scalable]。

常见的偏好奖励模型为 prompt 与回答输出一个标量分数。Bradley-Terry 模型将两个回答的分数差转换为偏好概率；单个分数既不是文本相似度，也不是概率。
本节后面还会介绍Outcome Reward Model（ORM，结果奖励模型，预测补全是否正确）和Process Reward Model（PRM，过程奖励模型，为推理过程中的每一步打分）。
如无特殊说明，本文提到的奖励模型均指预测文本偏好的模型。

## 奖励模型的训练

训练RLHF标准奖励模型有两种主流表达方式——它们在数值上是等价的。
规范做法源自Bradley-Terry偏好模型 [@BradleyTerry]。
Bradley-Terry模型用于衡量同一分布下两个事件（如$i$和$j$）的成对比较满足$i > j$的概率：

$$P(i > j) = \frac{p_i}{p_i + p_j}$$ {#eq:bradterry}

要训练奖励模型，需要设计一个损失函数，使模型输出满足上述关系。
首先，将语言模型转为输出标量值的模型，该标量可以是未归一化的 logit，而不是概率。
对于同一prompt下的两个补全$y_1$和$y_2$，用奖励模型$r_\theta$分别打分。

奖励模型在成对比较下的“成功概率”可写为：

$$P(y_1 > y_2) = \frac{\exp(r(y_1))}{\exp(r(y_1)) + \exp(r(y_2))}$$ {#eq:bradterryrm}

最大化上述函数的对数似然（或等价地，最小化负对数似然），即可得到奖励模型的训练损失：

$$
\begin{aligned}
\theta^* = \arg\max_\theta P(y_w > y_l) &= \arg\max_\theta \frac{\exp(r_\theta(y_w))}{\exp(r_\theta(y_w)) + \exp(r_\theta(y_l))} \\
&= \arg\max_\theta \frac{1}{1 + \exp(-(r_\theta(y_w) - r_\theta(y_l)))} \\
&= \arg\max_\theta \sigma \left( r_\theta(y_w) - r_\theta(y_l) \right) \\
&= \arg\min_\theta - \log \left( \sigma \left(r_\theta(y_w) - r_\theta(y_l)\right) \right)
\end{aligned}
$$ {#eq:bradterryrm_deriv}

常见的两种写法如下：

- [@ouyang2022training]等采用：
  $$
  \mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(x, y_w) - r_{\theta}(x, y_l) \right) \right)
  $$ {#eq:rewardmodeling1}

- [@askell2021general]等采用：
  $$
  \mathcal{L}(\theta) = \log \left( 1 + e^{r_{\theta}(x, y_l) - r_{\theta}(x, y_w)} \right)
  $$ {#eq:rewardmodeling2}

## 模型结构

一种常见实现是使用 `AutoModelForSequenceClassification` 并设置 `num_labels=1`：在主干模型的序列表示上添加线性评分头，每个回答独立输出一个标量。具体使用哪个 token 的表示取决于模型的池化实现。

推理时先获得分数 $r(x,y)$；对同一 prompt 的两个回答，再用 $\sigma(r(x,y_1)-r(x,y_2))$ 估计第一个回答更受偏好的概率。不要把单个 logit 当作“被选中概率”。

## 实现示例

奖励模型损失的实现非常简单。
更多的工程挑战在于设置独立的数据加载器和推理流程。
只要数据加载器正确，损失函数可这样写：

```python
import torch.nn as nn
rewards_chosen = model(**inputs_chosen).logits.squeeze(-1)
rewards_rejected = model(**inputs_rejected).logits.squeeze(-1)

loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

一些早期工作只训练 1 个 epoch 以减轻过拟合；训练轮数仍应根据数据规模和验证集表现决定。

## 变体

奖励建模仍是RLHF领域相对探索较少的部分。
许多流行工作对传统奖励模型损失做了修改，但尚未形成统一最佳实践。

### 偏好间隔损失（Preference Margin Loss）

如果标注员给出的是Likert量表上的分数或排序，可以利用这些关系的幅度信息进行训练。
常见做法是将数据二值化（隐含1/0），但利用更多信息有助于提升模型训练效果。
Llama 2提出用两个数据点间的间隔$m(r)$来区分偏好强度：

$$\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(x, y_w) - r_{\theta}(x, y_l) - m(r) \right) \right)$$ {#eq:rewardmodelingmargin}

需要注意的是，Llama 3中去除了margin项，因为团队发现随着规模扩大，收益递减。

### 单prompt多比较的平衡

InstructGPT研究了每个prompt用不同数量补全进行训练的影响，并在奖励模型训练时做了平衡 [@ouyang2022training]。
做法是按prompt对每个比较加权更新损失。
同一 prompt 的比较应共同参与损失计算；当各 prompt 的回答数不同，还需显式做 prompt 内归一化，不能仅靠放进同一 batch 来保证权重平衡。以下假设每组 $K$ 个回答已按偏好从高到低排列：
损失函数如下：

$$\mathcal{L}(\theta) = -\mathbb{E}_{(x,y_{1:K})\sim D}\left[\frac{1}{\binom{K}{2}}\sum_{i<j}\log\sigma\left(r_\theta(x,y_i)-r_\theta(x,y_j)\right)\right]$$ {#eq:rewardmodelinginstructgpt}

### K-wise损失函数

还有很多其他形式可以用于RLHF中的人类偏好建模。
比如Starling 7B和34B等早期RLHF模型 [@zhu2024starling]，采用了基于Plackett-Luce模型的K-wise损失函数 [@liu2019learning]。

Zhu等（2023）[@zhu2023principled]形式化如下：
对于某个prompt或状态$s^i$，采样$K$个动作$(a_0^i, a_1^i, \cdots, a_{K-1}^i)$，然后由标注员给出排序$\sigma^i: [K] \mapsto [K]$，其中$\sigma^i(0)$为最优动作。
概率建模如下：

$$P(\sigma^i|s^i,a_0^i,a_1^i,\ldots,a_{K-1}^i) = \prod_{k=0}^{K-1} \frac{\exp(r_{\theta\star}(s^i,a_{\sigma^i(k)}^i))}{\sum_{j=k}^{K-1}\exp(r_{\theta\star}(s^i,a_{\sigma^i(j)}^i))}$$ {#eq:kwise_rm}

当$K=2$时，退化为成对Bradley-Terry模型。
训练完成后，这类模型在RLHF训练中用法与其他奖励模型类似。

## 结果奖励模型（Outcome Reward Models, ORM）

大多数语言模型及AI系统的*偏好微调*都采用上述Bradley-Terry模型。
对于推理密集型任务，则可以采用Outcome Reward Model（ORM，结果奖励模型）。
ORM 使用回答是否正确的结果标签，样本可表示为 $(x,y,z)$，其中 $z\in\{0,1\}$。这不要求每个正确回答必须与一个错误回答配对。

模型结构与标准奖励模型类似，都是在主模型顶部加一个线性层输出logit（RM），但ORM的训练目标略有不同 [@cobbe2021training]：

> [我们]用联合目标训练验证器，让模型不仅学习原有的语言建模目标，还要学会判断补全是否正确。  
> 架构上，验证器是语言模型，在最后的unembedding层加一个小的标量head，对每个token输出预测。  
> 这个head仅用一个偏置参数和一个增益参数作用于语言模型输出的logit。

该工作将结果标签用于 token 位置上的辅助预测；这是一种实现方式。ORM 的定义取决于结果级监督，而不要求逐 token 输出，也不要求输出一个分类 token。
形式化地，参考[@lyu2025exploring]：

$$\mathcal{L}_{\text{CE}} = -\mathbb{E}_{(s,r)\sim \mathcal{D}}[r\log p_\theta(s) + (1-r)\log(1-p_\theta(s))]$$ {#eq:orm_loss}

其中$r \in \{0,1\}$为二元标签，1代表正确答案，0代表错误，$p_\theta(s)$为模型预测正确概率。

这类模型仍在使用，但在开源RLHF工具中支持较少。
比如*Let's Verify Step by Step* [@lightman2023let]用的也是类似ORM，但没有用语言建模损失。
最终损失是每个token的交叉熵，判断最终答案是否正确。

由于支持有限，Outcome Reward Model（ORM）一词在不同文献中定义略有差异。
有些文献（如[@lyu2025exploring]）沿用Cobbe等2021年定义，其他文献则不同。

## 过程奖励模型（Process Reward Models, PRM）

过程奖励模型（Process Reward Models, PRMs），最初称为过程监督奖励模型（Process-supervised Reward Models），用于在推理链条的每一步输出分数。
PRM 的监督目标是各个推理步骤，ORM 的监督目标是整个回答。步骤标签常放在步骤末尾的分隔 token 上，其余位置忽略损失；输出位置与评分头的具体结构并不是这些模型类别的定义。

以下为HuggingFace TRL [@vonwerra2022trl]中每步标签的打包示例：
```
# 获取分隔符token的ID并加入补全序列
separator_ids = tokenizer.encode(step_separator, add_special_tokens=False)
completions_ids = [completion + separator_ids for completion in completions_ids]

# 创建标签
labels = [[-100] * (len(completion) - 1) + [label] for completion, label in zip(completions_ids, labels)]
```

传统上，PRM用语言建模head在每个推理步骤结束时输出token（如遇到双换行或特殊token）。
标签可以是错误、中性、正确三类，也可以是二元标签；分类器输出的是相应类别的分数或概率。
这些标签不一定表示模型是否“走在正确道路上”，而是该步是否正确。

## 奖励模型 vs. 结果RM vs. 过程RM vs. 价值函数

上述各种奖励模型展示了RLHF及后训练中衡量“质量”的多种方式。
下表总结了各类模型的预测内容、训练方式及结构。

| 模型类别 | 预测内容 | 训练方式 | 语言模型结构 |
|----------|----------|----------|--------------|
| **偏好奖励模型** | 回答的相对偏好分数 | 成对或多元比较 | 序列表示与标量评分头 |
| **结果奖励模型** | 回答正确性 | 结果级标签 | 分类头或生成式评分等 |
| **过程奖励模型** | 各步骤的正确性或质量 | 步骤级标签 | 步骤位置上的评分或分类头等 |
| **价值函数** | 当前策略下的期望回报 | 回报目标或自举目标 | 状态表示与标量回归头 |

Table: 奖励模型类型对比。{#tbl:rm_compare}

补充说明：

- 在偏好微调和推理训练中，价值函数常用折扣因子1，这使其与ORM更为接近，但训练损失不同。
- 过程奖励模型可通过从中间状态rollout、收集结果数据来监督。这融合了多种思想，但若损失按推理步标签，则更适合称为PRM。

## 生成式奖励建模

由于偏好数据昂贵，研究者开始探索用现有大语言模型（LLM）充当“裁判”以评判人类偏好或用于评测 [@zheng2023judging]。
核心思想是：用prompt让LLM作为公正裁判，给出评判指令、问题和两个补全（类似人工标注流程）。
MT-Bench [@zheng2023judging] 的评测prompt如下：

```
[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]
```

由于LLM裁判在评测中的高效，催生了大量基于LLM的评测方法，如AlpacaEval [@dubois2024length]、Arena-Hard [@li2024crowdsourced]、WildBench [@lin2024wildbench]等，许多团队甚至直接用LLM裁判替代奖励模型生成和利用偏好数据。

“生成式奖励模型”（Generative Reward Models）也成为活跃研究领域 [@mahan2024generative] [@zhang2024generative] [@ankner2024critique]（包括专门训练为“裁判”的模型 [@kim2023prometheus]），但在奖励模型评测上，LLM裁判通常不如专业奖励模型，说明奖励建模仍是当前RLHF的重要技术。

一个常用技巧是将LLM裁判的采样温度设为0，以减少评分的随机性。

## 延伸阅读

奖励建模的学术文献在2024年逐步成熟。
早期进展主要集中在建立基准和识别行为模式。
首个奖励模型基准RewardBench为奖励模型测试提供了通用基础设施 [@lambert2024rewardbench]。
此后，奖励模型评测扩展到类似通用后训练模型的多种评测，包括已知答案的准确性评测 [@lambert2024rewardbench]，以及LLM裁判或与其他基准相关的“体验型”评测 [@wen2024rethinking]。

新基准示例包括多语种RewardBench（M-RewardBench）[@gureja2024m]、RAG-RewardBench [@jin2024rag]、RMB [@zhou2024rmb]、RM-Bench [@liu2024rm]（通用对话）、ReWordBench（拼写错误）[@wu2025rewordbench]、MJ-Bench [@chen2024mj]、多模态RewardBench [@yasunaga2025multimodal]、VL RewardBench [@li2024vlrewardbench]、VLRMBench [@ruan2025vlrmbench]（视觉语言模型）、Preference Proxy Evaluations [@frick2024evaluate]、RewardMATH [@kim2024evaluating]等。
过程奖励模型（PRM）也有自己的新基准，如PRM Bench [@song2025prmbench]、视觉类VisualProcessBench [@wang2025visualprm]和ViLBench [@tu2025vilbench]。

想了解奖励模型*训练*的最新进展，可查阅相关新方法，如面向特定方面的奖励模型 [@wang2024interpretable]、高质量人类数据集 [@wang2024helpsteer2] [@wang2024helpsteer2p]、大规模训练 [@adler2024nemotron]、大规模实验 [@touvron2023llama]、数据去偏 [@park2024offsetbias]等。
