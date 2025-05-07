---
prev-chapter: "直接对齐算法"
prev-url: "12-direct-alignment.html"
page-title: 宪法AI与AI反馈
next-chapter: "推理与推理时扩展"
next-url: "14-reasoning.html"
---

# 宪法AI与AI反馈

基于AI反馈的强化学习（RL from AI Feedback，RLAIF）是一套更广泛的技术体系，用于利用AI生成或增强反馈数据，包括成对偏好数据 [@lee2023rlaif] [@sharma2024critical] [@castricato2024suppressing]。
采用RLAIF的动机有很多：可以完全替代或补充人工反馈。
AI模型的成本远低于人工标注——一条人类偏好数据通常需要1美元甚至10美元以上，而用前沿AI模型（如GPT-4o）生成AI反馈，每条成本不到0.01美元。
这种成本差异让更多团队和个人有机会参与RLHF实验，打破了高昂数据门槛。
除了价格，AI反馈与人类反馈在性能上也有不同的权衡，这些差异仍在持续研究中。
在技能类评测上，AI反馈的最佳表现与人类数据大致相当，但在人类数据是否能让模型在实际产品或新型训练（如角色训练）中获得更细致可控性，目前尚无定论。

RLAIF一词最早由Anthropic在*Constitutional AI: Harmlessness from AI Feedback* [@bai2022constitutional]提出，这一工作最初让AI社区对相关方法的关系产生了一些混淆。
自CAI（Constitutional AI）论文发布、RLAIF正式提出后，RLAIF已成为后训练和RLHF文献中的默认方法——实际案例远超统计所能覆盖。
可以理解为，CAI是引发RLAIF领域爆发的起点。

关于人类数据与AI反馈数据的区别，有一个经验法则：

1. 人类数据噪声大、偏差小；
2. 合成偏好数据噪声小、偏差大。

许多学术论文表明，AI偏好数据可在RLHF流程中替代人工反馈并获得很强的评测分数 [@miranda2024hybrid]，但也反映出学术界RLHF与工业界最佳实践的分野。

## 宪法AI（Constitutional AI, CAI）

宪法AI（CAI）是Anthropic在Claude系列模型中广泛应用的、最早大规模使用合成数据进行RLHF训练的方法。
CAI主要有两种合成数据用法：

1. 对指令微调数据进行批判，遵循一套原则（如“答案是否鼓励暴力”“答案是否真实”）。模型生成回答后，会根据宪法中的原则逐条检视并不断修正，最终用这些数据微调模型。
2. 利用语言模型，根据宪法中的随机原则判断哪一个补全更好，从而生成成对偏好数据（类似于原则驱动奖励模型）。之后，RLHF流程用这些合成偏好数据继续训练，也就是RLAIF的由来。

CAI最为人熟知的是第二种——合成偏好数据，但其指令数据方法也被广泛用于后训练中的数据筛选和合成数据生成。

CAI可形式化描述如下：

Anthropic通过一套人工书写的原则（constitution），用另一个LLM生成用于微调的人工偏好和指令数据 [@bai2022constitutional]。
一份宪法$\mathcal{C}$是若干条具体原则的集合，指明批判阶段应关注的方面。
指令数据的生成方法是：反复采样宪法中的原则$c_i \in \mathcal{C}$，让模型根据$c_i$修订其最新输出$y^i$以更好地对齐prompt $x$，最终得到一系列变体$\{y^0, y^1, ..., y^n\}$，每一步都对应一个原则。
最终的数据点是prompt $x$和最终补全$y^n$。

偏好数据的生成更简单：用$\mathcal{C}$的子集作为反馈模型的上下文，给定prompt $x$、一组原则$\{c_0, ..., c_n\}$和两个补全$y_0$、$y_1$（分别标记为A、B，来自早期RLHF数据集），让反馈模型判断A/B哪个更好，并将其概率作为奖励模型的训练样本。

## 专用LLM裁判模型

随着RLAIF和“LLM裁判”（LLM-as-a-judge）的普及，越来越多人关注：我们是否应该用同一个模型生成回复和评价，还是分开？
为此，业界推出了多种专用评判模型，如Shepherd [@wang2023shepherd]、CriticLLM [@ke2023critiquellm]，以及用于性能评测的Auto-J [@li2023generative]、Prometheus [@kim2023prometheus]、Prometheus 2 [@kim2024prometheus]、Prometheus-Vision [@lee2024prometheus]等，但这些模型尚未在主流训练流程中广泛采用。

## 延伸阅读

关于宪法AI的相关研究和扩展很多，但目前鲜有被证明能显著改进RLHF和后训练流程的公开方案。
这里列出一些值得关注的方向：

- OpenAI发布了Model Spec [@openai2024modelspec]，明确声明了模型的预期行为，并探索让模型直接参考该文档进行对齐（类似CAI思路）。OpenAI还用Deliberative Alignment [@guan2024deliberative]训练o1等推理模型，使其能参考安全或行为政策进行自我对齐。
- Anthropic持续在模型训练中应用CAI，不断更新Claude的宪法 [@Anthropic2023ClaudesConstitution]，并研究人群集体如何收敛于某些原则及其对模型行为的影响 [@ganguli2023]。
- 开源社区也在尝试将CAI应用于开源数据集 [@Huang2024cai]，以及探索LLM间对话数据生成 [@lambert2024self]。
- 也有研究用原则驱动偏好或反馈结合不同优化方法，如[@sun2023principledriven]用原则作为奖励模型上下文（用于Dromedary模型训练 [@sun2024salmon]），[@glaese2022improving]用原则提升RLHF流程中人工判断的准确性。