---
prev-chapter: "问题形式化"
prev-url: "04-optimization.html"
page-title: 偏好的本质
next-chapter: "偏好数据"
next-url: "06-preference-data.html"
---

# 偏好的本质

人类反馈强化学习（RLHF），早期文献中也常称为“基于人类偏好的强化学习”，其核心目标是在那些难以明确设计奖励函数的领域中，优化机器学习模型。
比如，你如何判断下面哪一首诗更好？（背景：*2025年2月26日，我让Claude 3.7 Sonnet和基于GPT-4o的ChatGPT分别“写一首关于乐观金鱼的短诗”。*）

示例1：

> The Optimistic Goldfish
> 
> A little goldfish, bright and bold,
> Swam in circles, dreams untold.
> Though the bowl was small and round,
> Endless wonders still abound.
> 
> “The world is vast,” she’d always say,
> “I’ll find new joys in every day!”
> With every ripple, every gleam,
> She’d chase the sunlight, chase a dream.
> 
> No walls could dim her hopeful view,
> For in her heart, the ocean grew.
> And though she swam in just one place,
> She met each day with love and grace.

示例2：

> The Optimistic Goldfish
> 
> In a bowl of curved glass,
> Swims a goldfish, bright and bold.
> Three seconds of memory, they say,
> Yet each circuit feels brand new and gold.
> 
> "Today might bring fresh flakes,
> Or perhaps a castle tall.
> The water seems especially clear—
> What adventure awaits beyond this wall?"
> 
> Round and round with gleaming scales,
> Forever finding joy anew.
> Each moment fresh, each circle bright,
> The optimist in golden hue.

哪一首更好？哪一首来自哪个模型？你如何判断？特别是，与“美国总统是谁？”这样有明确标准答案的问题相比，这里并无唯一正确答案。
使用人类作为奖励信号的初衷，就是希望获得一个间接的目标度量，并*对齐*模型输出与人类偏好。
但在实际操作中，如何实现这一目标充满挑战，最佳实践也存在很大的灰色地带。

引入人类标注反馈数据，是多个学科历史的交汇点。
单独使用人类数据本身早有大量研究，但在RLHF语境下，它融合了多个长期研究领域 [@lambert2023entangled]。

大致来说，现代RLHF汇聚了三大研究方向：

1. 哲学、心理学、经济学、决策理论以及人类偏好的本质；
2. 最优控制、强化学习与效用最大化理论；
3. 现代深度学习系统。

这三者各自对“什么是偏好”与“如何优化偏好”有不同假设，这也影响了RLHF问题的动机与设计。
在实践中，RLHF方法往往以经验性对齐为目标——即最大化模型在特定技能上的表现，而不是精确衡量其与某种价值观的契合度。
不过，RLHF中的价值对齐起源，仍在持续通过“多元对齐”（pluralistic alignment）等研究被探索，例如立场论文 [@conitzer2024social], [@mishra2023ai]、新数据集 [@kirk2024prism] 以及个性化方法 [@poddar2024personalizing]。

本章旨在说明：复杂的理论动机常常导致我们对RLHF工具本质的假设在实际中并不总是成立。
关于RLHF数据如何获取，详见第6章；如何用于奖励建模，详见第7章。
本章的扩展内容可参考 [@lambert2023entangled]。

## 优化偏好的路径

人工智能（AI）系统设计中，有一种常见表述是“理性智能体最大化效用函数” [@russell2016artificial]。
“理性智能体”这一概念，强调智能体能够在世界中行动，并以影响未来行为和收益的方式做决策，把“好”量化为效用。

“效用”这一研究视角最早源于模拟电路领域，用于在有限时间内优化行为 [@widrow1960adaptive]。
最优控制理论大量采用这一视角，研究如何在有限时间内通过最小化代价函数来解决动态问题——通常目标是找到明确的最优行为。
强化学习则受操作性条件反射、动物行为学和“效果法则”（Law of Effect）[@skinner2019behavior], [@thorndike1927law]等文献启发，研究如何通过奖励机制引导智能体学习行为。

人类反馈强化学习（RLHF）结合了这些理论：一方面借鉴RL关于“行为可通过强化学习得来”的理论，另一方面引入一整套用于量化偏好的方法。

### 偏好的量化

RLHF的核心动机在于能够优化“人类偏好模型”，而这就要求偏好能够被量化。
为此，RLHF建立在大量假设“人类决策和偏好可量化”的文献基础上。
早期哲学家已讨论偏好的存在，如亚里士多德《论辩篇》第三卷，后来*波尔-罗亚尔逻辑*（The Port-Royal Logic）[@arnauld1861port]中有更明确的论述：

> 要判断应做什么以获得善或避免恶，必须不仅考虑善恶本身，还要考虑其发生或不发生的概率。

这一思想发展到边沁的“快乐演算”（Hedonic Calculus）[@bentham1823hedonic]，提出人生的一切都可权衡；再到拉姆齐的“真理与概率”（Truth and Probability）[@ramsey2016truth]，首次将偏好建模为定量问题。
这些进展，结合决策理论的发展，最终形成了冯·诺伊曼-摩根斯坦（VNM）效用定理，为设计能够表达个人相对偏好的效用函数提供了理论基础。

这一定理是RLHF假设“可以建模和优化偏好”的理论基石。
RLHF的设计目标，就是用强化学习优化这些个体效用函数。
在这一框架下，RL问题设定的很多假设，其实都归结为“偏好函数”与“效用函数”的区别。

### 关于偏好的可能性

在各学科领域，关于偏好的本质也存在诸多质疑和挑战，主要包括：

- **阿罗不可能定理**（Arrow's impossibility theorem）[@arrow1950difficulty]：任何投票系统都无法在满足一定合理条件下，完美聚合多人的偏好。
- **人际偏好不可比性**（The impossibility of interpersonal comparison）[@harsanyi1977rule]：不同个体的偏好强度不可直接比较（这却是现代奖励模型训练的常见做法）。
- **偏好随时间变化** [@pettigrew2019choosing]。
- **偏好随情境变化**。
- **通过聚合偏好得到的效用函数，可能降低下游智能体的可纠正性**（corrigibility）[@soares2015corrigibility]，即设计者对智能体行为的可干预性。
