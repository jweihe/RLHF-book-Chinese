---
prev-chapter: "引言"
prev-url: "01-introduction.html"
page-title: 相关工作
next-chapter: "定义与背景"
next-url: "03-setup.html"
---

# 关键相关工作

本章将介绍推动RLHF（人类反馈强化学习）领域发展至今的核心论文和项目。
这不是一份关于RLHF及其相关领域的全面综述，而是一个起点，带你梳理这一路走来的关键节点。
内容有意聚焦于促成ChatGPT诞生的近年工作。
在RL领域，关于偏好学习还有大量更深入的研究 [@wirth2017survey]。
如果你需要更详尽的文献列表，建议查阅专门的综述论文 [@kaufmann2023survey], [@casper2023open]。

## 起源至2018年：基于偏好的强化学习

近年来，随着深度强化学习（Deep RL）的发展，这一领域逐渐被大众熟知，并扩展为各大科技公司探索大语言模型（LLM）应用的主流方向之一。
不过，今天许多RLHF核心技术其实都可以追溯到早期关于“基于偏好强化学习”的文献。

*TAMER: Training an Agent Manually via Evaluative Reinforcement*（TAMER：通过评价型强化手动训练智能体）首次提出了让人类对智能体行为进行打分，进而训练奖励模型的思想 [@knox2008tamer]。几乎同期，COACH等方法提出了基于actor-critic结构的算法，将人类正负反馈用于调整优势函数 [@macglashan2017interactive]。

Christiano等人在2017年的论文是RLHF领域的里程碑，他们首次将RLHF应用于Atari游戏轨迹的偏好学习 [@christiano2017deep]。研究发现，在某些场景下，让人类在多个轨迹间做选择，比直接与环境交互更高效。该方法采用了一些巧妙的设计，尽管如此，成果依然令人印象深刻。
随后，这一方法被进一步扩展，提出了更直接的奖励建模方式 [@ibarz2018reward]。
TAMER也在一年后被引入深度学习领域，发展为Deep TAMER [@warnell2018deep]。

这一时期的转折点在于：奖励模型（reward model）被提出作为研究“对齐”（alignment）的通用工具，而不仅仅是解决RL问题的手段 [@leike2018scalable]。

## 2019至2022年：人类偏好驱动的语言模型强化学习

人类反馈强化学习（RLHF），早期也常被称为“基于人类偏好的强化学习”，很快被AI实验室采纳，成为大语言模型扩展能力的重要方法。
大量相关工作始于2018年的GPT-2与2020年的GPT-3之间。
2019年最早的代表作 *Fine-Tuning Language Models from Human Preferences* 与现代RLHF研究有诸多相似之处 [@ziegler2019fine]，如奖励模型、KL距离、反馈流程图等——只是当时的评测任务和模型能力有所不同。
此后，RLHF被广泛应用于多种任务。
当时最受关注的应用领域包括：
- 通用文本摘要 [@stiennon2020learning]
- 递归式书籍摘要 [@wu2021recursively]
- 指令遵循（InstructGPT）[@ouyang2022training]
- 浏览器辅助问答（WebGPT）[@nakano2021webgpt]
- 答案引用支持（GopherCite）[@menick2022teaching]
- 通用对话（Sparrow）[@glaese2022improving]

除了应用实践，一些奠定RLHF未来方向的论文也值得关注，包括：
1. 奖励模型过度优化（Reward model over-optimization）[@gao2023scaling]：强化学习优化器可能会对偏好数据过拟合；
2. 语言模型作为对齐研究的通用载体 [@askell2021general]；
3. Red teaming（红队测试）[@ganguli2022red]：即评估语言模型安全性的流程。

RLHF在对话模型中的应用也在不断完善。
Anthropic在Claude早期版本中大量采用RLHF [@bai2022training]，同时首批RLHF开源工具也陆续出现 [@ramamurthy2022reinforcement], [@havrilla-etal-2023-trlx], [@vonwerra2022trl]。

## 2023年至今：ChatGPT时代

ChatGPT的发布明确强调了RLHF在其训练中的关键作用 [@openai2022chatgpt]：

> 我们使用人类反馈强化学习（RLHF）训练了该模型，方法与InstructGPT相同，但数据收集方式略有不同。

自此之后，RLHF被广泛应用于主流大语言模型的训练中。
例如，Anthropic的Constitutional AI（Claude）[@bai2022constitutional]，Meta的Llama 2 [@touvron2023llama] 和 Llama 3 [@dubey2024llama]，Nvidia的Nemotron [@adler2024nemotron]，Ai2的Tülu 3 [@lambert2024t] 等等。

如今，RLHF正逐步发展为更广义的“偏好微调”（Preference Fine-Tuning，PreFT）领域，涵盖了许多新兴应用，比如：
- 针对中间推理步骤的过程奖励（process reward）[@lightman2023let]
- 受直接偏好优化（DPO, Direct Preference Optimization）启发的直接对齐算法 [@rafailov2024direct]
- 基于代码或数学题执行反馈的学习 [@kumar2024training], [@singh2023beyond]
- 以及受OpenAI o1启发的在线推理方法 [@openai2024o1]