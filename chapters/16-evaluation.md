---
prev-chapter: "合成数据与蒸馏"
prev-url: "15-synthetic.html"
page-title: 评测
next-chapter: "过度优化"
next-url: "17-over-optimization.html"
---

# 评测

评测方法始终在不断演进。
理解大语言模型的评测（尤其是后训练阶段），关键在于：当前流行的评测体系其实反映了主流训练实践和目标的变迁。
虽然有挑战性的评测推动了模型能力的突破，但大多数评测的设计初衷还是为新模型提供有用的信号。

本章将以小故事的方式，梳理RLHF早期历史中流行的评测体系，帮助读者理解其中的共性、细节与常见失效模式。

RLHF与后训练的评测经历了几个明显阶段：

1. **早期聊天阶段**：最早的RLHF或偏好微调模型，评测重点是模型的对话表现，尤其是与GPT-4等强模型的比较。典型评测有MT-Bench [@zheng2023judging]、AlpacaEval [@dubois2024length]、Arena-Hard [@li2024crowdsourced]。这些评测领域现在被归类为“聊天”或“指令跟随”。
2. **多技能时代**：随着时间推移，业界逐渐认识到RLHF不仅能提升聊天能力，还能改善多种技能。例如，Tülu评测套件涵盖知识（MMLU [@hendrycks2020measuring]、PopQA [@mallen2023llm_memorization]、TruthfulQA [@lin2021truthfulqa]）、推理（BigBenchHard [@suzgun2022challenging]、DROP [@dua2019drop]）、数学（MATH [@hendrycksmath2021]、GSM8K [@cobbe2021gsm8k]）、代码（HumanEval [@chen2021codex]、HumanEval+ [@evalplus]）、指令跟随 [@zhou2023instructionfollowingevaluationlargelanguage]、安全性（多项评测综合）。这反映了后训练已被视为多面手，而不仅仅是安全或聊天的解决方案。
3. **推理与工具阶段**：当前后训练的主流方向是挑战性更高的推理与工具使用任务，包括知识密集型难题（如GPQA Diamond [@rein2023gpqa]、Humanity's Last Exam [@phan2025hle]）、复杂软件工程任务（如SWE-Bench+ [@aleithan2024swebenchplus]、LiveCodeBench [@jain2024livecodebench]），以及高难度数学题（如最近的AIME竞赛题）。

未来还会有更多新领域不断涌现。
随着AI产业化，评测的激励机制也在变化，变得多方参与、多元化。
自ChatGPT发布以来，私有评测如Scale Leaderboard [@scale2024seal]、社区驱动评测如ChatBotArena [@chiang2024chatbot]，以及第三方评测公司如ArtificialAnalysis、Epoch AI等大量涌现。
本章会结合这些评测的实际落地细节进行讲解。

## 提示格式化：从Few-shot到Zero-shot再到CoT

**Prompting**（提示工程）本质上是一个动词，但也被认为是一门可以专门练习和训练的“手艺”[@schulhoff2024prompt]。
Prompt是为语言模型组织信息和上下文的方式。
日常交互中的prompt通常很简单，但在高级场景下，精心设计的prompt往往决定了模型能否成功完成任务。

在评测中，prompt设计对模型表现影响巨大。
有些提示格式（见下文）甚至能让模型表现从60%跌到接近0。
同样，prompt的变化也能帮助模型在训练中学得更好。
业界常说，“会写prompt”能让你提前体验“未来”模型的能力，突破常规用法的天花板。

现代大模型的高阶prompt往往是一份完整的报告（动辄上千token）。
这种行为改变了模型性能的评测和理解方式。

早期语言模型只被当作智能补全工具。
若想让模型更灵活地完成任务，通常会给出多个示例，再加一个待补全的prompt，这就是few-shot或in-context learning [@brown2020language]，当时还没有指令微调或RLHF。
例如：

```
# Few-Shot Prompt for a Question-Answering Task
You are a helpful assistant. Below are example interactions to guide your style:

### Example 1
User: "What is the capital of France?"
Assistant: "The capital of France is Paris."

### Example 2
User: "Who wrote the novel '1984'?"
Assistant: "George Orwell wrote '1984.'"

# Now continue the conversation using the same style.
User: "Can you explain what a neural network is?"
Assistant:
```

对于MMLU风格的多选题，也可以这样few-shot：

```
# Few-Shot Prompt

Below are examples of MMLU-style questions and answers:

### Example 1
Q: A right triangle has legs of lengths 3 and 4. What is the length of its hypotenuse?
Choices:
(A) 5
(B) 6
(C) 7
(D) 8

Correct Answer: (A)

### Example 2
Q: Which of the following is the chemical symbol for Sodium?
Choices:
(A) Na
(B) S
(C) N
(D) Ca

Correct Answer: (A)

### Now answer the new question in the same style:

Q: Which theorem states that if a function f is continuous on a closed interval [a,b], then f must attain both a maximum and a minimum on that interval?
Choices:
(A) The Mean Value Theorem
(B) The Intermediate Value Theorem
(C) The Extreme Value Theorem
(D) Rolle’s Theorem

Correct Answer:
```

这里可以直接采样生成答案（A/B/C/D），也可以计算各选项的概率，看正确答案是否概率最大（如[@robinson2023leveraging]所述）。
概率法既可以用选项字母，也可以用完整答案文本，两种都合理，但实际评测更常用选项概率。

few-shot提示的常见问题是模型不遵守格式，导致答案判错。
设计评测时，in-context示例数量也是参数，通常3到8个甚至更多。

few-shot提示发展过程中，出现了链式思维（chain-of-thought, CoT）示例，即示例中包含详细推理过程（后来发展为明确提示模型“逐步思考” [@wei2022chain]）：

```
# standard prompting
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The answer is ...

# chain of thought prompting
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?

A: The cafeteria had 23 apples originally. They..
```

随着模型能力提升，zero-shot评测（零样本学习）成为主流 [@wei2022finetuned]。
FLAN（Finetuned Language Net）证明了指令微调后的模型能泛化到未见过的zero-shot问题 [@wei2022finetuned]（T0 [@sanh2022multitask]也有类似结果）。
这也推动了指令微调（IFT）的流行，为RLHF和后训练奠定了基础。
zero-shot问题示例如下：

```
User: "What is the capital of France?"
Assistant:
```

自2022年起，早期RLHF代表作如InstructGPT等陆续出现。
这些模型的核心能力和用例，转向更开放的生成式场景。
随着开放性增强，生成式评测愈发流行，因为这更贴近真实应用。
在ChatGPT发布后的几年里，RLHF研究仍保留多选评测作为对比。

到2024年底、2025年初，推理模型兴起，模型行为发生重大变化——每个答案前都会输出长链式思维（CoT）推理过程。
这时，模型不再需要经典的“think step by step”提示（见[@kojima2022large]）。

比如，为了让模型在多选题上输出CoT，可用如下特殊prompt（Tülu 3 [@lambert2024t]）：

```
Answer the following multiple-choice question by giving the correct answer letter in parentheses. Provide CONCISE reasoning for the answer, and make sure to finish the response with “Therefore, the answer is (ANSWER_LETTER)” where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.

Question: {question}
(A) {choice_A}
(B) {choice_B}
(C) …

Answer the above question and REMEMBER to finish your response with the exact phrase “Therefore, the answer is (ANSWER_LETTER)” where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.
```

尤其当模型用特殊格式分隔思考token和答案token时，评测体系也随之更新。
如今，评测正转向链式思维生成的开放式测试。

## 评测的使用与观察

![Epoch AI报告：主流AI评测随时间迅速饱和。CC-BY许可。](images/benchmark-performance.jpeg)

公司内部的大模型评测只能与同行横向对比（且误差很大），因为内部评测流程与外部评测不一致。
内部评测本质上是“训练集”，用于爬坡调优；
而社区用来比较领先模型的公开评测，无法确定是否被作为训练集、测试集或验证集。

随着评测分数成为企业营销的核心，评测流程在公司内部不断变化。
据说一些大厂会为GSM8k、MATH等重要评测设计“定制prompt”。
这些做法变化极快。

大模型评测体系常被视为营销手段，因为评测没有标准的“真理源”。
前沿实验室会根据自身需求调整评测套件。
公开的分数只是实验室模型的输出结果，输入细节并未全部披露。
这些输入细节极为敏感，不同公司（OpenAI、Meta、Anthropic、Google）各不相同。
即便是完全开源的评测标准，也很难保证可复现性。
专注于自家模型，是实现可重复评测的唯一途径。
当然，技术团队的出发点是好的。

如今，前沿大模型的评测既是科学，也是艺术。

不同团队会选择不同评测作为“真正的测试集”，但没人会公开选择了哪些。
比如，MATH和GSM8k等推理评测都自带训练集，prompt本身就能提升表现。
用同分布prompt提升分数，与泛化到新任务是两回事。

事实上，这些“训练集”本身就是高质量数据，模型训练时用它们会直接受益。
如果企业*没有*把相关评测作为核心指标，直接用评测集训练也很合理——高质量数据才是模型开发的最大瓶颈。

主流AI实验室往往在少数关键评测上爬坡，最后在核心公开集上报分。
有些内部跟踪指标（如GPT-4报告中的交叉熵loss预测 [@achiam2023gpt]）甚至不对外公开。

后训练评测高度依赖人工评测。
生成式大模型的人工评测常用Elo排名（如Anthropic早期论文中的宪法AI），奖励模型的人工评测则看一致性。
也可以通过A/B测试窗口让用户对比两模型（详见“偏好数据”章节）。

实验室聚焦的评测集，形成了训练与评测的紧密耦合。
比如，MMLU曾是重点评测，推理模型时代GPQA成为新宠。
实验室会不断调整评测以适应自身需求，如OpenAI发布SWE-Bench-Verified [@openai2024swebench]。
实际上还有很多内部评测集未公开。

内部评测对下游训练的最大作用，是**提升训练对比的统计效力**。
通过调整评测，实验室可以降低关键信号的噪声，更好地做训练决策。

现代大模型训练栈的后训练流程极为复杂。
评测语言模型不仅仅是看答案的log概率，而是要生成大量token。
前沿实验室常用一些“小技巧”提升任务表现——最常见的是为特定评测设计专用prompt。

另一个导致评测混乱的例子，是推理时扩展（Inference-time scaling）被引入评测对比。
推理时扩展表明，模型通过生成更多token可以提升表现。
因此，控制推理token总数对评测分数影响很大，但目前还未成为业界常规。

后训练数据格式也会导致模型在不同评测格式下表现差异巨大。
比如两个流行开源数学数据集 [@li2024numinamath] 和 MetaMath [@yu2023metamath]，仅仅因为答案格式不同（Numina用`\boxed{XYZ}`，MetaMath用`The answer is: XYZ`），联合训练反而比只用单一格式效果更差。
强模型通常能兼容多种格式，但也会有最擅长的主格式。

总之，关于闭源模型评测，我们可以总结几点：

* 我们并不知道实验室爬坡的核心测试集，所以有些评测只是代理指标。
* 前沿模型的推理流程越来越复杂，涉及特殊system prompt、特殊token等，实际对评测影响如何我们并不清楚；
* 我们也无法获知所有用于数值报告的评测格式和细节。

## 数据污染（Contamination）

当前大模型训练（不仅限于RLHF和后训练）面临的重大问题之一，是训练数据有意或无意地包含了评测数据，这就是*数据污染*（dataset contamination），而*去污染*（decontamination）则是相应的防控措施。
去污染通常通过在训练集和测试集之间做n-gram（字符或token）匹配搜索实现 [@singh2024evaluation]。
数据污染常见于多阶段网络爬取训练数据，评测集常被公开在可爬取的网站上，或用户把评测题输入模型，结果被未来模型采集进训练数据。

例如，在Tülu 3的评测去污染过程中，作者发现多个流行开源数据集都被RLHF常用评测污染 [@lambert2024t]。
如UltraFeedback与TruthfulQA、Evol-CodeAlpaca与HumanEval、NuminaMath与MATH、WildChat与安全评测等均有重合，都是通过8-gram重叠检测到的。

对于未公开训练数据的模型，研究者会制作轻微扰动的新基准（如MATH [@huang2025math]），检验模型是否专门记住了原题或原格式。
在这种扰动基准上的高方差并不等于污染，但可能暗示模型在某些格式上过拟合，未必能迁移到真实世界。

## 工具与平台

目前已有许多开源评测工具可选。
包括英国安全研究院的Inspect AI [@inspectAI2024]、HuggingFace的LightEval [@fourrier2023lighteval]（Open LLM Leaderboard背后引擎 [@open-llm-leaderboard-v2]）、Eleuther AI的evaluation harness [@gao2023evalharness]（基于GPT-Neo-X评测配置 [@gpt-neox-20b]）、AI2基于OLMES的库 [@gu2024olmes]、斯坦福CRFM的HELM [@liang2023helm]、Mosaic（现Databricks）的Eval Gauntlet [@mosaicml2024gauntlet]等。