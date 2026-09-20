# 内容勘误与修订记录

## 2026-09 中文版修订

本次以仓库现有 19 章译稿为基础，通读并重点检查了定义、公式、实现片段及明显过度概括的表述。以下为中文版维护性修订，不代表原作者发布了同样的修订，也不是与英文出版版逐句对照完成的声明。

用于辅助比较的英文历史快照为 [`6acbd61`](https://github.com/natolambert/rlhf-book/tree/6acbd61d048cf2aded7ab918294d11f1916f2c73)（2025-04-16 附近）。这不是已确认的最初翻译基线。部分问题在该英文快照中也存在，例如 REINFORCE 的“无需奖励模型”表述；不能将所有问题简单归为翻译错误。

原稿的模型、行业与“当前最佳实践”讨论保留约 2025 年 4 月的历史语境。第一人称属于 Nathan Lambert。此次未重新调查所有行业传闻、未复现实验论文的训练结果，也未验证全部外部链接；尚需持续进行逐句、逐图与版本对照。

## 已修正的主要问题

| 章节 | 原问题 | 修订与依据 |
| --- | --- | --- |
| 1–2 | 将所有 RLHF 归为对比损失；GPT-2 年份写作 2018 | 区分奖励/偏好损失与策略梯度；GPT-2 为 2019 年。见 [GPT-2 原始报告](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)。 |
| 3 | 轨迹求积与终止状态不匹配；蒸馏公式只保留一个 token | 统一有限轨迹边界；完整分布蒸馏对词表求和。见 [知识蒸馏](https://arxiv.org/abs/1503.02531)。 |
| 3 | 将零样本“逐步思考”提示归给少样本 CoT | 区分 [Wei 等](https://arxiv.org/abs/2201.11903)与 [Kojima 等](https://arxiv.org/abs/2205.11916)。 |
| 4 | “状态转移不存在”、简化时“去掉奖励模型”；比较对象误写成 prompt 对 | 限定单轮 bandit 建模，保留 token 级 MDP；奖励评分对象为 prompt-回答，偏好比较同一 prompt 的回答。见 [InstructGPT](https://arxiv.org/abs/2203.02155)。 |
| 5 | 将效用理论起源归为模拟电路；将效用公理当作 RLHF 保证 | 改为自适应系统的相关应用，并明确个体效用表示的假设与聚合限制。 |
| 7 | RM 标量被写成相似概率或二分类结果；代码未提取 logits；多回答公式漏求和 | 区分标量分数与 sigmoid 分数差；提取 `.logits`；补齐组内比较求和与归一化。见 [InstructGPT §3.2](https://arxiv.org/abs/2203.02155)。 |
| 7 | ORM 必须成对、PRM 必须三类、价值网络输出分类 | 按监督目标区分模型类别，不以特定评分头定义整个类别。见 [训练验证器](https://arxiv.org/abs/2110.14168)、[过程监督](https://arxiv.org/abs/2305.20050)。 |
| 8 | `kl_div` 向默认概率 target 传入 log 概率；未屏蔽 prompt/padding | 设置 `log_target=True`，明确 KL 方向、词表与序列聚合、采样估计的限制；纠正 margin 的含义。见 [PyTorch 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html)。 |
| 9 | 聊天模板未严格验证角色；把末轮监督写成唯一方式 | 修正可选 system 与交替角色检查，区分全 assistant 监督和末轮监督；模板为教学用。 |
| 10 | 五个 prompt 的索引范围写成四个；Top-5 误含展平索引 19 | 更正为五个 prompt，以及索引集合 `{0, 5, 8, 11, 14}`；数值测试直接解析正文。另说明统计学拒绝采样与奖励筛选的区别，以及跨 prompt 分数不可直接比较。 |
| 11 | REINFORCE 无需奖励模型；策略条件方向反了；任意确定性策略取 max-Q | 改为可省价值网络、仍需奖励；使用 `π(a|s)`；区分固定策略价值与最优贪心价值。见 [REINFORCE](https://link.springer.com/article/10.1007/BF00992696)、[Sutton 与 Barto 教材](http://incompleteideas.net/book/the-book-2nd.html)。 |
| 11 | PPO 是硬信任域约束；裁剪 logratio；逐 token 与序列裁剪等价 | 改为裁剪概率比的代理目标，不保证硬 KL 或参数边界；重新整理逐 token 公式。见 [PPO](https://arxiv.org/abs/1707.06347)。 |
| 11 | GRPO 全对/全错组优势更大；同一 prompt 的所有回答共享优势 | 完全相同奖励的组优势为 0；每条回答内部共享优势；明确稳定项、分组布局与标准差约定。见 [DeepSeekMath §4.1](https://arxiv.org/html/2402.03300v3)、[Dr. GRPO](https://arxiv.org/abs/2503.20783)。 |
| 11 | GAE 偏差/方差写反；PPO 多加维度并用即时奖励训练价值函数 | 修正自举与蒙特卡洛的权衡；示例使用固定 GAE 与回报目标、有效 token 统计，明确终止与截断。见 [GAE](https://arxiv.org/abs/1506.02438)。 |
| 11 | 白化等同于 0–1 缩放；聚合算例为 2.27；梯度未清零 | 改为均值 0、方差 1；token 平均为 2.2、序列平均为 2.35；独立比较三种聚合梯度。 |
| 12 | DPO 不训练策略、β 直接固定 KL、max/min 目标值错误相等 | 区分策略参数与隐式奖励；β 不直接指定实际 KL；用 argmax/argmin 推导最优策略关系，明确有限训练的限制。见 [DPO §4 与附录](https://arxiv.org/html/2305.18290v3)。 |
| 12 | IPO 译为“身份”；ORPO、SimPO 描述缺关键机制 | IPO 改为“恒等偏好优化”；说明 ORPO 的 odds ratio 与 SimPO 的无参考模型、长度归一化和奖励间隔。见 [IPO](https://arxiv.org/abs/2310.12036)、[ORPO](https://arxiv.org/abs/2403.07691)、[SimPO](https://arxiv.org/abs/2405.14734)。 |
| 13–14 | AI 生成“人工偏好”；RL 仅更新正确样本；固定年代/轮数作为必要条件 | 区分人工原则与 AI 标签；负优势也可更新错误回答；限定 o1 已公开信息与 R1 报告的证据范围。见 [CAI](https://arxiv.org/abs/2212.08073)、[DeepSeek-R1](https://arxiv.org/abs/2501.12948)。 |
| 15–18 | 将合成数据成功视为消除坍塌风险；混淆测试与调参数据；真实质量曲线称作训练损失；固定 β 等于固定 KL | 限定经验结论的范围，澄清数据污染与独立评测，纠正代理奖励和真实目标的区别。参数量与单榜分数不能决定综合能力。 |

## 如何复核

- `make html epub && make check`：构建、章节链接与引用锚点检查。
- 安装 `torch numpy jinja2` 后执行 `python3 scripts/check_examples.py`：从书中提取实际代码块，检查 KL 方向与掩码、PPO 形状和价值目标、GRPO 零方差组与梯度方向、RLOO 分组、DPO 梯度、独立梯度聚合、Top-K 索引和聊天角色。
- `make pdf` 或 `make pdf PDF_ENGINE=tectonic`：从同一 Markdown 生成新版 PDF；再按构建指南渲染页面检查，不能只凭编译成功判定排版正确。

后续纠错请保留原文片段、章节位置、推导或可复现例子和一手来源。新增解释应标明维护性修订或译者注，不把作者历史观点改写成未经验证的当下事实。
