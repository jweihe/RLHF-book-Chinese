---
prev-chapter: "正则化"
prev-url: "08-regularization.html"
page-title: 指令微调
next-chapter: "拒绝采样"
next-url: "10-rejection-sampling.html"
---

# 指令微调（Instruction Finetuning）

早期的语言模型仅被训练用于预测序列中的下一个token，并未针对具体任务进行适配。
大约在GPT-3发布 [@brown2020language] 时，语言模型主要通过“上下文学习”（in-context learning）使用，即给模型展示一些示例，然后让其完成类似任务。

这其实结合了自然语言处理（NLP）领域的两大趋势——历史上模型多为某一具体任务而训练。
而随着模型规模增大，多个研究结果显示，标准化任务数据的处理方式可以极大提升下游表现。
统一任务框架的代表性工作包括 T5 模型（*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*）[@raffel2020exploring]、FLAN数据集（*Finetuned Language Models Are Zero-Shot Learners*）[@wei2021finetuned]、T0模型（*Multitask Prompted Training Enables Zero-Shot Task Generalization*）[@sanh2021multitask]、Natural Instructions数据集（*Cross-Task Generalization via Natural Language Crowdsourcing Instructions*）[@mishra2021cross] 等。
这些洞见推动了“微调”语言模型的时代到来。
在RLHF及相关方法出现之前，所有微调几乎都是**指令微调**（Instruction Finetuning, IFT），也称为**有监督微调**（Supervised Finetuning）。

如今，指令微调（Instruction Tuning）已非常成熟，成为众多语言建模流程中的标准步骤。
本质上，IFT是将语言模型适配到特定任务的最简单方法。
它为RLHF打下了基础，使模型能够适应标准的指令格式（如问答），也是现代技术应用到新领域时的首选工具。

指令微调本质上使用的还是预训练语言模型的自回归损失函数。

## 聊天模板与指令结构

RLHF流程中的核心环节之一，是将用户请求格式化为tokenizer和语言模型易于处理的格式。
负责管理用户交互结构的工具被称为**聊天模板**（chat template）。

下面是一个聊天模板的代码示例，我们将逐步解析：

```jinja
{% if messages[0]['role'] == 'system' %}
    {% set offset = 1 %}
{% else %}
    {% set offset = 0 %}
{% endif %}

{{ bos_token }}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}
{% endfor %}

{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
```
这段代码会把Python中包含消息和角色的字典列表，转换为语言模型可预测的token序列。

所有输入模型的信息都会被赋予一个角色（role）。
传统上有三种角色：`system`、`user` 和 `assistant`。

- `system`标签只用于对话的第一条消息，通常包含不会直接暴露给用户的系统指令。这些**system prompt**可为模型提供额外上下文（如日期时间），或修正特定行为。例如，可以让模型“你是一个总是用海盗风格回复的友好聊天机器人”。
- `user`即用户输入，`assistant`则为模型输出。

要将这些信息转为token序列，就需要使用上述代码。
模型会用一系列**特殊token**来分隔不同消息。
举例：如果用户提问“How many helicopters can a human eat in one sitting?”，传入模型的token序列大致如下：

```
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
```

注意，序列最后是`<|im_start|>assistant`，这提示模型继续生成token，直到遇到序列结束token（如`<|im_end|>`）。

通过将所有问答对（以及后续偏好微调数据）都封装成这种格式，现代语言模型能始终如一地遵循这一交互协议。这也是指令微调模型与用户、与存储在GPU等设备上的模型之间传递信息的“语言”。

多轮对话也可以直接扩展为如下格式：

```
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
Oh just 6.<|im_end|>
<|im_start|>user
Are you sure about that?<|im_end|>
<|im_start|>assistant
```

在开源生态中，常用的做法是将聊天模板以jinja代码形式保存在tokenizer中，通过`apply_chat_template`自动应用。

上述模板衍生自OpenAI早期的Chat Markup Language（ChatML），旨在规范消息格式。
现在，OpenAI及其他模型提供商采用更为分层的系统，允许用户自定义system message，同时还可能有更高层级的隐藏指令 [@wallace2024instruction]。

市面上还有许多其他聊天模板。例如，Zephyr的模板 [@tunstall2023zephyr]：

```
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
```

Tülu的模板：

```
<|user|>
How are you doing?
<|assistant|>
I'm just a computer program, so I don't have feelings, but I'm functioning as expected. How can I assist you today?<|endoftext|>
```

此外，许多聊天模板还会包含工具调用等任务的特殊格式和token。

## 指令微调的最佳实践

指令微调作为后训练和构建有用语言模型的基础，已被广泛验证。
实现高效指令微调的方法有很多。
例如，部分参数量化的高效微调（如QLoRA）大大降低了训练门槛 [@dettmers2023qlora]。
在对话对齐等窄领域（不涉及复杂技能如数学或编程），小规模、高质量数据集也能取得很强表现 [@zhou2023lima]。

ChatGPT发布后不久，仅1万条样本（如No Robots数据集）的人类数据就能达到SOTA [@no_robots]。
几年后，大规模合成数据集在大多数任务上效果最佳 [@lambert2024t]。

一些通用原则包括：

* 高质量数据是性能提升的关键。模型真正学到的是“补全内容”，而非prompt（很多情况下prompt不参与预测）。
* 约100万条prompt即可训练出优秀的RLHF和后训练模型。继续扩展数据规模虽有提升，但收益递减明显。
* 最佳prompt应与下游任务分布相似。
* 若指令微调后还有多阶段训练，模型可一定程度上修复流程中的噪声，整体优化优先于单阶段最优。
* 指令微调（及后训练、直接对齐算法等）中，通常会对prompt部分做mask，仅对补全token计算损失。
* 多轮训练样本同理——只对“最后一轮”生成部分计损失，前面assistant的回复可作为prompt但被mask。