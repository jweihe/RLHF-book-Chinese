# RLHF 中文手册 · RLHF Book Chinese

[![文档构建](https://github.com/jweihe/RLHF-book-Chinese/actions/workflows/static.yml/badge.svg)](https://github.com/jweihe/RLHF-book-Chinese/actions/workflows/static.yml)
[![内容许可](https://img.shields.io/badge/content-CC_BY--NC--SA_4.0-blue)](LICENSE-Content.md)
[![代码许可](https://img.shields.io/badge/code-MIT-green)](LICENSE-Code.md)
[![欢迎贡献](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

**从人类偏好到语言模型后训练：系统学习 SFT、奖励建模、PPO、DPO 与 AI 反馈。**

本项目是 Nathan Lambert 开源书籍 [Reinforcement Learning from Human Feedback](https://github.com/natolambert/rlhf-book) 的中文翻译，包含 **19 章**，覆盖基础概念、训练方法、评测与开放问题。适合具有机器学习基础、希望理解大语言模型后训练的学生、工程师和研究者。

**[在线阅读](https://jweihe.github.io/RLHF-book-Chinese/) · [下载 PDF](https://github.com/jweihe/RLHF-book-Chinese/raw/refs/heads/main/RLHF-book-Chinese.pdf) · [按章阅读](#章节导航) · [提 Issue](https://github.com/jweihe/RLHF-book-Chinese/issues/new/choose) · [Fork 项目](https://github.com/jweihe/RLHF-book-Chinese/fork) · [参与贡献](CONTRIBUTING.md)**

> **版本说明：** 当前译本元数据标注为 **2025 年 4 月 16 日**，这不是已核实的上游 commit。尚未完成与英文出版版或原站最新版的逐章核对，因此不承诺内容完全一致。仓库 Markdown、预构建 PDF 与 Release 可能处于不同修订版本；阅读最新修订请以 `chapters/` 为准。详见 [版本与常见问题](docs/FAQ.md)。

## 从哪里开始

- **快速建立全局认识：** 第 1 → 3 → 4 章，理解 RLHF 在后训练中的位置、符号和训练流程。
- **沿着训练流程学习：** 第 9 → 6 → 7 → 8 → 11 → 12 章，串起 SFT、偏好数据、奖励模型、正则化、PPO 与 DPO。
- **进一步阅读研究专题：** 第 13–17 章，了解 AI 反馈、推理、合成数据、评测与过度优化。

建议具备概率、梯度优化与 Transformer 基础；强化学习术语可从第 3 章查起。本仓库以教材内容和示例为主，不是可直接启动大规模训练的框架。

## 阅读方式

| 方式 | 入口与说明 |
| --- | --- |
| 在线阅读 | [中文阅读站](https://jweihe.github.io/RLHF-book-Chinese/)，含全书与 19 个独立章节页 |
| PDF | [仓库 PDF](RLHF-book-Chinese.pdf)，适合离线阅读，可能落后于源文件 |
| Markdown | 下方 19 章链接，适合阅读修订、检索和提交纠错；GitHub 不完整支持 Pandoc 引用语法 |
| Release | [发布记录](https://github.com/jweihe/RLHF-book-Chinese/releases)；当前 `v1.0` 附件为 PDF |
| HTML / EPUB | 可在本地生成，见[构建指南](docs/BUILDING.md)；不将未发布的格式标成可下载版本 |

## 章节导航

| 章 | 主题 | 阅读重点 |
| --- | --- | --- |
| 01 | [引言](chapters/01-introduction.md) | RLHF 与后训练的全局图景 |
| 02 | [关键相关工作](chapters/02-related-works.md) | 技术发展与代表性论文 |
| 03 | [定义与背景](chapters/03-setup.md) | 语言建模、强化学习及 RLHF 术语 |
| 04 | [训练概览](chapters/04-optimization.md) | 训练阶段与优化问题 |
| 05 | [偏好的本质](chapters/05-preferences.md) | 偏好的理论基础与建模假设 |
| 06 | [偏好数据](chapters/06-preference-data.md) | 数据来源、收集与质量 |
| 07 | [奖励建模](chapters/07-reward-models.md) | 奖励模型的目标与训练 |
| 08 | [正则化](chapters/08-regularization.md) | KL 约束及策略偏移 |
| 09 | [指令微调](chapters/09-instruction-tuning.md) | IFT / SFT 与数据设计 |
| 10 | [拒绝采样](chapters/10-rejection-sampling.md) | 生成、打分和筛选 |
| 11 | [策略梯度算法](chapters/11-policy-gradients.md) | PPO、GRPO、RLOO 等方法 |
| 12 | [直接对齐算法](chapters/12-direct-alignment.md) | DPO 等偏好优化方法 |
| 13 | [宪法 AI 与 AI 反馈](chapters/13-cai.md) | Constitutional AI、RLAIF 与 LLM 裁判 |
| 14 | [推理训练与推理时扩展](chapters/14-reasoning.md) | 推理能力与计算预算 |
| 15 | [合成数据与蒸馏](chapters/15-synthetic.md) | 数据生成与知识迁移 |
| 16 | [评测](chapters/16-evaluation.md) | 基准、比较与评测设计 |
| 17 | [过度优化](chapters/17-over-optimization.md) | 奖励过优化与泛化问题 |
| 18 | [风格与信息](chapters/18-style.md) | 回答风格与信息质量 |
| 19 | [产品、用户体验与模型个性](chapters/19-character.md) | 后训练与实际用户体验 |

参考文献保存在 [`chapters/bib.bib`](chapters/bib.bib)。

## 本地构建

HTML / EPUB 需要 Python 3、Make、Pandoc 和版本兼容的 pandoc-crossref；PDF 还需要 XeLaTeX、LaTeX 宏包与中文字体。完整安装步骤、输出位置和常见错误见 [构建指南](docs/BUILDING.md)。

```bash
git clone https://github.com/jweihe/RLHF-book-Chinese.git
cd RLHF-book-Chinese
make html              # 首页、19 个章节页及资源
make epub              # EPUB，公式使用 MathML
make check             # 回归测试与 HTML 本地链接检查
```

`make pdf` 生成 PDF，`make docx` 生成 Word 文件，`make` 生成全部格式。所有产物位于 `build/`。

## 欢迎 Fork、Issue 和 PR

**不必一次贡献一整章。** 修正一个错字、说明一个公式中的疑问、补上一个失效链接，都能帮助后来的读者。

- **发现问题：** [提交 Issue](https://github.com/jweihe/RLHF-book-Chinese/issues/new/choose)，写清章节、原文位置、问题和建议；不确定怎么改也可以报告。
- **直接改进：** [Fork 仓库](https://github.com/jweihe/RLHF-book-Chinese/fork)，修改 Markdown 并提交 PR。流程与术语规范见 [贡献指南](CONTRIBUTING.md)。
- **帮助传播：** Star 收藏，将具体章节分享给学习小组、课程或同事，并保留原作者与中文仓库链接。

优先欢迎：翻译校对、术语一致性、公式与引用检查、构建与阅读体验修复，以及有来源的上游版本对照。新增解释请标明「译者注」，避免与原作者观点混淆。

## 维护方向

- [ ] 逐章记录对应的英文上游版本与差异。
- [ ] 整理统一术语表与读者勘误。
- [ ] 校验多格式产物后发布带版本记录的阅读包。
- [x] 提供 [GitHub Pages 在线阅读入口](https://jweihe.github.io/RLHF-book-Chinese/)，并在主分支检查通过后自动部署。

这些是开放的贡献方向，不代表已经完成或承诺发布时间。欢迎先用 Issue 讨论范围。

## 引用与致谢

原著作者：[Nathan Lambert](https://github.com/natolambert)。中文翻译维护：[Junwei He](https://github.com/jweihe)；仓库元数据注明使用 GPT-4.1 辅助翻译。感谢 [所有贡献者](https://github.com/jweihe/RLHF-book-Chinese/graphs/contributors)，以及 [Pandoc](https://pandoc.org/) 和 [pandoc-book-template](https://github.com/wikiti/pandoc-book-template)。

引用具体理论或研究时，请引用原著及对应论文；引用本中文资源时，可使用：

```bibtex
@misc{rlhf_book_chinese,
  author = {Lambert, Nathan},
  title  = {RLHF 中文手册},
  year   = {2025},
  url    = {https://github.com/jweihe/RLHF-book-Chinese},
  note   = {中文翻译维护：Junwei He；请同时注明所使用的 commit 或发布版本}
}
```

内容与代码分别采用 [CC BY-NC-SA 4.0](LICENSE-Content.md) 与 [MIT](LICENSE-Code.md) 许可；原著、插图和引用材料保留原有署名及许可说明。
