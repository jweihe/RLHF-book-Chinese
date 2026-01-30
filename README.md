# RLHF 中文手册

[![代码许可证](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/jweihe/RLHF-book-Chinese/blob/main/LICENSE-Code.md)
[![内容许可证](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey)](https://github.com/jweihe/RLHF-book-Chinese/blob/main/LICENSE-Content.md)
[![构建状态](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/jweihe/RLHF-book-Chinese)
[![Pandoc](https://img.shields.io/badge/built%20with-Pandoc-blue)](https://pandoc.org/)

> **基于人类反馈的强化学习（RLHF）技术指南**
>
> 面向语言模型的后训练 RLHF 简明手册，涵盖从基础理论到最新研究进展的完整知识框架。

---

## 📖 简介

本手册是 [`rlhf-book`](https://github.com/natolambert/rlhf-book) 的中文翻译版本，旨在为中文社区提供高质量的 RLHF 技术资源。

**主要内容：**
- RLHF 的技术根源与跨学科理论融合
- 完整的 RLHF 流程：指令调优 → 奖励模型训练 → 策略优化
- 关键算法：拒绝采样、强化学习、直接对齐
- 前沿话题：合成数据、评估方法、过优化问题
- 开放性科学问题与研究方向

**适合读者：**
- 具有定量分析背景的研究者
- 对大语言模型训练感兴趣的开发者
- 想深入了解 RLHF 技术原理的从业者

---

## 📥 快速开始

### 阅读手册

**推荐方式：** 直接下载 [PDF 版本](https://github.com/jweihe/RLHF-book-Chinese/blob/main/RLHF-book-Chinese.pdf) 阅读

**其他格式：**
- [EPUB 版本](https://github.com/jweihe/RLHF-book-Chinese/releases)（适合电子书阅读器）
- [HTML 版本](https://github.com/jweihe/RLHF-book-Chinese/releases)（适合浏览器阅读）

### 本地构建

如果你想自己构建手册：

#### 1. 安装依赖

**Linux:**
```bash
sudo apt-get install pandoc make texlive-fonts-recommended texlive-xetex
```

**macOS:**
```bash
brew install pandoc make pandoc-crossref
```

#### 2. 克隆仓库
```bash
git clone https://github.com/jweihe/RLHF-book-Chinese.git
cd RLHF-book-Chinese
```

#### 3. 构建文档
```bash
make          # 生成所有格式
make pdf      # 仅生成 PDF
make epub     # 仅生成 EPUB
make html     # 仅生成 HTML
```

构建产物将输出到 `build/` 目录。

---

## 📚 目录结构

```
RLHF-book-Chinese/
├── chapters/              # 章节源文件（Markdown）
│   ├── 01-introduction.md
│   ├── 02-related-works.md
│   ├── 03-setup.md
│   ├── 04-optimization.md
│   ├── 05-preferences.md
│   ├── 06-preference-data.md
│   ├── 07-reward-models.md
│   ├── 08-regularization.md
│   ├── 09-instruction-tuning.md
│   ├── 10-rejection-sampling.md
│   ├── 11-policy-gradients.md
│   ├── 12-direct-alignment.md
│   ├── 13-cai.md
│   ├── 14-reasoning.md
│   ├── 15-synthetic.md
│   ├── 16-evaluation.md
│   ├── 17-over-optimization.md
│   ├── 18-style.md
│   ├── 19-character.md
│   └── bib.bib
├── images/                # 图片资源
├── templates/             # Pandoc 模板
├── metadata.yml           # 元数据配置
├── Makefile               # 构建脚本
├── README.md              # 本文件
├── LICENSE-Code.md        # 代码许可证（MIT）
├── LICENSE-Content.md     # 内容许可证（CC-BY-NC-SA-4.0）
└── RLHF-book-Chinese.pdf  # 预构建 PDF
```

---

## 📑 章节概览

| 章节 | 标题 | 主要内容 |
|------|------|----------|
| 01 | 简介 | RLHF 概述与技术背景 |
| 02 | 相关工作 | 文献综述与理论基础 |
| 03 | 环境设置 | 实验环境与工具配置 |
| 04 | 优化基础 | RLHF 优化框架介绍 |
| 05 | 偏好设置 | 人类偏好数据收集 |
| 06 | 偏好数据 | 数据集构建与处理 |
| 07 | 奖励模型 | 奖励模型训练方法 |
| 08 | 正则化 | 模型正则化技术 |
| 09 | 指令调优 | 指令微调实践 |
| 10 | 拒绝采样 | 拒绝采样算法 |
| 11 | 策略梯度 | 策略梯度方法 |
| 12 | 直接对齐 | 直接偏好优化（DPO）等 |
| 13 | CAI | 上下文感知指令调优 |
| 14 | 推理能力 | 推理增强技术 |
| 15 | 合成数据 | 合成数据生成 |
| 16 | 评估方法 | 模型评估指标 |
| 17 | 过优化 | 过优化问题分析 |
| 18 | 风格控制 | 输出风格调整 |
| 19 | 性格塑造 | 模型性格定制 |

---

## 🔗 引用本手册

如果你在研究或项目中使用了本手册，请按以下格式引用：

```bibtex
@book{rlhf-chinese-handbook-2025,
  author       = {He, Junwei},
  title        = {RLHF 中文手册},
  subtitle     = {基于人类反馈的强化学习技术指南},
  year         = {2025},
  url          = {https://github.com/jweihe/RLHF-book-Chinese},
  note         = {翻译自 Nathan Lambert 去作《Reinforcement Learning from Human Feedback》}
}
```

---

## 🛠️ 技术栈

本项目基于 [Pandoc 书籍模板](https://github.com/wikiti/pandoc-book-template) 构建，使用以下工具：

- **[Pandoc](https://pandoc.org/)** - 通用文档转换器
- **[LaTeX](https://www.latex-project.org/)** - PDF 排版引擎
- **[Make](https://www.gnu.org/software/make/)** - 构建自动化

---

## 📄 许可证

- **代码**：[MIT License](LICENSE-Code.md)
- **内容**：[CC-BY-NC-SA 4.0](LICENSE-Content.md)

> **注意**：本翻译版本仅用于技术传播目的，原内容版权归 Nathan Lambert 所有。使用本手册时请遵守相关许可协议。

---

## 🤝 贡献

欢迎贡献！如果你发现翻译错误或有改进建议：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

**贡献类型：**
- 🐛 Bug 修复
- 📝 翻译改进
- ✨ 新功能
- 📚 文档完善
- 🎨 样式优化

---

## 📞 联系方式

- **原作者**：[Nathan Lambert](https://github.com/natolambert)
- **译者**：[Junwei He He](https://github.com/jweihe)
- **问题反馈**：[提交 Issue](https://github.com/jweihe/RLHF-book-Chinese/issues)

---

## 🙏 致谢

感谢以下项目和社区的支持：

- [Nathan Lambert](https://github.com/natolambert) 的原英文手册
- [Pandoc](https://pandoc.org/) 团队提供的强大工具
- 所有为本项目贡献建议和反馈的朋友

---

## 📊 项目状态

- ✅ 翻译完成
- ✅ PDF 构建通过
- ✅ 文档完善
- 🔄 持续更新中

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！**

[⬆ 返回顶部](#rlhf-中文手册)

</div>
