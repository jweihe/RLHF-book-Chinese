# 构建与验证

## 依赖

HTML、EPUB 和 DOCX 使用 Python 3、Make、Pandoc 与 pandoc-crossref。请从 [Pandoc](https://github.com/jgm/pandoc/releases) 和 [pandoc-crossref](https://github.com/lierdakil/pandoc-crossref/releases) 官方发布页安装；crossref 发布说明注明其编译时使用的 Pandoc 版本，两者应匹配。本次维护使用 Pandoc 3.11 / pandoc-crossref v0.3.25a 验证 HTML 和 EPUB。

macOS 已有 Homebrew 时：

```bash
brew install pandoc pandoc-crossref python
pandoc --version
pandoc-crossref --version
```

Linux 安装 `make`、`python3`，再安装匹配系统架构的上述官方二进制。不要直接混用发行版旧 Pandoc 与最新 crossref；仓库历史遗留的 `.deb` 也不是推荐安装入口。

## 构建命令

```bash
make -j4 html epub
make check
python3 -m http.server 8000 --directory build/html
```

浏览器访问 `http://localhost:8000`。HTML 公式由 MathJax 渲染，默认需要网络；EPUB 使用内置 MathML，显示效果取决于阅读器支持。

| 命令 | 输出 |
| --- | --- |
| `make html` | `build/html/index.html`、`build/html/c/*.html` 及资源 |
| `make epub` | `build/epub/book.epub` |
| `make pdf` | `build/pdf/book.pdf` |
| `make docx` | `build/docx/book.docx` |
| `make latex` | `build/latex/book.tex` 与排版资源 |
| `make` | HTML、EPUB、PDF、DOCX 全部格式 |
| `make check` | Python 回归测试、20 页 HTML 正文与本地资源/链接/锚点检查 |

`make check` 前先运行 `make html`；检查不访问外部站点，也不能代替内容与排版校对。

## PDF 编译与校对

新版模板使用 `ctexbook`、Fandol 中文字体、TeX Gyre 西文字体、可换行代码块和中文图表编号。无需依赖操作系统中的 Noto 字体。

Ubuntu 可安装 XeLaTeX 环境：

```bash
sudo apt-get install texlive-xetex texlive-lang-chinese texlive-latex-extra texlive-fonts-recommended
make pdf
```

macOS 可安装完整 MacTeX；也可使用本次验证的 [Tectonic 0.17.0](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0)：

```bash
make pdf PDF_ENGINE=tectonic
```

首次使用 Tectonic 会联网下载宏包与 Fandol 字体，之后使用本地缓存。可通过 `TECTONIC_CACHE_DIR` 指定缓存位置。不要把依赖下载失败误认为内容错误；查看编译日志中的第一处错误。

`make latex` 输出 `build/latex/book.tex` 和 `images/`，可在该目录中继续用 XeLaTeX 或 Tectonic 编译。中文源文件保留 UTF-8，不做破坏性的“全 ASCII”替换。

PDF 编译通过后仍须渲染检查封面、目录、所有章节、图表、长公式与代码页。例如：

```bash
pdftoppm -scale-to 1600 -png build/pdf/book.pdf /tmp/rlhf-page
```

本次修订还使用 PyMuPDF 检查页面边界、文本提取与图像尺寸。README 的预览通过 `scripts/make_pdf_previews.py` 从实际 PDF 渲染生成；不要用与下载版不同的设计稿替代。

## 教学示例验证

```bash
python3 -m venv .venv
.venv/bin/pip install torch numpy jinja2
.venv/bin/python scripts/check_examples.py
```

这组检查直接执行章节中的代码片段，覆盖 KL 方向/掩码、PPO 形状与回报目标、GRPO、RLOO、DPO、梯度聚合、Top-K 算例及聊天角色。它验证局部算法与数据约定，不是完整分布式训练系统的验收。

## 自动检查与 Pages

GitHub Actions 在 PR 和主分支提交上构建 HTML/EPUB/PDF 并检查链接与教学示例，产物保存在 workflow 的 artifact 中，GitHub 下载 artifact 通常需要登录。自动检查通过后仍需阅读者校对内容。

Pages 部署与检查分开：维护者需要在仓库 Settings → Pages 选择 GitHub Actions。只有本仓库 `main` 分支的非 PR 运行会进入部署；PR 运行只检查和上传普通构建产物，不会部署；主分支部署需要先启用 Pages。确认实际部署地址可访问后再把它加入 README。

## 常见问题

- `pandoc-crossref: not found`：安装 crossref 并加入 `PATH`。
- JSON / Pandoc API 版本报错：安装兼容的 Pandoc 与 crossref。
- `xelatex: not found` / 字体缺失：安装 PDF 的额外依赖，或先运行 `make html epub`。
- 公式或引用显示异常：提供章节、小节、工具版本与日志；不得通过关闭过滤器来掩盖引用错误。
- 根目录 PDF 与构建结果不同：根目录 PDF 随正式修订更新；阅读站 `book.pdf` 在构建成功后自动刷新。Release v1.0 为旧版，请勿混淆。
