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

## PDF 额外依赖

PDF 使用 XeLaTeX 和 Noto CJK 字体。Ubuntu 可安装：

```bash
sudo apt-get install texlive-xetex texlive-lang-chinese texlive-latex-extra fonts-noto-cjk
make pdf
```

macOS 可安装完整 MacTeX，并安装 Noto Serif CJK SC、Noto Sans CJK SC 与 Noto Sans Mono CJK SC 字体；仅安装 Pandoc 不会带来完整的 TeX 环境。字体或宏包错误请先检查 XeLaTeX 日志。HTML/EPUB 检查成功不表示 PDF 已验证。

## 自动检查与 Pages

GitHub Actions 在 PR 和主分支提交上构建 HTML/EPUB 并检查链接，产物保存在 workflow 的 artifact 中，GitHub 下载 artifact 通常需要登录。自动检查通过后仍需阅读者校对内容。

Pages 部署与检查分开：维护者需要在仓库 Settings → Pages 选择 GitHub Actions，并设置仓库变量 `ENABLE_PAGES=true`。只有本仓库 `main` 分支的非 PR 运行会进入部署；未启用 Pages 时仍可正常检查和下载构建产物。确认实际部署地址可访问后再把它加入 README。

## 常见问题

- `pandoc-crossref: not found`：安装 crossref 并加入 `PATH`。
- JSON / Pandoc API 版本报错：安装兼容的 Pandoc 与 crossref。
- `xelatex: not found` / 字体缺失：安装 PDF 的额外依赖，或先运行 `make html epub`。
- 公式或引用显示异常：提供章节、小节、工具版本与日志；不得通过关闭过滤器来掩盖引用错误。
- 根目录 PDF 与构建结果不同：历史 PDF 不会自动刷新，请明确记录发布产物对应的 commit。
