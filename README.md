# RLHF 中文手册  
可以直接下载RLHF-book-Chinese.pdf文件进行阅读

本手册是 [`rlhf-book`](https://github.com/natolambert/rlhf-book) 的中文翻译版本，旨在为中文社区提供基于人类反馈的强化学习（RLHF）的简明技术指南。原项目由 Nathan Lambert 创作，本翻译版本在保留原内容核心的基础上，优化了中文表达以提升可读性。

---

基于 [**Pandoc 书籍模板**](https://github.com/wikiti/pandoc-book-template) 构建。  

[![代码许可证](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/wikiti/pandoc-book-template/blob/master/LICENSE.md)  
[![内容许可证](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey)](https://github.com/natolambert/rlhf-book/blob/main/LICENSE-Content.md)  


### 引用说明  
建议使用以下格式引用本手册（基于原英文项目信息）：  
```bibtex
@book{rlhf2024,
  author       = {Nathan Lambert},
  title        = {Reinforcement Learning from Human Feedback},
  year         = {2024},
  url          = {https://rlhfbook.com}
}
```

----

## 工具链  
本仓库包含构建 [Pandoc](http://pandoc.org/) 文档的模板，可将 Markdown 文件编译为 PDF、EPUB、HTML 等多种格式。原英文项目的工具链与构建逻辑完整保留，确保翻译版本与原项目在技术实现上的一致性。

## 快速使用  
快捷命令：  
- `make` 生成所有格式文件  
- `make files` 准备图片和 PDF 链接等资源  

### 安装依赖  

#### Linux 系统  
```sh
sudo apt-get install pandoc make texlive-fonts-recommended texlive-xetex
```

#### Mac 系统  
```sh
brew install pandoc make pandoc-crossref
```

### 目录结构  

```
rlhf-book/        # 根目录  
├── build/        # 构建产物目录（存储 PDF、EPUB 等）  
├── chapters/     # 章节 Markdown 文件（中文翻译内容）  
├── images/       # 图片资源（与原项目共享）  
│   └── cover.png # EPUB 封面  
├── metadata.yml  # 元数据配置（适配中文）  
└── Makefile      # 构建脚本
```

### 构建输出  

#### 生成PDF  
```sh
make pdf  # 输出至 build/pdf/
```

#### 生成EPUB  
```sh
make epub  # 输出至 build/epub/
```

#### 生成网页  
```sh
make html  # 输出至 build/html/
```

### 参考资料  
- [原英文项目仓库](https://github.com/natolambert/rlhf-book)  
- [Pandoc 官方文档](http://pandoc.org/MANUAL.html)  
- [Markdown 语法指南](https://www.markdownguide.org/)  

---  
**注**：本翻译版本仅为技术传播目的，内容版权归原作者 Nathan Lambert 所有，遵循原项目的 CC-BY-NC-SA-4.0 许可协议。若有内容疑问或建议，欢迎提交 Issue 或联系翻译团队。