####################################################################################################
# Configuration
####################################################################################################

# Build configuration

SHELL := /bin/bash -o pipefail
.DELETE_ON_ERROR:

BUILD = build
MAKEFILE = Makefile
OUTPUT_FILENAME = book
OUTPUT_FILENAME_HTML = index
METADATA = metadata.yml
CHAPTERS = $(sort $(wildcard chapters/*.md))
NESTED_HTML_DIR = $(BUILD)/html/c
CHAPTER_HTMLS = $(patsubst chapters/%.md,$(NESTED_HTML_DIR)/%.html,$(CHAPTERS))
TOC = --toc --toc-depth 3
METADATA_ARGS = --metadata-file $(METADATA)
IMAGES = $(shell find images -type f)
TEMPLATES = $(shell find templates/ -type f)
COVER_IMAGE = images/cover.png
MATH_FORMULAS = --mathjax # --webtex, is default for PDF/ebook. Consider resetting if issues.
BIBLIOGRAPHY = --bibliography=chapters/bib.bib --citeproc --csl=templates/ieee.csl

# Chapters content
CONTENT = python3 scripts/combine_chapters.py $(CHAPTERS)
CONTENT_FILTERS = tee # Use this to add sed filters or other piped commands

# Debugging

DEBUG_ARGS = --verbose

# Pandoc filtes - uncomment the following variable to enable cross references filter. For more
# information, check the "Cross references" section on the README.md file.

FILTER_ARGS = --filter pandoc-crossref

# Combined arguments

ARGS = $(TOC) $(MATH_FORMULAS) $(METADATA_ARGS) $(FILTER_ARGS) $(DEBUG_ARGS) $(BIBLIOGRAPHY)
	
PANDOC_COMMAND = pandoc

# Per-format options

DOCX_ARGS = --standalone --reference-doc templates/docx.docx
EPUB_ARGS = --template templates/epub.html --epub-cover-image $(COVER_IMAGE) --mathml
HTML_ARGS = --template templates/html.html --standalone --to html5 --listings
# PDF_ARGS = --template templates/pdf.tex --pdf-engine xelatex --listings 
PDF_ARGS = --template templates/pdf.tex --pdf-engine xelatex --listings --variable mainfont="Noto Serif CJK SC" --variable sansfont="Noto Sans CJK SC" --variable monofont="Noto Sans Mono CJK SC"  # 等宽中文字体

NESTED_HTML_TEMPLATE = templates/chapter.html

# Per-format file dependencies

BASE_DEPENDENCIES = $(MAKEFILE) scripts/combine_chapters.py $(CHAPTERS) $(METADATA) $(IMAGES) $(TEMPLATES)
DOCX_DEPENDENCIES = $(BASE_DEPENDENCIES)
EPUB_DEPENDENCIES = $(BASE_DEPENDENCIES)
HTML_DEPENDENCIES = $(BASE_DEPENDENCIES)
PDF_DEPENDENCIES = $(BASE_DEPENDENCIES)

MKDIR_CMD = mkdir -p
RMDIR_CMD = rm -r
ECHO_BUILDING = @echo "building $@..."
ECHO_BUILT = @echo "$@ was built\n"

####################################################################################################
# Basic actions
####################################################################################################

.PHONY: all book clean epub html pdf docx nested_html latex html_assets files check

all:	book

book:	epub html pdf docx

clean:
	$(RMDIR_CMD) $(BUILD)

####################################################################################################
# File builders
####################################################################################################

epub:	$(BUILD)/epub/$(OUTPUT_FILENAME).epub

html:	$(BUILD)/html/$(OUTPUT_FILENAME_HTML).html nested_html

	
pdf:	$(BUILD)/pdf/$(OUTPUT_FILENAME).pdf

docx:	$(BUILD)/docx/$(OUTPUT_FILENAME).docx

latex:	$(BUILD)/latex/$(OUTPUT_FILENAME).tex

$(BUILD)/epub/$(OUTPUT_FILENAME).epub:	$(EPUB_DEPENDENCIES)
	$(ECHO_BUILDING)
	$(MKDIR_CMD) $(BUILD)/epub
	$(CONTENT) | $(CONTENT_FILTERS) | $(PANDOC_COMMAND) $(ARGS) $(EPUB_ARGS) -o $@
	$(ECHO_BUILT)


$(BUILD)/docx/$(OUTPUT_FILENAME).docx:	$(DOCX_DEPENDENCIES)
	$(ECHO_BUILDING)
	$(MKDIR_CMD) $(BUILD)/docx
	$(CONTENT) | $(CONTENT_FILTERS) | $(PANDOC_COMMAND) $(ARGS) $(DOCX_ARGS) -o $@
	$(ECHO_BUILT)
	
# Assets are copied independently so make -j never races chapter generation.
html_assets:
	mkdir -p $(BUILD)/html/images $(NESTED_HTML_DIR)/images
	cp -R images/. $(BUILD)/html/images/
	cp -R images/. $(NESTED_HTML_DIR)/images/
	cp templates/nav.js templates/header-anchors.js favicon.ico $(BUILD)/html/
	cp templates/nav.js templates/header-anchors.js favicon.ico $(NESTED_HTML_DIR)/

$(NESTED_HTML_DIR)/%.html: chapters/%.md $(HTML_DEPENDENCIES)
	mkdir -p $(NESTED_HTML_DIR)
	$(PANDOC_COMMAND) $(ARGS) --template $(NESTED_HTML_TEMPLATE) --standalone --to html5 -o $@ $< --mathjax

nested_html: $(CHAPTER_HTMLS) html_assets

$(BUILD)/html/$(OUTPUT_FILENAME_HTML).html: $(HTML_DEPENDENCIES)
	mkdir -p $(BUILD)/html
	$(CONTENT) | $(CONTENT_FILTERS) | $(PANDOC_COMMAND) $(ARGS) $(HTML_ARGS) -o $@

check:
	python3 -m unittest discover -s tests
	python3 scripts/check_html.py $(BUILD)/html

# ArXiv‑compatible LaTeX build rule
$(BUILD)/latex/$(OUTPUT_FILENAME).tex: $(PDF_DEPENDENCIES)
	$(ECHO_BUILDING)
	$(MKDIR_CMD) -p $(BUILD)/latex

	# 1. Generate the LaTeX file with Pandoc (tell Pandoc where to find images)
	$(CONTENT) \
	  | $(CONTENT_FILTERS) \
	  | $(PANDOC_COMMAND) $(ARGS) $(PDF_ARGS) --resource-path=. -o $@

	# 2. Flatten image paths — copy every referenced image into the build dir root
	$(foreach img,$(IMAGES), cp $(img) $(BUILD)/latex/$(notdir $(img));)

	# 3a. Strip directory prefixes in \includegraphics paths
	sed -E -i.bak 's|(\\includegraphics(\[[^]]*\])?\{)[^/}]+/|\1|g' $@

	# 3b. Restore missing \includegraphics inside \pandocbounded{}
	perl -CSD -pi -e 's/\\pandocbounded\{([^{}]+)\}\}/\\pandocbounded{\\includegraphics{$$1}}/g' $@

	# 3c. Unicode → ASCII/TeX normalisation (one perl pass per rule for clarity)
	perl -CSD -pi -e 's/\x{2060}//g;'                                        $@  # WORD JOINER
	perl -CSD -pi -e 's/\x{03C4}/\\tau/g;'                                  $@  # τ
	perl -CSD -pi -e 's/[\x{2018}\x{2019}]/\x27/g;'                       $@  # curly apostrophes
	perl -CSD -pi -e 's/[\x{201C}\x{201D}]/\x22/g;'                       $@  # curly quotes
	perl -CSD -pi -e 's/\x{2026}/.../g;'                                    $@  # ellipsis
	perl -CSD -pi -e 's/\x{00A9}/\\textcopyright{}/g;'                    $@  # © symbol

	# 4. Copy bibliography and CSL files required by arXiv
	cp chapters/bib.bib    $(BUILD)/latex/
	cp templates/ieee.csl  $(BUILD)/latex/

	# 5. Warn (but don\'t fail) if any non‑ASCII bytes remain
	@REM_BYTES=$$(grep -nP "[\x80-\xFF]" $@ || true); \
	if [ -n "$$REM_BYTES" ]; then \
	  echo "[WARN] Non‑ASCII bytes still present in $@:"; \
	  echo "$$REM_BYTES" | head; \
	else \
	  echo "[INFO] All bytes ASCII‑safe after post‑processing."; \
	fi

	$(ECHO_BUILT)




$(BUILD)/pdf/$(OUTPUT_FILENAME).pdf:	$(PDF_DEPENDENCIES)
	$(ECHO_BUILDING)
	$(MKDIR_CMD) $(BUILD)/pdf
	$(CONTENT) | $(CONTENT_FILTERS) | $(PANDOC_COMMAND) $(ARGS) $(PDF_ARGS) -o $@
	$(ECHO_BUILT)

# Compatibility target for existing publishing commands.
files: html pdf
	cp $(BUILD)/pdf/$(OUTPUT_FILENAME).pdf $(BUILD)/html/book.pdf
