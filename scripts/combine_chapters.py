"""Combine chapters without leaking per-page navigation into book metadata."""
import re
import sys
from pathlib import Path


def combine(paths):
    chapters = []
    links = {p.stem + '.html': '#chapter-' + p.stem.split('-')[0] for p in paths}
    for path in paths:
        text = path.read_text(encoding='utf-8')
        if text.startswith('---\n'):
            parts = text.split('\n---\n', 1)
            if len(parts) != 2:
                raise ValueError(f'Unclosed front matter: {path}')
            text = parts[1]
        identifier = 'chapter-' + path.stem.split('-')[0]
        text = re.sub(r'^# (.+)$', lambda m: '# ' + m[1] + ' {#' + identifier + '}', text, count=1, flags=re.M)
        for target, anchor in links.items():
            text = text.replace('](' + target + ')', '](' + anchor + ')')
        chapters.append(text.strip())
    return '\n\n'.join(chapters) + '\n'


if __name__ == '__main__':
    print(combine([Path(p) for p in sys.argv[1:]]), end='')
