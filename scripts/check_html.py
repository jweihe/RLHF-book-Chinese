"""Check generated pages for empty content, missing local assets and links."""
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links = []
        self.ids = set()
        self.bodies = 0
        self.content = False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.bodies += tag == 'body'
        if attrs.get('id'):
            self.ids.add(attrs['id'])
        if tag in ('div', 'main') and attrs.get('id') == 'content':
            self.content = True
        for name in ('href', 'src'):
            if attrs.get(name):
                self.links.append(attrs[name])


def check(root):
    pages = {p.resolve(): Page(p.read_text(encoding='utf-8')) for p in root.rglob('*.html')}
    errors = []
    if len(pages) != 20:
        errors.append(f'Expected index + 19 chapters, found {len(pages)} pages')
    for path, page in pages.items():
        if page.bodies != 1 or not page.content:
            errors.append(f'{path.name}: expected one body and a content container')
        for url in page.links:
            parts = urlsplit(url)
            if parts.scheme or parts.netloc:
                continue
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if not target.exists():
                errors.append(f'{path.name}: missing local target {url}')
            elif parts.fragment and target in pages and unquote(parts.fragment) not in pages[target].ids:
                errors.append(f'{path.name}: missing anchor {url}')
    index = pages.get((root / 'index.html').resolve())
    if not index or not all('chapter-' + str(i).zfill(2) in index.ids for i in range(1, 20)):
        errors.append('Index is missing chapter content')
    return errors


if __name__ == '__main__':
    errors = check(Path(sys.argv[1] if len(sys.argv) > 1 else 'build/html'))
    if errors:
        sys.exit('\n'.join(errors))
    print('Validated 20 HTML pages: content, local assets, links and anchors.')
