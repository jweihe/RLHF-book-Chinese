import tempfile
import unittest
from pathlib import Path
from scripts.combine_chapters import combine
from scripts.check_html import check


class BookBuildTests(unittest.TestCase):
    def test_navigation_metadata_removed_and_book_links_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / '01-introduction.md'
            second = Path(directory) / '02-related-works.md'
            first.write_text('---\npage-title: First\nnext-url: 02-related-works.html\n---\n# First\n[Next](02-related-works.html)\n\n---\nBody separator\n')
            second.write_text('---\npage-title: Second\n---\n# Second\nText\n')
            result = combine([first, second])
            self.assertNotIn('page-title:', result)
            self.assertIn('[Next](#chapter-02)', result)
            self.assertIn('# Second {#chapter-02}', result)
            self.assertIn('\n---\nBody separator', result)

    def test_missing_front_matter_terminator_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            chapter = Path(directory) / '01-test.md'
            chapter.write_text('---\npage-title: Bad\n# Heading')
            with self.assertRaises(ValueError):
                combine([chapter])

    def test_checker_detects_original_empty_index_and_broken_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'index.html').write_text('<html><body><body><img src="images/missing.png"></body></html>')
            errors = check(root)
            self.assertTrue(any('one body' in e for e in errors))
            self.assertTrue(any('missing local target' in e for e in errors))
            self.assertTrue(any('missing chapter content' in e for e in errors))
