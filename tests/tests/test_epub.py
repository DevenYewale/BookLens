import unittest
from io import BytesIO
from zipfile import ZipFile
from epub_reader import read_epub
from analysis import read_book

def fixture(encrypted=False, href='c1.xhtml'):
    b=BytesIO()
    with ZipFile(b,'w') as z:
        z.writestr('mimetype','application/epub+zip')
        z.writestr('META-INF/container.xml','<container><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
        z.writestr('OEBPS/content.opf',f'<package><manifest><item id="a" href="{href}" media-type="application/xhtml+xml"/><item id="b" href="c2.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="b"/><itemref idref="a"/></spine></package>')
        z.writestr('OEBPS/c1.xhtml','<html><body><p>Alice Morgan returned to London after finding the missing letter in Oxford.</p><script>secret script</script></body></html>')
        z.writestr('OEBPS/c2.xhtml','<html><body><h1>First chapter</h1><p>James Morgan visited Paris and borrowed an ancient book from the town library.</p></body></html>')
        if encrypted:z.writestr('META-INF/encryption.xml','<encryption/>')
    return b.getvalue()

class EpubTests(unittest.TestCase):
    def test_reading_order_and_text(self):
        p=read_book(fixture(),'sample.epub')
        self.assertIn('James Morgan',p[0]);self.assertIn('Alice Morgan',p[1]);self.assertNotIn('secret script',p[1])
    def test_drm(self):
        with self.assertRaisesRegex(ValueError,'DRM'):read_epub(fixture(True),2000000)
    def test_corrupt(self):
        with self.assertRaises(ValueError):read_epub(b'not zip',2000000)
    def test_missing_metadata(self):
        b=BytesIO()
        with ZipFile(b,'w') as z:z.writestr('x','y')
        with self.assertRaises(ValueError):read_epub(b.getvalue(),2000000)
    def test_external_chapter(self):
        with self.assertRaisesRegex(ValueError,'path'):read_epub(fixture(href='https://example.com/c1'),2000000)
    def test_limit(self):
        with self.assertRaisesRegex(ValueError,'million'):read_epub(fixture(),10)
if __name__=='__main__':unittest.main()
