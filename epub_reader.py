"""Bounded, offline EPUB text reader. Never extracts files or fetches resources."""
from html.parser import HTMLParser
from io import BytesIO
import posixpath
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET
from zipfile import ZipFile, BadZipFile

class TextReader(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0
    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style', 'nav'}:
            self.hidden += 1
        if tag in {'p','div','br','h1','h2','h3','li','section'}:
            self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag in {'script', 'style', 'nav'} and self.hidden:
            self.hidden -= 1
        if tag in {'p','div','h1','h2','h3','li','section'}:
            self.parts.append('\n')
    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

def read_epub(data, max_chars):
    try:
        with ZipFile(BytesIO(data)) as z:
            items = z.infolist()
            if len(items) > 10000 or sum(i.file_size for i in items) > 64*1024*1024:
                raise ValueError('EPUB archive is too large when expanded (64 MB / 10,000 entries maximum).')
            names = z.namelist()
            if len(set(names)) != len(names):
                raise ValueError('EPUB has duplicate archive entries.')
            if any(i.flag_bits & 1 for i in items) or 'META-INF/encryption.xml' in names:
                raise ValueError('Encrypted or DRM-protected EPUBs are not supported. Use an unencrypted copy.')
            def bounded(name, limit):
                info = z.getinfo(name)
                if info.file_size > limit:
                    raise ValueError('EPUB contains an oversized chapter or metadata file.')
                return z.read(info)
            def xml(name):
                raw = bounded(name, 1024*1024)
                if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
                    raise ValueError('EPUB metadata contains unsupported XML declarations.')
                return ET.fromstring(raw)
            container = xml('META-INF/container.xml')
            roots = [e for e in container.iter() if e.tag.split('}')[-1] == 'rootfile']
            root = next((e for e in roots if e.get('media-type') == 'application/oebps-package+xml'), None)
            if root is None:
                raise ValueError('EPUB has no valid package document.')
            opf = root.get('full-path', '')
            if not opf or opf.startswith('/') or '..' in opf.split('/'):
                raise ValueError('EPUB package path is invalid.')
            package = xml(opf)
            manifest = {e.get('id'): e for e in package.iter() if e.tag.split('}')[-1] == 'item'}
            spine = [e for e in package.iter() if e.tag.split('}')[-1] == 'itemref' and e.get('linear', 'yes') != 'no']
            pages, total = [], 0
            for ref in spine:
                item = manifest.get(ref.get('idref'))
                if item is None:
                    raise ValueError('EPUB spine refers to a missing chapter.')
                if item.get('media-type') not in {'application/xhtml+xml', 'text/html'}:
                    raise ValueError('EPUB contains an unsupported non-text reading-order item.')
                href = urlsplit(item.get('href', ''))
                path = posixpath.normpath(posixpath.join(posixpath.dirname(opf), unquote(href.path)))
                if href.scheme or href.netloc or path.startswith(('/', '../')) or '\\' in path:
                    raise ValueError('EPUB chapter path is invalid.')
                parser = TextReader()
                parser.feed(bounded(path, 8*1024*1024).decode('utf-8-sig'))
                parser.close()
                text = ''.join(parser.parts).strip()
                total += len(text)
                if total > max_chars:
                    raise ValueError('Book exceeds 2 million characters. Split it into volumes; nothing was truncated.')
                pages.append(text)
            if not pages:
                raise ValueError('EPUB has no readable chapters in its reading order.')
            return pages
    except (BadZipFile, KeyError, ET.ParseError, UnicodeDecodeError, RuntimeError, NotImplementedError) as exc:
        raise ValueError('Could not read this EPUB. Use a valid, unencrypted UTF-8 EPUB with text chapters.') from exc
