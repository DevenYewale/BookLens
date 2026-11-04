import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spacy
from analysis import read_book, analyze, clean_page, entity_key

class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp=spacy.load('en_core_web_sm',disable=['parser']);cls.nlp.add_pipe('sentencizer')
    def test_cleaning(self):
        self.assertEqual(clean_page('a long-\nwinded  book'), 'a longwinded book')
        self.assertEqual(entity_key("Alice's"), 'alice')
    def test_reject_empty(self):
        with self.assertRaises(ValueError):read_book(b'','x.txt')
    def test_reject_encoding(self):
        with self.assertRaises(ValueError):read_book(b'\xff'*30,'x.txt')
    def test_reject_type(self):
        with self.assertRaises(ValueError):read_book(b'test','x.exe')
    def test_pipeline_evidence_and_end(self):
        text=('Alice Morgan lived in London. James Morgan visited Alice Morgan in London. '
              'James Morgan borrowed an ancient book from the library. '
              'Alice Morgan found a letter inside the book. '
              'At last, the friends returned the book and saved the library.')
        r=analyze(read_book(text.encode(),'x.txt'),self.nlp,3)
        self.assertIn('Alice Morgan',[x['name'] for x in r['characters']])
        self.assertIn('London',[x['name'] for x in r['places']])
        self.assertTrue(r['summary'][-1]['text'].startswith('At last'))
        self.assertEqual(r['statistics']['sentences'],5)
        for kind in ['summary','details']:
            for item in r[kind]: self.assertIn(item['text'],text)
    def test_page_provenance(self):
        r=analyze(['Alice Morgan went to London to borrow a book.', 'James Morgan stayed in Paris to read another book.'],self.nlp,3)
        self.assertEqual({s['page'] for s in r['summary']},{1,2})
    def test_long_word_not_silently_lost(self):
        with self.assertRaises(ValueError):analyze(['x'*40001],self.nlp)

if __name__=='__main__':unittest.main()
