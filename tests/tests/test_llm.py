import unittest, llm, analysis

def test_chunks_bounded():
    pages = ['Word. ' * 4000, 'Another page. ' * 200]
    ch = llm.make_chunks(pages, 3000)
    assert all(len(c) <= 3200 for c in ch) and len(ch) > 3

def test_summarize_and_roles():
    calls = []
    def fake(prompt, n, **k):
        calls.append(prompt); return '{"role": "protagonist", "reason": "x"}' if 'JSON' in prompt else 'S.'
    r = llm.summarize_book(['A b c. ' * 3000, 'D e f. ' * 3000], ask_fn=fake)
    assert r['summary'] == 'S.' and r['sections'] >= 2
    bad = llm.classify_roles([{'name': 'X', 'mentions': 1, 'evidence': []}], ask_fn=lambda *a, **k: 'nonsense')
    assert bad[0]['role'] is None

def test_timeline_order():
    t = llm.place_timeline([{'name': 'B', 'first_sentence': 9, 'evidence': []}, {'name': 'A', 'first_sentence': 2, 'evidence': []}])
    assert [x['name'] for x in t] == ['A', 'B']

def test_roles_abstain_without_cited_sentence():
    ent = [{'name': 'Darrow', 'mentions': 3, 'evidence': [{'text': 'Darrow ran.'}]}]
    assert llm.classify_roles(ent, ask_fn=lambda *a, **k: '{"role": "hero", "sentence": 5}')[0]['role'] is None
    assert llm.classify_roles(ent, ask_fn=lambda *a, **k: '{"role": null, "sentence": null}')[0]['role'] is None
    def fake(p, n, **k):
        return 'yes' if p.startswith('Sentence:') else '{"role": "hero", "sentence": 1}'
    ok = llm.classify_roles(ent, ask_fn=fake)[0]
    assert ok['role'] == 'hero' and ok['evidence'] == 'Darrow ran.'
    def deny(p, n, **k):
        return 'no' if p.startswith('Sentence:') else '{"role": "hero", "sentence": 1}'
    assert llm.classify_roles(ent, ask_fn=deny)[0]['role'] is None

def test_scene_quote_must_be_in_text():
    pages = ['They arrived. In Oxford the group met at the gate.']
    good = lambda p, n, **k: 'yes' if p.startswith('Sentence:') else '{"place": "Oxford", "quote": "In Oxford the group met at the gate."}'
    nope = lambda p, n, **k: 'no' if p.startswith('Sentence:') else '{"place": "Oxford", "quote": "In Oxford the group met at the gate."}'
    assert llm.scene_chronology(pages, ask_fn=nope) == []
    bad = lambda *a, **k: '{"place": "Paris", "quote": "They went to Paris."}'
    assert llm.scene_chronology(pages, ask_fn=good)[0]['place'] == 'Oxford'
    assert llm.scene_chronology(pages, ask_fn=bad) == []


class BoilerplateAndAliasTests(unittest.TestCase):
    def test_mask_keeps_offsets(self):
        t = 'Header Gutenberg\n*** START OF THE BOOK ***\nBody text.\n*** END OF THE BOOK ***\nLicense Gutenberg'
        out = analysis.mask_boilerplate([t])[0]
        self.assertEqual(len(out), len(t))
        self.assertNotIn('Gutenberg', out)
        self.assertIn('Body text.', out)

    def test_no_markers_untouched(self):
        self.assertEqual(analysis.mask_boilerplate(['plain']), ['plain'])

    def test_titled_possessive_not_own_mention(self):
        self.assertFalse(llm._own_mention('Cratchit', "Mrs. Cratchit, Cratchit's wife"))
        self.assertTrue(llm._own_mention('Cratchit', 'Cratchit came home.'))
