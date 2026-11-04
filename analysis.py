"""Evidence-first book NLP. No network calls or generative API."""
from collections import Counter, defaultdict
from io import BytesIO
import re
import numpy as np
from pypdf import PdfReader
from epub_reader import read_epub
from sklearn.feature_extraction.text import TfidfVectorizer

MAX_CHARS = 2_000_000


_PG_START = re.compile(r'\*\*\* ?START OF[^\n]*', re.I)
_PG_END = re.compile(r'\*\*\* ?END OF[^\n]*', re.I)


def mask_boilerplate(pages):
    """If the text carries explicit START/END markers, blank everything outside them with spaces (same length, so offsets stay valid)."""
    out = []
    for p in pages:
        s, e = _PG_START.search(p), _PG_END.search(p)
        if s and e and s.end() < e.start():
            blank = lambda t: re.sub(r'[^\n]', ' ', t)
            p = blank(p[:s.end()]) + p[s.end():e.start()] + blank(p[e.start():])
        out.append(p)
    return out


def read_book(data, filename):
    if len(data) > 25 * 1024 * 1024:
        raise ValueError('Maximum upload size is 25 MB.')
    if filename.lower().endswith('.pdf'):
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise ValueError('Password-protected PDFs are not supported. Upload an unlocked copy.')
        pages = [p.extract_text() or '' for p in reader.pages]
    elif filename.lower().endswith('.epub'):
        pages = read_epub(data, MAX_CHARS)
    elif filename.lower().endswith('.txt'):
        try:
            pages = [data.decode('utf-8-sig')]
        except UnicodeDecodeError as exc:
            raise ValueError('Please save the text file as UTF-8.') from exc
    else:
        raise ValueError('Choose a searchable PDF, an EPUB or a UTF-8 TXT file.')
    if sum(map(len, pages)) > MAX_CHARS:
        raise ValueError('Book exceeds 2 million characters. Split it into volumes; nothing was truncated.')
    if sum(len(re.findall(r'\b\w+\b', p)) for p in pages) < 20:
        raise ValueError('Not enough readable text. Scanned PDFs need OCR first.')
    return mask_boilerplate(pages)


def clean_page(text):
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    return re.sub(r'\s+', ' ', text).strip()


def entity_key(name):
    name = re.sub(r"['â€™]s$", '', name.strip())
    return re.sub(r'\s+', ' ', name).casefold()


def summarize(sentences, matrix, count):
    """Coverage by position bins, then salience with redundancy penalty.

    Returns original sentences in reading order, never generated prose.
    Sparse TF-IDF avoids an all-pairs sentence similarity matrix.
    """
    n = len(sentences)
    count = min(max(int(count), 1), n)
    centroid = np.asarray(matrix.mean(axis=0)).ravel()
    scores = np.asarray(matrix @ centroid).ravel()
    eligible = [i for i, s in enumerate(sentences) if 5 <= len(s['text'].split()) <= 100]
    if not eligible:
        eligible = list(range(n))
    picked = []
    for b in range(count):
        candidates = [i for i in eligible if n*b//count <= i < n*(b+1)//count]
        if not candidates:
            continue
        def rank(i):
            similarity = max((float((matrix[i] @ matrix[j].T).toarray()[0, 0]) for j in picked), default=0)
            return float(scores[i]) - 0.35 * similarity
        chosen = candidates[-1] if b == count - 1 else max(candidates, key=rank)
        if sentences[chosen]['text'] not in {sentences[j]['text'] for j in picked}:
            picked.append(chosen)
    return [sentences[i] for i in sorted(picked)], scores


LOCATIVE = {'in', 'at', 'to', 'from', 'into', 'near', 'toward', 'towards', 'across', 'through', 'inside',
            'outside', 'around', 'beyond', 'over', 'under', 'onto', 'upon', 'within', 'throughout', 'reach', 'reached'}
TITLES = {'mr', 'mrs', 'ms', 'dr', 'lord', 'lady', 'sir', 'captain', 'professor', 'king', 'queen', 'master', 'sergeant', 'general'}
NAME_LABELS = {'PERSON', 'GPE', 'LOC', 'FAC', 'ORG', 'NORP', 'DATE', 'PRODUCT', 'WORK_OF_ART', 'EVENT'}
NOT_NAMES = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
             'november', 'december', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}


def entity_kind_hint(ent):
    """Candidate filter. spaCy labels are only a first guess; classify_entity decides."""
    if ent.label_ not in NAME_LABELS:
        return None
    words = ent.text.split()
    if len(words) > 3 or not all(w[:1].isupper() for w in words):
        return None
    if ent.text.casefold() in NOT_NAMES or not any(c.isalpha() for c in ent.text):
        return None
    return 'candidate'


def classify_entity(st):
    """Decide character vs place from evidence in the text itself.

    A place is mostly introduced by locative words (in/at/to/from...). A character is
    tagged PERSON, acts as the subject (followed by a verb) or has a title. Names that
    show neither pattern are dropped instead of guessed.
    """
    n = st['n']
    person_votes = st['PERSON'] + st['act']
    place_votes = st['loc']
    place_label = st['GPE'] + st['LOC'] + st['FAC']
    if st['DATE'] * 2 >= n and st['DATE'] > st['PERSON'] + place_label:
        return None  # a date/time expression, not a person or place
    if place_votes >= 2 and place_votes / n >= 0.25 and place_votes >= person_votes * 0.6:
        return 'places'
    if place_votes >= 1 and place_label * 2 >= n and place_votes >= person_votes:
        return 'places'
    if st['PERSON'] * 2 >= n or (person_votes >= 2 and person_votes > place_votes):
        return 'characters'
    return None


def analyze(pages, nlp, summary_count=8):
    sentences, mentions = [], defaultdict(list)
    stats = defaultdict(Counter)
    variants = defaultdict(Counter)
    words, pos = Counter(), Counter()
    # Batch by page and bounded text chunk, preserving page provenance.
    chunks = []
    for page_no, page in enumerate(pages, 1):
        cleaned = clean_page(page)
        # Never split a word, and avoid max_length failures on single TXT pages.
        buffer = []
        length = 0
        for token in re.findall(r'\S+\s*', cleaned):
            if length + len(token) > 40000 and buffer:
                chunks.append((''.join(buffer), {'page': page_no}))
                buffer, length = [], 0
            if len(token) > 40000:
                raise ValueError('Text contains an unusually long token. Check PDF extraction or input encoding.')
            buffer.append(token); length += len(token)
        if buffer:
            chunks.append((''.join(buffer), {'page': page_no}))
    for doc, ctx in nlp.pipe(chunks, as_tuples=True, batch_size=8):
        for sent in doc.sents:
            text = sent.text.strip()
            if not text:
                continue
            idx = len(sentences)
            sentences.append({'text': text, 'page': ctx['page'], 'sentence': idx + 1})
            for ent in sent.ents:
                kind = entity_kind_hint(ent)
                if kind:
                    key = entity_key(ent.text)
                    variants[key][ent.text] += 1
                    mentions[key].append(idx)
                    st = stats[key]
                    st['n'] += 1
                    st[ent.label_] += 1
                    prev = doc[ent.start - 1].lower_ if ent.start > 0 else ''
                    nxt = doc[ent.end] if ent.end < len(doc) else None
                    if prev in LOCATIVE and ent.start > 0 and doc[ent.start - 1].text.islower():
                        st['loc'] += 1
                    if nxt is not None and (nxt.pos_ in {'VERB', 'AUX'} or nxt.lower_ in {'and', 'said'}) and prev not in LOCATIVE:
                        st['act'] += 1
                    if prev in TITLES or (nxt is not None and nxt.lower_ in {'s', "'s", '\u2019s'} and prev not in LOCATIVE):
                        st['act'] += 1
            covered = {t.i for e in sent.ents for t in e}
            for tok in sent:
                if tok.i in covered or tok.pos_ != 'PROPN' or not tok.is_alpha or len(tok.text) < 3 or not tok.text[:1].isupper():
                    continue
                if tok.lower_ in NOT_NAMES or tok.is_stop:
                    continue
                key = entity_key(tok.text)
                variants[key][tok.text] += 1
                mentions[key].append(idx)
                st = stats[key]
                st['n'] += 1
                st['PROPN'] += 1
                prev = doc[tok.i - 1].lower_ if tok.i > 0 else ''
                nxt = doc[tok.i + 1] if tok.i + 1 < len(doc) else None
                if prev in LOCATIVE and doc[tok.i - 1].text.islower():
                    st['loc'] += 1
                if nxt is not None and nxt.pos_ in {'VERB', 'AUX'} and prev not in LOCATIVE:
                    st['act'] += 1
                if prev in TITLES:
                    st['act'] += 1
        words.update(t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop)
        pos.update(t.pos_ for t in doc if t.is_alpha)
    if not sentences:
        raise ValueError('No sentences found in the text.')
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=12000, sublinear_tf=True)
        matrix = vectorizer.fit_transform(s['text'] for s in sentences)
    except ValueError as exc:
        raise ValueError('Not enough English vocabulary for TF-IDF analysis.') from exc
    summary, scores = summarize(sentences, matrix, summary_count)
    entities = {'characters': [], 'places': []}
    for key, indices in mentions.items():
        kind = classify_entity(stats[key])
        if kind is None:
            continue
        unique_indices = list(dict.fromkeys(indices))
        entities[kind].append({'name': variants[key].most_common(1)[0][0], 'mentions': len(indices),
                               'label_votes': dict(stats[key]),
                               'forms': list(variants[key]),
                               'first_sentence': unique_indices[0] + 1,
                               'ner_labelled': (stats[key]['PERSON'] + stats[key]['GPE'] + stats[key]['LOC'] + stats[key]['FAC']) * 2 >= stats[key]['n'],
                               'bins': [sum(1 for i in indices if i * 10 // len(sentences) == b) for b in range(10)],
                               'spread': [sentences[unique_indices[k * len(unique_indices) // 8]] for k in range(min(8, len(unique_indices)))],
                               'evidence': [sentences[i] for i in unique_indices[:3]]})
    for values in entities.values():
        values.sort(key=lambda x: (-x['mentions'], x['name']))
        # Possible aliases are only flagged for the reader, never merged.
        for v in values:
            toks = set(v['name'].casefold().split())
            v['possible_same_as'] = [o['name'] for o in values if o is not v and toks < set(o['name'].casefold().split())]
    means = np.asarray(matrix.mean(axis=0)).ravel()
    features = vectorizer.get_feature_names_out()
    terms = [{'phrase': features[i], 'score': round(float(means[i]), 5)} for i in means.argsort()[-20:][::-1]]
    detail_indices = sorted(range(len(sentences)), key=lambda i: float(scores[i]), reverse=True)
    details, seen = [], set()
    for i in detail_indices:
        if sentences[i]['text'] not in seen and 5 <= len(sentences[i]['text'].split()) <= 100:
            details.append(sentences[i]); seen.add(sentences[i]['text'])
        if len(details) == 10:
            break
    tokens = [w.casefold() for p in pages for w in re.findall(r"\b[^\W\d_]+\b", p)]
    return {**entities, 'summary': summary, 'details': details, 'keyphrases': terms,
            'statistics': {'words': len(tokens), 'unique_words': len(set(tokens)), 'sentences': len(sentences),
                           'pages': len(pages), 'type_token_ratio': round(len(set(tokens))/max(len(tokens),1), 4)},
            'frequent_lemmas': words.most_common(20), 'pos_tags': pos.most_common(),
            'method': 'spaCy entity candidates reclassified by context (locative words, verb-subject use, titles); TF-IDF unigrams/bigrams; position-stratified extractive summary'}
