"""Local LLM layer for BookLens. Talks only to a local Ollama server (no paid API)."""
import json, re, urllib.request

URL = 'http://localhost:11434/api/generate'
MODEL = 'qwen2.5:7b-instruct-q4_K_M'
ROLES = ['protagonist', 'antagonist', 'ally or mentor', 'family or love interest', 'minor or background', 'uncertain']


def ask(prompt, num_predict=300, timeout=600, fmt=None):
    body = {'model': MODEL, 'prompt': prompt, 'stream': False,
            'options': {'num_ctx': 8192, 'num_predict': num_predict, 'temperature': 0.2}}
    if fmt:
        body['format'] = fmt
    req = urllib.request.Request(URL, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())['response'].strip()


def available():
    try:
        urllib.request.urlopen('http://localhost:11434/api/tags', timeout=3).read()
        return True
    except Exception:
        return False


def make_chunks(pages, limit=9000):
    chunks, buf = [], ''
    for page in pages:
        text = re.sub(r'\s+', ' ', page).strip()
        while len(text) > limit:
            cut = text.rfind('. ', 0, limit) + 1 or limit
            if buf:
                chunks.append(buf); buf = ''
            chunks.append(text[:cut].strip()); text = text[cut:].strip()
        if text:
            if len(buf) + len(text) > limit and buf:
                chunks.append(buf); buf = ''
            buf = (buf + ' ' + text).strip()
    if buf:
        chunks.append(buf)
    return chunks


def summarize_book(pages, progress=None, ask_fn=ask):
    chunks = make_chunks(pages)
    parts = []
    for i, c in enumerate(chunks, 1):
        if progress: progress(i, len(chunks) + 1, 'Summarising section %d of %d' % (i, len(chunks)))
        parts.append(ask_fn('Summarise the plot events in this section of a book in 2-3 sentences. Use only the text below. '
                            'Write names exactly as they appear. Do not add outside knowledge.\n\nTEXT:\n' + c, 160))
    level = parts
    while len(level) > 1 and sum(map(len, level)) > 9000:
        nxt = []
        for j in range(0, len(level), 8):
            nxt.append(ask_fn('Merge these ordered section summaries into one shorter summary of 4-5 sentences. Keep the order of events.\n\n' + '\n'.join(level[j:j + 8]), 260))
        level = nxt
    if progress: progress(len(chunks) + 1, len(chunks) + 1, 'Writing final summary')
    final = ask_fn('Using only these ordered section summaries, write a concise summary of the whole book in about 200 words: '
                   'the main plot points and events in order, then 2-3 major themes. Do not invent facts.\n\n' + '\n'.join(level), 450)
    return {'summary': final, 'sections': len(chunks), 'section_summaries': parts}


def _norm(t):
    return re.sub(r'\s+', ' ', t).strip().casefold()


def entails(claim, sentence, ask_fn=ask):
    """Second pass: the sentence alone must clearly state the claim. Anything but a plain yes counts as no."""
    try:
        r = ask_fn('Sentence: %s\nClaim: %s\nDoes the sentence by itself clearly state the claim? Answer only yes or no.' % (sentence, claim), 5)
    except Exception:
        return False
    return r.strip().lower().startswith('yes')


def _own_mention(name, sentence):
    """True if the name occurs as itself, not only inside a titled form (Mrs. X) or a possessive (X's)."""
    for m in re.finditer(r'\b' + re.escape(name) + r'\b', sentence):
        before, after = sentence[:m.start()], sentence[m.end():]
        if re.search(r'\b(Mr|Mrs|Miss|Ms|Dr|Master)\.?\s*$', before) or re.match(r"\s*['\u2019]s\b", after):
            continue
        return True
    return False


def classify_roles(entities, ask_fn=ask, limit=8):
    """Role suggestions must cite one of the numbered sentences shown to the model; otherwise abstain."""
    out = []
    for e in entities[:limit]:
        sents = [s['text'][:300] for s in e.get('spread', e['evidence'])]
        row = {'name': e['name'], 'mentions': e['mentions'], 'role': None, 'evidence': None,
               'status': 'abstained: no supporting sentence'}
        if sents:
            ev = '\n'.join('%d. %s' % (k + 1, t) for k, t in enumerate(sents))
            prompt = ('Name: %s\nNumbered sentences from the book:\n%s\n\nDoes one sentence clearly show what role this name plays in the story '
                      '(for example: central figure, helper, opponent, family member, minor figure)? Reply as JSON: '
                      '{"role": "short phrase or null", "sentence": number or null}. Use null if no sentence clearly shows it.' % (e['name'], ev))
            try:
                data = json.loads(ask_fn(prompt, 80, fmt='json'))
                k = data.get('sentence')
                if data.get('role') and isinstance(k, int) and 1 <= k <= len(sents) and _own_mention(e['name'], sents[k - 1]):
                    role = str(data['role'])[:60]
                    if entails('%s is %s in the story.' % (e['name'], role), sents[k - 1], ask_fn):
                        row.update(role=role, evidence=sents[k - 1], status='suggested; cited sentence checked by a second pass')
                    else:
                        row['status'] = 'abstained: cited sentence does not clearly state the role'
            except Exception:
                row['status'] = 'abstained: model reply unusable'
        out.append(row)
    return out


def scene_chronology(pages, progress=None, ask_fn=ask):
    """Per-section location of the main action. A row is kept only if the quote and place are both in that section's text."""
    chunks = make_chunks(pages)
    rows = []
    for i, c in enumerate(chunks, 1):
        if progress: progress(i, len(chunks), 'Locating scenes in section %d of %d' % (i, len(chunks)))
        try:
            data = json.loads(ask_fn('Where does the main action of this section take place? Reply as JSON: {"place": "name of the place as written in the text, or null", '
                                     '"quote": "one sentence copied exactly from the text that shows the action at that place, or null"}. Use null if the text does not say.\n\nTEXT:\n' + c, 120, fmt='json'))
        except Exception:
            continue
        place, quote = data.get('place'), data.get('quote')
        if isinstance(place, str) and isinstance(quote, str) and _norm(quote) in _norm(c) and _norm(place) in _norm(quote) \
                and entails('The events in this sentence take place at or in %s.' % place, quote, ask_fn):
            rows.append({'section': i, 'of': len(chunks), 'place': place, 'quote': quote})
    return rows


def place_timeline(places, limit=12):
    rows = [p for p in places[:limit * 2] if p.get('first_sentence')]
    rows.sort(key=lambda p: p['first_sentence'])
    return [{'name': p['name'], 'first_sentence': p['first_sentence'], 'bins': p.get('bins', []),
             'evidence': p['evidence'][0] if p['evidence'] else None} for p in rows[:limit]]
