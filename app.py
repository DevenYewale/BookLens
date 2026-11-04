import json
import streamlit as st
import spacy
from analysis import read_book, analyze
import llm

st.set_page_config(page_title='BookLens | Book Analysis', page_icon='ðŸ“–', layout='wide')
st.title('BookLens')
st.caption('An evidence-first NLP reading companion')
st.info('English searchable PDF, EPUB or UTF-8 TXT. Results are statistical suggestions, not verified literary interpretation. Processing runs on the host laptop; no paid AI API. Web uploads pass through Cloudflare. Demo only: do not upload private documents.')

@st.cache_resource
def pipeline():
    nlp = spacy.load('en_core_web_sm', disable=['parser'])
    nlp.add_pipe('sentencizer')
    return nlp

def run(data, name, count):
    pages = read_book(data, name)
    st.session_state['pages'] = pages
    st.session_state.pop('ai', None)
    return analyze(pages, pipeline(), count)

with st.sidebar:
    st.header('Your book')
    upload = st.file_uploader('Choose a book', type=['pdf', 'txt', 'epub'])
    count = st.slider('Summary sentences', 3, 15, 8)
    st.caption('Limits: 25 MB, 2 million characters. No silent truncation. Scans require OCR. PDF page numbers refer to file pages; TXT evidence uses sentence numbers.')
    submit = st.button('Analyze book', type='primary', disabled=upload is None)
    demo = st.button('Try the built-in sample')

sample = '''Alice Morgan lived in London with her brother James Morgan. Alice Morgan worked at the library in London. James Morgan found an old map inside a book. Alice Morgan and James Morgan decided to visit Oxford. In Oxford, Alice Morgan met Professor Helen Carter. Helen Carter explained that the map showed a hidden garden. Alice Morgan searched the garden while James Morgan waited by the gate. A storm forced them to return to London. Helen Carter sent Alice Morgan a letter about the missing key. James Morgan discovered the key inside the original book. Alice Morgan returned to Oxford with James Morgan. They opened the garden gate and found a box of family letters. Alice Morgan donated the letters to the library in London. James Morgan wrote an account of their journey. Their discovery reunited Helen Carter with her family's history.'''
if submit or demo:
    try:
        with st.spinner('Reading the whole text and collecting evidence...'):
            data, name = (sample.encode(), 'sample.txt') if demo else (upload.getvalue(), upload.name)
            result = run(data, name, count)
        st.session_state['analysis'] = (name, result)
    except OSError:
        st.error('English NLP model is missing. Run: python -m spacy download en_core_web_sm')
    except Exception as exc:
        st.error(f'Could not analyze the book: {exc}')

if 'analysis' in st.session_state:
    name, result = st.session_state['analysis']
    st.subheader(name)
    columns = st.columns(4)
    for col, label, value in zip(columns, ['Words', 'Sentences', 'People detected', 'Places detected'],
                                 [result['statistics']['words'], result['statistics']['sentences'], len(result['characters']), len(result['places'])]):
        col.metric(label, value)
    def evidence(item):
        ref = f"PDF page {item['page']}" if name.lower().endswith('.pdf') else f"Sentence {item['sentence']}"
        st.write(item['text'])
        st.caption(ref)
    tabs = st.tabs(['Summary', 'Characters', 'Places', 'Key details', 'NLP Analysis', 'Local AI'])
    with tabs[0]:
        st.caption('Original sentences selected across the book in reading order. Contains spoilers. May omit plot links and keep unresolved pronouns.')
        for item in result['summary']:
            evidence(item)
    for tab, kind in [(tabs[1], 'characters'), (tabs[2], 'places')]:
        with tab:
            st.caption('Ranked by detected mentions, not narrative importance. Only spelling/case variants are grouped; first names and surnames are not guessed to be the same person.')
            if not result[kind]: st.warning('No entities detected in this category. Fictional names can be missed.')
            for entity in result[kind][:40]:
                tag = '' if entity.get('ner_labelled') else ' · no NER label (name or generic referent, unresolved)'
                with st.expander(f"{entity['name']} · {entity['mentions']} mentions{tag}"):
                    if entity.get('possible_same_as'): st.caption('Possibly related to (not merged): ' + ', '.join(entity['possible_same_as']))
                    for item in entity['evidence']: evidence(item)
    with tabs[3]:
        st.caption('Candidate key details ranked by vocabulary salience. These are source sentences, not verified events.')
        for item in result['details']: evidence(item)
        st.subheader('Key phrases')
        st.dataframe(result['keyphrases'], hide_index=True, width='stretch')
    with tabs[4]:
        st.write(result['method'])
        st.json(result['statistics'])
        a,b = st.columns(2)
        with a:
            st.write('Frequent content lemmas'); st.dataframe(result['frequent_lemmas'], column_config={'0':'Lemma','1':'Count'}, hide_index=True)
        with b:
            st.write('Part-of-speech counts'); st.dataframe(result['pos_tags'], column_config={'0':'POS','1':'Count'}, hide_index=True)
        st.caption('Course links: regex text cleaning, tokenization, frequency distributions, type-token ratio, POS tagging, named-entity recognition and bigrams. TF-IDF and extractive ranking extend those techniques. No custom model training claimed.')
    with tabs[5]:
        st.caption('Runs a local Qwen2.5-7B model through Ollama on this laptop. No paid API. Uses only the uploaded text. Slow on a full book (many minutes). Output is a model suggestion and can be wrong.')
        if not llm.available():
            st.warning('Local model server is not running (Ollama on port 11434).')
        else:
            import time, jobs
            import hashlib as _hl; _h = _hl.sha256(); _ = [_h.update(str(_p).encode('utf-8','ignore') + b'\x00') for _p in st.session_state['pages']]; fp = _h.hexdigest()
            tok = st.session_state.get('job_token') or st.query_params.get('job')
            job = jobs.get_owned(tok, fp)
            if st.button('Run local AI summary, roles and place timeline') and not (job and job['state'] == 'running'):
                tok = jobs.start(fp, st.session_state['pages'], result['characters'], result['places'])
                st.session_state['job_token'] = tok
                st.query_params['job'] = tok
                job = jobs.get_owned(tok, fp)
            if job and job['state'] == 'running':
                st.progress(min(job['frac'], 1.0), text=job['msg'])
                st.caption('Runs in the background on the laptop and continues if this page disconnects. Keep this page address: reopening it with the same book shows the progress or result. Anyone with the full address can see it, kept in memory for 2 hours.')
                time.sleep(3); st.rerun()
            elif job and job['state'] == 'error':
                st.error('Local AI failed: ' + job['msg'])
            elif job and job['state'] == 'done':
                st.session_state['ai'] = job['ai']
        ai = st.session_state.get('ai')
        if ai:
            st.subheader('Summary')
            st.write(ai['summary']['summary'])
            st.caption('Built from %d section summaries. Check against the Summary tab evidence.' % ai['summary']['sections'])
            st.download_button('Download Local AI result (JSON)', __import__('json').dumps(ai, ensure_ascii=False), 'local_ai_result.json', key='dl_ai')
            st.subheader('Character roles (only where a cited sentence supports it)')
            st.caption('The model must cite one sentence. Without one, the role is left blank rather than guessed.')
            st.dataframe([{'name': r['name'], 'mentions': r['mentions'], 'role': r['role'] or '-', 'cited sentence': r['evidence'] or '', 'status': r['status']} for r in ai['roles']], hide_index=True)
            st.subheader('Story locations by section (sourced)')
            st.caption('Per section of the text, the place where the model says the main action happens. Kept only if the quoted sentence is in that section and names the place. Sections with no clear place are omitted.')
            for row in ai.get('scenes', []):
                st.write('Section %d of %d: %s' % (row['section'], row['of'], row['place']))
                st.caption(row['quote'])
            st.subheader('Place mentions across the text (not a story travel timeline)')
            st.caption('This lists detected place names by where each first appears. A mention does not mean a character was there or that events happened there in this order. Bins count mentions across ten equal slices of the book.')
            for row in ai['timeline']:
                with st.expander('%s - first at sentence %d' % (row['name'], row['first_sentence'])):
                    st.write(row['bins'])
                    if row['evidence']: evidence(row['evidence'])
    st.download_button('Download analysis JSON', json.dumps(result, indent=2, ensure_ascii=False), file_name='book_analysis.json', mime='application/json')
else:
    st.subheader('From book to evidence')
    st.write('Upload a book or try the sample. Explore detected people and places, inspect source sentences, and see how the NLP pipeline works.')
