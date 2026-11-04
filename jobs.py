import hashlib, secrets, threading, time, traceback
import llm
_J = {}
_L = threading.Lock()

def key(name, pages):
    h = hashlib.sha256()
    h.update(str(len(pages)).encode())
    for p in pages:
        h.update(str(p).encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()

def get(k):
    with _L:
        return dict(_J[k]) if k in _J else None

def _set(k, **kw):
    with _L:
        _J[k].update(kw, updated=time.time())

def purge():
    with _L:
        for t in [t for t, v in _J.items() if time.time() - v['updated'] > 7200]:
            del _J[t]

def get_owned(token, fp):
    j = get(token) if token else None
    return j if j and j.get('fp') == fp else None

def start(fp, pages, chars, places):
    purge()
    k = secrets.token_urlsafe(16)
    with _L:
        _J[k] = {'fp': fp, 'state': 'running', 'msg': 'Starting', 'frac': 0.0, 'started': time.time(), 'updated': time.time()}
    def work():
        try:
            def prog(i, n, msg): _set(k, frac=i / n, msg=msg)
            summ = llm.summarize_book(pages, prog)
            _set(k, msg='Roles', frac=1.0)
            roles = llm.classify_roles(chars)
            scenes = llm.scene_chronology(pages, prog)
            ai = {'summary': summ, 'roles': roles, 'scenes': scenes, 'timeline': llm.place_timeline(places)}
            _set(k, state='done', ai=ai, msg='Done')
        except Exception:
            _set(k, state='error', msg=traceback.format_exc()[-1500:])
    threading.Thread(target=work, daemon=True).start()
    return k
