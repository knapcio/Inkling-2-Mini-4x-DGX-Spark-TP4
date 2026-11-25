# Usage (any Python 3): python3 tools/profile_eager.py <profile dir or trace.json.gz>
"""List the non-graph (eager) GPU kernels per step of a DSpark trace, by name, with the CPU op stack
(cpu_op / python_function events enclosing the launching runtime call). Usage: eager.py TRACE_DIR"""
import bisect, collections, glob, gzip, json, os, sys

path = sys.argv[1]
if os.path.isdir(path):
    path = sorted(glob.glob(os.path.join(path, '*.trace.json*')))[0]
tr = json.load(gzip.open(path, 'rt'))
ev = tr['traceEvents']
gpu = sorted((e for e in ev if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')),
             key=lambda e: e['ts'])
ann = sorted((e for e in ev if e.get('ph') == 'X' and e.get('cat') == 'gpu_user_annotation'
              and str(e.get('name', '')).startswith('step[')), key=lambda e: e['ts'])
drafts = [a for a in ann if a['name'].startswith('step[DRAFT')]
verifs = [a for a in ann if a['name'].startswith('step[VERIFY')]
# runtime launches by correlation
rt = {e['args'].get('correlation'): e for e in ev if e.get('ph') == 'X' and e.get('cat') == 'cuda_runtime'
      and 'correlation' in e.get('args', {})}
cpu = [e for e in ev if e.get('ph') == 'X' and e.get('cat') in ('cpu_op', 'python_function', 'user_annotation')]
cpu_by_tid = collections.defaultdict(list)
for e in cpu:
    cpu_by_tid[(e['pid'], e['tid'])].append(e)
for v in cpu_by_tid.values():
    v.sort(key=lambda e: e['ts'])


def stack(r):
    lst = cpu_by_tid.get((r['pid'], r['tid']), [])
    out = [e for e in lst if e['ts'] <= r['ts'] and e['ts'] + e['dur'] >= r['ts'] + r['dur']]
    out.sort(key=lambda e: e['ts'])
    names = [e['name'] for e in out]
    py = [n for n in names if '.py(' in n]
    ops = [n for n in names if '.py(' not in n]
    return ' > '.join(py[-4:]), ' > '.join(ops[-3:])


starts = [g['ts'] for g in gpu]
agg = collections.defaultdict(lambda: [0, 0.0, None])
nper = 0
for i in range(len(drafts) - 1):
    d, dn = drafts[i], drafts[i + 1]
    v = [x for x in verifs if d['ts'] <= x['ts'] < dn['ts']]
    if len(v) != 1:
        continue
    v = v[0]
    nper += 1
    for g in gpu[bisect.bisect_left(starts, d['ts']):bisect.bisect_left(starts, dn['ts'])]:
        if (g.get('args', {}).get('graph id') or 0) != 0:
            continue
        seg = ('draft' if g['ts'] < d['ts'] + d['dur'] else 'gap_dv' if g['ts'] < v['ts']
               else 'verify' if g['ts'] < v['ts'] + v['dur'] else 'gap_vd')
        r = rt.get(g.get('args', {}).get('correlation'))
        st = stack(r) if r else ('?', '?')
        key = (seg, g['name'][:70], st[0][-160:], st[1][-90:])
        a = agg[key]
        a[0] += 1
        a[1] += g['dur']
tot = collections.Counter()
for (seg, *_), (n, dur, _) in agg.items():
    tot[seg] += dur
print(f"{path}: {nper} periods; eager us/step by segment:", {k: round(v / nper, 1) for k, v in tot.items()})
for (seg, name, py, ops), (n, dur, _) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    if dur / nper < 5:
        continue
    print(f"{seg:7s} {n / nper:5.1f}x {dur / nper:8.1f} us  {name}\n        py: {py}\n        ops: {ops}")
