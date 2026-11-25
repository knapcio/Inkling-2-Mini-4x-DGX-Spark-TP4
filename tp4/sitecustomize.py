"""TP4 start-up switch, loaded in every Python process of the container when tp4/ is first on PYTHONPATH.

INKLING_TP4_ROCE=1: install the RoCEnante hooks (inkling_sgl_roce.py; needs roce/ mounted at
INKLING_TP4_ROCE_PATH, default /opt/inkling-roce). Unset or 0: nothing is imported here and the
engine runs the image's own bytes. Then chain to the next sitecustomize on sys.path (the
single-box overlay's, when it is mounted behind tp4/), which this file shadows."""
import os
import sys

# Fail closed: Python only warns about a failing sitecustomize, and a rank that silently came up
# on NCCL while its peers wait in a RoCE vote would hang the boot (or serve unrouted).
if os.environ.get('INKLING_TP4_ROCE', '').strip().lower() not in ('', '0', 'off', 'false', 'no'):
    try:
        import inkling_sgl_roce
        inkling_sgl_roce.install()
    except BaseException as _exc:
        print(f'INKLING_TP4_ROCE: refusing to start: {_exc!r}', file=sys.stderr, flush=True)
        os._exit(78)

_here = os.path.dirname(os.path.abspath(__file__))
for _d in sys.path:
    if not _d or os.path.abspath(_d) == _here:
        continue
    _p = os.path.join(_d, 'sitecustomize.py')
    if os.path.isfile(_p):
        with open(_p) as _f:
            exec(compile(_f.read(), _p, 'exec'), {'__name__': 'sitecustomize_chained', '__file__': _p})
        break
