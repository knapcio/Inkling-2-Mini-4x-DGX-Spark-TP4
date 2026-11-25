#!/usr/bin/env python3
"""GPU check + microbench of the batched kv12s DSpark settle and the pressure probe's segment sums (no server).

  TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas python3 tools/gpu_check_settle.py

1. Settle: two identical DeviceLayers get the same multi-request verify steps (B requests x 8 rows, rows crossing
   page ends, random accepted lengths, graph-padding rows at loc 0). Lane A settles the v2.7.2 way (one candidate
   page per location), lane B the v2.7.3 way (one per row, shared prep). After every step: page status words,
   staging locks, event words and every accepted row's bits must agree, and both must equal the written K/V.
2. Timing: 8 lanes settled per step, v2.7.2 loop vs shared prep + row-compacted commit (eager, as served).
3. Pressure probe: segsum.segment_sums == torch.mm(locks.float(), segments) on 0/1 lock words; timed.
"""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

from inkling_kv12s.device import DeviceLayer  # noqa: E402
from inkling_kv12s.storage import DIM, PAGE, STAGE_MASK, STAGE_SHIFT, Layout  # noqa: E402

HEADS = 8
WIDTH = 8
FAILED = []


def log(*a):
    print(*a, flush=True)


def legacy_pages(loc):
    loc = loc.contiguous()
    return loc, torch.where(loc.remainder(PAGE) == PAGE - 1, loc // PAGE, 0)


def status(lane, pages):
    f = lane.flags[pages].long()
    staged = f >= 2
    # status class (0 empty / 1 packed / 2 staged) and the staged row count; slot ids may differ only by claim order
    return torch.where(staged, 2 + (f & STAGE_MASK) * 4, f).tolist()


def check_settle(device, B, steps, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pages_per_req = 4
    npages = 1 + B * pages_per_req
    stride = Layout(HEADS).size
    lanes = []
    for _ in range(2):
        raw = torch.zeros(npages * stride, dtype=torch.uint8, device=device)
        lanes.append(DeviceLayer(raw, HEADS, stride, 0, 2 * B + 8, 64, 8))
    table = [list(range(1 + i * pages_per_req, 1 + (i + 1) * pages_per_req)) for i in range(B)]
    store = torch.zeros((npages * PAGE, 2, HEADS, DIM), dtype=torch.bfloat16, device=device)
    prefix = [int(torch.randint(PAGE - 12, PAGE - 1, (1,), generator=g)) if i % 3 else int(
        torch.randint(1, PAGE - 9, (1,), generator=g)) for i in range(B)]
    history = []
    for i in range(B):  # prompt rows (non-tentative writes)
        pos = torch.arange(prefix[i])
        phys = torch.tensor(table[i], dtype=torch.int64)[pos // PAGE] * PAGE + pos % PAGE
        phys = phys.to(device)
        kv = torch.randn((phys.numel(), 2, HEADS, DIM), generator=None, device=device).to(torch.bfloat16)
        store[phys] = kv
        for lane in lanes:
            lane.write(phys, kv[:, 0], kv[:, 1])
        history += phys.tolist()
    pad = 2  # trailing graph-padding requests: loc 0 (the sink page)
    for step in range(steps):
        loc = torch.zeros((B + pad, WIDTH), dtype=torch.int64)
        for i in range(B):
            pos = torch.arange(prefix[i], prefix[i] + WIDTH)
            loc[i] = torch.tensor(table[i], dtype=torch.int64)[pos // PAGE] * PAGE + pos % PAGE
        loc = loc.to(device)
        kv = torch.randn((loc.numel(), 2, HEADS, DIM), device=device).to(torch.bfloat16)
        for lane in lanes:
            lane.write(loc.view(-1), kv[:, 0], kv[:, 1], tentative=True)
        acc = torch.randint(0, WIDTH + 1, (B + pad,), generator=g)
        acc[B:] = 0
        accept = acc.to(device=device, dtype=torch.int32)
        lanes[0].settle(loc, accept, prepared=legacy_pages(loc))
        lanes[1].settle(loc, accept)  # row-compacted (shared prep path builds the same)
        torch.cuda.synchronize()
        flat = loc.view(-1)
        for i in range(B):
            a = int(acc[i])
            rows = flat[i * WIDTH:i * WIDTH + a]
            store[rows] = kv[i * WIDTH:i * WIDTH + a]
            history += rows.tolist()
            prefix[i] += a
        all_pages = list(range(1, npages))
        sa, sb = status(lanes[0], all_pages), status(lanes[1], all_pages)
        if sa != sb:
            FAILED.append(f"B={B} step {step}: page status differs")
        if not torch.equal(lanes[0].stage_locks.ne(0).sum(), lanes[1].stage_locks.ne(0).sum()):
            FAILED.append(f"B={B} step {step}: staging lock count differs")
        if not torch.equal(lanes[0].events, lanes[1].events):
            FAILED.append(f"B={B} step {step}: events differ")
        want = store[history]
        for name, lane in zip("AB", lanes):
            got = lane.read_rows(history)
            if not torch.equal(got.contiguous().view(torch.int16), want.contiguous().view(torch.int16)):
                FAILED.append(f"B={B} step {step}: lane {name} accepted rows differ from the written K/V")
    packed = sum(1 for s in status(lanes[1], list(range(1, npages))) if s == 1)
    log(f"  settle B={B}: {steps} steps, {len(history)} accepted rows, {packed} pages sealed; "
        f"{'identical' if not FAILED else 'FAILED'}")


def time_settle(device, B, lanes_n=8, reps=60):
    stride = Layout(HEADS).size
    npages = 1 + B * 2
    lanes = []
    for _ in range(lanes_n):
        raw = torch.zeros(npages * stride, dtype=torch.uint8, device=device)
        lanes.append(DeviceLayer(raw, HEADS, stride, 0, 2 * B + 8, 64, 8))
    pos = torch.arange(40, 40 + WIDTH)
    loc = torch.stack([(1 + 2 * i) * PAGE + pos for i in range(B)]).to(device)
    accept = torch.full((B,), 3, dtype=torch.int32, device=device)
    for lane in lanes:
        lane.write(loc.view(-1), torch.zeros((B * WIDTH, HEADS, DIM), dtype=torch.bfloat16, device=device),
                   torch.zeros((B * WIDTH, HEADS, DIM), dtype=torch.bfloat16, device=device), tentative=True)

    def old():
        for lane in lanes:
            lane.settle(loc, accept, prepared=legacy_pages(loc))

    def new():
        prepared = lanes[0].settle_pages(loc)
        for lane in lanes:
            lane.settle(loc, accept, prepared=prepared)

    res = {}
    for name, fn in (("v2.7.2", old), ("batched", new)) * 2:
        fn(); torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            e0.record(); fn(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        res[name] = statistics.median(ts[5:])
    log(f"  settle timing B={B} ({B * WIDTH} rows, {lanes_n} lanes, eager): v2.7.2 {res['v2.7.2']:.1f} us, "
        f"batched {res['batched']:.1f} us -> {(res['v2.7.2'] - res['batched']) / 1000:.3f} ms per step")


def check_segsum(device):
    from inkling_kv12s.segsum import segment_sums

    for lanes, live, fallback, overflow in ((8, 64, 64, 256), (8, 200, 64, 1024), (3, 17, 0, 5)):
        n = live + fallback + overflow
        locks = (torch.rand((lanes, n), device=device) < 0.3).to(torch.int32)
        seg = torch.zeros((n, 3), dtype=torch.float32, device=device)
        seg[:live, 0] = 1
        seg[live:live + fallback, 1] = 1
        seg[live + fallback:, 2] = 1
        ref = torch.mm(locks.to(torch.float32), seg)
        got = segment_sums(locks, live, fallback)
        ok = torch.equal(ref, got)
        if not ok:
            FAILED.append(f"segsum {lanes}x{n}")
        ts = {"mm": [], "segsum": []}
        for _ in range(50):
            for k, fn in (("mm", lambda: torch.mm(locks.to(torch.float32), seg)),
                          ("segsum", lambda: segment_sums(locks, live, fallback))):
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); fn(); e1.record(); torch.cuda.synchronize()
                ts[k].append(e0.elapsed_time(e1) * 1e3)
        log(f"  pressure probe {lanes}x{n}: {'equal' if ok else 'DIFFERS'}; cast+mm "
            f"{statistics.median(ts['mm'][5:]):.1f} us, segsum {statistics.median(ts['segsum'][5:]):.1f} us")


def main():
    device = torch.device("cuda")
    torch.manual_seed(5)
    log("ptxas:", os.environ.get("TRITON_PTXAS_BLACKWELL_PATH", "bundled"))
    for B in (1, 4, 24):
        check_settle(device, B, steps=40, seed=B)
    for B in (1, 8, 24):
        time_settle(device, B)
    check_segsum(device)
    log("FAILED: " + "; ".join(FAILED[:10]) if FAILED else "ALL OK")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
