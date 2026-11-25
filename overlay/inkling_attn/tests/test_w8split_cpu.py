"""CPU check of the per-request 4/8-warp split (ops.W8_FROM): a mirror of the kernel's work-item loop and selection
block in csrc/attn_kernel.cuh

    if (p.sel_mode != 0) { pre = (long long)seq_k[b] - sq; if ((pre >= sel_keys) != (sel_mode == 2)) return; }

and of the two launches in csrc/inkling_attn_ext.cu (grid8: sel_mode 2, 128-row blocks, ceil(T/128) + n_req tiles;
grid: sel_mode 1, 64-row blocks). T (rows of q) is independent of cu_seqlens_q. Checks, per head:
  - the union of the two launches writes exactly the physical rows the single 4-warp launch writes, each once;
  - every request goes entirely to the launch its unclamped prefix (seqused_k - its rows) selects.

Run: python3 overlay/inkling_attn/tests/test_w8split_cpu.py   (or pytest)
"""
import random


def launch_rows(cu_q, seq_k, T, pt_cols, page_size, bm, sel_mode, sel_keys):
    """[(physical row, request)] written by one launch (one head)."""
    n_req = len(cu_q) - 1
    tiles = (T + bm - 1) // bm + n_req
    rows = []
    for lin in range(tiles):
        tile = lin
        b, start, sq, prev_end = -1, 0, 0, 0
        for i in range(n_req):
            s = max(cu_q[i], prev_end)
            e = min(cu_q[i + 1], T)
            ln = e - s if e > s else 0
            prev_end = max(prev_end, e)
            nb = (ln + bm - 1) // bm
            if tile < nb:
                b, start, sq = i, s, ln
                break
            tile -= nb
        if b < 0:
            continue
        sk = min(max(seq_k[b], 0), pt_cols * page_size)  # used by the attention itself, not by the selection
        del sk
        if sel_mode != 0:
            pre = seq_k[b] - sq
            if (pre >= sel_keys) != (sel_mode == 2):
                continue
        m0 = tile * bm
        rows += [(start + r, b) for r in range(m0, min(m0 + bm, sq))]
    return rows


def request_rows(cu_q, T):
    """rows each request owns under the kernel's guards: [max(cu[i], prev_end), min(cu[i+1], T))."""
    out, prev_end = [], 0
    for i in range(len(cu_q) - 1):
        s, e = max(cu_q[i], prev_end), min(cu_q[i + 1], T)
        prev_end = max(prev_end, e)
        out.append(max(0, e - s))
    return out


def check(cu_q, seq_k, T, pt_cols, page_size, w8_from):
    single = launch_rows(cu_q, seq_k, T, pt_cols, page_size, 64, 0, 0)
    r8 = launch_rows(cu_q, seq_k, T, pt_cols, page_size, 128, 2, w8_from)
    r4 = launch_rows(cu_q, seq_k, T, pt_cols, page_size, 64, 1, w8_from)
    phys = [r for r, _ in r8 + r4]
    assert len(phys) == len(set(phys)), "a physical row is written twice"
    assert sorted(r8 + r4) == sorted(single), "the split writes other rows than the single launch"
    lens = request_rows(cu_q, T)
    for b, sq in enumerate(lens):
        if sq == 0:
            continue
        want8 = seq_k[b] - sq >= w8_from
        owners = {"8" for _, rb in r8 if rb == b} | {"4" for _, rb in r4 if rb == b}
        assert owners == ({"8"} if want8 else {"4"}), (b, owners, want8)


def test_random():
    rng = random.Random(0)
    for _ in range(4000):
        page_size = rng.choice([64, 128])
        n_req = rng.randint(1, 6)
        sqs = [rng.choice([0, 1, 7, 63, 64, 65, 127, 128, 129, 300, 2048]) for _ in range(n_req)]
        w8_from = rng.choice([1, 64, 128, 1000, 131072])
        pre = [rng.choice([0, 5, w8_from - 1, w8_from, w8_from + 1, 2 * w8_from, 300000]) for _ in range(n_req)]
        seq_k = [s + max(p, 0) for s, p in zip(sqs, pre)]
        if rng.random() < 0.15:
            seq_k[rng.randrange(n_req)] = rng.choice([0, 3])  # inconsistent metadata (sk < sq)
        cu = [0]
        for s in sqs:
            cu.append(cu[-1] + s)
        if rng.random() < 0.1 and cu[-1] > 0:
            cu[rng.randrange(1, len(cu))] -= rng.randint(1, 70)  # non-monotone cu_seqlens: the kernel's guards apply
            cu = [max(0, c) for c in cu]
        T = max(cu) + rng.choice([0, 0, 1, 63, 200])  # q rows independent of cu_seqlens_q (padded buckets)
        if rng.random() < 0.1:
            T = max(0, max(cu) - rng.randint(1, 100))  # cu_seqlens_q beyond q: guarded by total_q
        pt_cols = (max(max(seq_k), 1) + page_size - 1) // page_size
        if rng.random() < 0.15:
            pt_cols = max(1, pt_cols // 4)  # page table narrower than seq_k: attention clamps, selection does not
        check(cu, seq_k, T, pt_cols, page_size, w8_from)


def test_threshold_direction():
    # one request exactly at / one below the threshold: owners must be 8 / 4 warps
    r8 = launch_rows([0, 100, 200], [100 + 1000, 100 + 999], 256, 64, 128, 128, 2, 1000)
    r4 = launch_rows([0, 100, 200], [100 + 1000, 100 + 999], 256, 64, 128, 64, 1, 1000)
    assert {b for _, b in r8} == {0} and {b for _, b in r4} == {1}
    # clamped page table: sq 16, seqused_k 256, capacity 128 keys, W8_FROM 128 -> 8 warps
    assert {b for _, b in launch_rows([0, 16], [256], 16, 1, 128, 128, 2, 128)} == {0}


def test_serving_shapes():
    for pfx in (0, 131071, 131072, 262144, 1048576 - 8192):
        check([0, 8192], [8192 + pfx], 8192, 8192, 128, 131072)
    check([0, 100, 4196, 8192], [100 + 300000, 4096 + 7, 3996 + 131072], 8192, 8192, 128, 131072)


if __name__ == "__main__":
    test_random()
    test_threshold_direction()
    test_serving_shapes()
    print("ok")
