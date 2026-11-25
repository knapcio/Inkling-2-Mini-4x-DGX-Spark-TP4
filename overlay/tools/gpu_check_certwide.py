#!/usr/bin/env python3
"""Check and time the batched certified head in the GB10 SGLang image.

Run from overlay/ (or a development checkout's root) with the server stopped. For example:
  python3 tools/gpu_check_certwide.py --model /model --hidden /work/hid
  python3 tools/gpu_check_certwide.py --model /model --synthetic
  python3 tools/gpu_check_certwide.py --fixture-only
  python3 tools/gpu_check_certwide.py --model WEIGHTS --hidden CAPTURE --cpu-count

The last form runs the interval model on the CPU and measures candidate
unions on captured rows. It does not test or time CUDA.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT if (ROOT / "inkling_cert").is_dir() else ROOT / "overlay"))  # overlay/ or a development checkout's root

from inkling_cert import build as cert_build, cert_math as cm, certwide as cw, certwide_math as wm, quant
from inkling_pico12 import build as pico_build, ops as pico
from inkling_wide import build as wide_build, ops as wide

ROWS = tuple(int(r) for r in os.environ.get("CW_ROWS", "24,64,128,192").split(","))  # CW_ROWS: crossover sweep
FIXTURE_ROWS = (17, 24, 33, 49, 64, 65, 97, 128, 129, 192)
KEY = "model.llm.unembed.weight"
REASON = {cw.CERTIFIED: "certified", cw.UNSAFE: "unsafe", cw.TOO_MANY: "too_many", cw.OFF: "off", cw.INVALID: "invalid"}


def load_head(model, device):
    from safetensors import safe_open

    index = model / "model.safetensors.index.json"
    name = json.loads(index.read_text())["weight_map"][KEY] if index.is_file() else "model-00001-of-00001.safetensors"
    with safe_open(str(model / name), framework="pt", device="cpu") as f:
        return f.get_tensor(KEY).to(device)


def load_capture(directory):
    if directory is None:
        return []
    result = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path) as capture:
            rows = capture["hidden"]
            if rows.ndim != 2 or rows.shape[1] != cm.K or rows.shape[0] < 8:
                raise ValueError(f"{path}: expected at least eight [2304] rows in hidden")
            if rows.dtype.kind in "iu" and rows.dtype.itemsize == 2:
                x = torch.from_numpy(rows.astype(np.int16, copy=True)).view(torch.bfloat16)
            elif rows.dtype.kind == "f":
                x = torch.from_numpy(rows.copy()).to(torch.bfloat16)
            else:
                raise ValueError(f"{path}: expected BF16 bits or floating-point hidden rows")
            result.append(x)
    return result


def captured_batches(captures, steps):
    """Eight adjacent oracle rows per request, with a cyclic transcript mix."""
    for step in range(steps):
        windows = []
        for request in range(max(ROWS) // 8):
            transcript = captures[(step + request) % len(captures)]
            # Different positions when the cohort wraps the capture set.
            block = (step * 7 + request // len(captures) * 17) % (len(transcript) // 8)
            windows.append(transcript[8 * block:8 * block + 8])
        yield torch.cat(windows)


def synthetic_batches(w, steps, v_real):
    """Use the real head with inputs that have a known large positive margin."""
    g = torch.Generator(device=w.device).manual_seed(13)
    for _ in range(steps):
        ids = torch.randint(v_real, (max(ROWS),), generator=g, device=w.device)
        probe = w[ids].float()
        noise = torch.randn(probe.shape, generator=g, device=w.device)
        # Directional logit about 55 and ||x|| about 223, like oracle rows.
        x = torch.nn.functional.normalize(probe) * 55 + torch.nn.functional.normalize(noise) * 215
        yield x.to(torch.bfloat16)


def describe(counts):
    if not counts:
        return "unavailable"
    q = np.percentile(counts, [50, 90, 99])
    return (f"n={len(counts)} mean={statistics.mean(counts):.1f} p50={q[0]:.0f} "
            f"p90={q[1]:.0f} p99={q[2]:.0f} max={max(counts)}")


def cpu_counts(w, batches, v_real):
    # Keep only per-(row, tile) upper bounds and each row's lower maximum.
    # The CPU GEMM is a proxy for the GPU screen arithmetic; this mode also
    # checks its interval against the BF16-weight CPU dot product.
    x = torch.cat(batches).float().numpy()
    nrows, v = len(x), w.shape[0]
    ht = np.empty((nrows, v // 16), np.float32)
    T = np.full(nrows, -np.inf, np.float32)
    norm = wm.activation_norm(x)
    peak_ratio = 0.0
    for start in range(0, v, 4096):
        rows = w[start:start + 4096].float().numpy()
        _, _, d = cm.quantize(rows)
        rt, nv = cm.tables(rows, d)
        r = wm.weight_radius(rt, nv)
        d = d.astype(np.float32)
        for a in range(0, nrows, 192):
            b = min(a + 192, nrows)
            s = x[a:b] @ d.T
            exact = x[a:b] @ rows.T
            lo, hi = wm.intervals(s, norm[a:b], r)
            assert np.all((lo <= exact) & (exact <= hi)), "CPU interval missed W dot x"
            peak_ratio = max(peak_ratio, float((np.abs(exact - s) / np.maximum(hi - s, 1e-30)).max()))
            valid = max(0, min(v_real - start, len(rows)))
            lo[:, valid:] = hi[:, valid:] = -np.inf
            T[a:b] = np.maximum(T[a:b], lo.max(-1))
            ht[a:b, start // 16:(start + len(rows)) // 16] = hi.reshape(b - a, -1, 16).max(-1)
    cutoff, safe = wm.thresholds(T, ht.max(-1))
    candidate = ht >= cutoff[:, None]
    print(f"CPU FP32 dot-product proxy, {nrows} rows, largest observed error/radius {peak_ratio:.4f}")
    for m in ROWS:
        counts = []
        status = Counter()
        for a in range(0, nrows, max(ROWS)):
            b = a + m
            if not np.all(safe[a:b] & (norm[a:b] <= 2.0**20)):
                status["unsafe"] += 1
                continue
            count = int(candidate[a:b].any(0).sum())
            counts.append(count)
            status["too_many" if count > cw.MAX_TILES or count == ht.shape[1] else "certified"] += 1
        print(f"M={m:3}: union {describe(counts)}; {dict(status)}", flush=True)


def packed_head(w, name):
    pack = pico.encode(w, name)
    return wide.make(w, name, "head", pack)


def raw_logits(x, entry, out):
    # Explicit S=1 also lets the small adversarial head use the same arithmetic.
    empty = out.new_empty(0)
    wide.EXT.gemm(
        x, x, x, x.shape[0], entry.buf, wide.BF16X, entry.n, entry.k,
        wide.NO_REGION, wide.NO_REGION, wide.NO_REGION, 0, 0, out,
        1, empty, empty, wide.SET.pdl, wide.SET.mt, wide.SET.mg, wide.SET.pfd,
    )


def stock_ids(out, v_real):
    from sglang.kernels.ops.activation.softcap import softcap_inplace_logits
    from sglang.kernels.ops.sampling import argmax

    softcap_inplace_logits(out, cm.CAP)
    out[:, v_real:] = -float("inf")
    return argmax(out)


def same_bits(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def sparse_indices(stack, m, count, v_real):
    tiles = stack.tiles[:count].long()
    assert tiles.unique().numel() == count, "candidate compaction duplicated a tile"
    col = (tiles[:, None] * 16 + torch.arange(16, device=tiles.device)).flatten()
    row = torch.arange(m, device=tiles.device) // 16
    keep = ((stack.groups[:count][None, :] >> row[:, None]) & 1).bool().repeat_interleave(16, -1)
    assert bool((col < v_real).any()) and bool((keep & (col < v_real)).any(dim=1).all())
    return col, keep


def check_rounding(stack, x, screen, exact, *, cpu_reference=False):
    reference_device = torch.device("cpu") if cpu_reference else x.device
    x64 = x.to(reference_device).double()
    xnorm = x64.norm(dim=1)
    worst = [0.0, 0.0]
    for start in range(0, stack.V, 4096):
        w = stack.entry.weight[start:start + 4096].to(reference_device)
        _, _, d64 = quant.quantize_rows(w)
        end = start + len(w)
        for i, (got, weight) in enumerate(((screen[:, start:end], d64), (exact[:, start:end], w.double()))):
            want = x64 @ weight.T
            scale = xnorm[:, None] * weight.norm(dim=1)[None, :]
            # Allow for the FP64 reference's own roundoff, including cancellation.
            budget = (2 * cm.C_ACC + 2.0**-40) * scale + cw.ABS_SLOP
            ratio = float((torch.abs(got.to(reference_device).double() - want) / budget).max())
            assert ratio <= 1, f"M={len(x)} {('screen', 'wide')[i]} exceeded its FP64 budget: {ratio:.5g}"
            worst[i] = max(worst[i], ratio)
    return worst


def check_intervals(stack, x, ref, debug, v_real, *, cpu_reference=False):
    screen, lo, hi = debug
    assert bool((stack.norms[:len(x)] <= 2.0**20).all()), "qualification inputs did not pass the norm gate"
    for start in range(0, v_real, 4096):
        sl = slice(start, min(start + 4096, v_real))
        assert bool(((lo[:, sl] <= ref[:, sl]) & (ref[:, sl] <= hi[:, sl])).all()), (
            f"M={len(x)} production interval missed a wide logit in vocab block {start}"
        )
    assert bool((lo[:, v_real:] == -torch.inf).all() & (hi[:, v_real:] == -torch.inf).all())
    for endpoints, maxima in ((lo, stack.lower), (hi, stack.upper)):
        assert same_bits(endpoints.view(len(x), -1, 16).amax(-1), maxima[:len(x)]), "emitted tile maxima differ"
    return check_rounding(stack, x, screen, ref, cpu_reference=cpu_reference)


def check_dense(out, ref):
    # Match the NaN handling of the CUDA argmax for out-of-range inputs.
    for a, b in zip(out.split(16), ref.split(16)):
        match = (a.view(torch.int32) == b.view(torch.int32)) | (a.isnan() & b.isnan())
        assert bool(match.all()), "dense fallback logits differ"


def compare(stack, x, out, ref, on, v_real, *, qualify=False, three_stages=False):
    raw_logits(x, stack.entry, ref)
    debug = tuple(torch.empty_like(ref) for _ in range(3)) if qualify else None
    capped = torch.empty_like(stack.sparse)
    out.fill_(7)
    got = stack.run(x, out, on, v_real, wide.SET, debug=debug, capped=capped, three_stages=three_stages)
    reason, count = stack.state.tolist()
    if qualify:
        screen_ratio, wide_ratio = check_intervals(stack, x, ref, debug, v_real)
        print(f"M={len(x):3} FP64 max error/budget: screen={screen_ratio:.5g}, wide={wide_ratio:.5g}", flush=True)
    if reason == cw.CERTIFIED:
        col, keep = sparse_indices(stack, len(x), count, v_real)
        assert same_bits(stack.sparse[:len(x), :count * 16][keep], ref[:, col][keep]), "compact raw bits differ"
        assert bool((out == 7).all()), "certified path wrote the fallback logit buffer"
    expected = stock_ids(ref, v_real)
    assert torch.equal(got, expected), f"M={len(x)} {REASON[reason]} token IDs differ"
    if reason == cw.CERTIFIED:
        live = keep & (col < v_real)
        assert same_bits(capped[:len(x), :count * 16][live], ref[:, col][live]), "compact capped bits differ"
    else:
        check_dense(out, ref)
    return reason, count


def fixture_head(cert_ext, device):
    w = torch.zeros((256, cm.K), dtype=torch.bfloat16, device=device)
    w[3, 0] = w[17, 0] = 1
    w[17, 1] = 2.0**-17  # one FP32 ulp when x[0] is between 64 and 128
    w[34, 2] = w[249, 2] = 1
    w[255, 3] = 1  # padded, inside a selected tile, and larger than any live logit
    entry = packed_head(w, "sparse fixture")
    return cw.CertWide(cert_ext, wide.EXT, entry, quant.build_tables(w))


def rounding_head(cert_ext, device):
    # Disjoint K slices exercise the two accumulation chains over the whole K.
    # One row cancels within/across MMA steps and FP16 scale groups. Another
    # spans widely separated exponents within a BF16X tile and within an MMA.
    w = torch.zeros((256, cm.K), dtype=torch.bfloat16, device=device)
    w[52, 16::2], w[52, 17::2] = 128, -128
    w[52, -1] = -127
    exponents = torch.arange(cm.K - 16, device=device) % 36 - 21
    w[70, 16:] = torch.exp2(exponents.float())
    tiny = torch.tensor([1, 2, 64, 128, 256], dtype=torch.int16, device=device).view(torch.bfloat16)
    w[90, 16:21] = tiny
    w[110, 16:21] = torch.tensor([1, -1, 1, -1, 1], device=device)
    entry = packed_head(w, "qualification")
    return cw.CertWide(cert_ext, wide.EXT, entry, quant.build_tables(w)), tiny


def fixture_inputs(stack, m, v_real, *, tie=True):
    x = torch.zeros((m, cm.K), device=stack.q.device, dtype=torch.bfloat16)
    x[:, 1], x[:, 3] = 8, 255
    ref = torch.empty((m, stack.V), device=x.device)
    for value in (100, 104, 108, 112, 116, 120, 124):
        x[:, 0] = value
        raw_logits(x, stack.entry, ref)
        assert bool((ref[:, 17] > ref[:, 3]).all()), "raw near-tie construction failed"
        stock_ids(ref, v_real)
        if torch.equal(ref[:, 3], ref[:, 17]):
            break
    else:
        raise AssertionError("could not construct a stock capped tie")
    if tie:
        x[16:, :2] = 0
        x[16:, 2] = 50
    return x


def rounding_fixture(stack, tiny, m, v_real, three_stages):
    x = torch.zeros((m, cm.K), device=stack.q.device, dtype=torch.bfloat16)
    x[::4, 16:] = 64
    x[1::4, 16:] = torch.exp2((10 - torch.arange(cm.K - 16, device=x.device) % 41).float())
    x[2::4, 16:21] = 2.0**16
    x[3::4, 16:21] = tiny
    assert float(x[3, 16].cpu().double()) == 2.0**-133, "BF16 subnormal fixture lost its input bits"
    raw = torch.empty((m, stack.V), device=x.device)
    on = torch.ones(1, dtype=torch.int32, device=x.device)
    raw_logits(x, stack.entry, raw)
    debug = tuple(torch.empty_like(raw) for _ in range(3))
    stack.screen(x, on, v_real, debug, three_stages=three_stages)
    ratios = check_intervals(stack, x, raw, debug, v_real, cpu_reference=True)
    print(f"M={m:3} cancellation/exponent/subnormal error/budget: screen={ratios[0]:.5g}, wide={ratios[1]:.5g}",
          flush=True)


def cutoff_fixture(stack):
    # Feed exact binary32 endpoints to the production threshold/compaction code.
    # Move a tile across the inclusive boundary by one representable float.
    device = stack.q.device
    m = 17
    lo = torch.full((m, stack.ntiles), -torch.inf, device=device)
    hi = torch.full_like(lo, -torch.inf)
    lo[:, 0] = hi[:, 0] = 100

    def select():
        stack.screen_ext.debug_select_wide(lo, hi, stack.threshold, stack.state, stack.tiles, stack.groups,
                                           cm.CAP, cm.TWO_E, cm.DMAX)
        reason, count = stack.state.tolist()
        assert reason == cw.CERTIFIED
        return {tile: mask for tile, mask in zip(stack.tiles[:count].tolist(), stack.groups[:count].tolist())}

    assert select() == {0: 3}
    cutoff = stack.threshold[:m].clone()
    slope = np.exp(-2 * 116 / cm.CAP)
    assert bool(((100 - cutoff.double()) * slope >= 2 * cm.TWO_E).all()), "cutoff lost the slope margin"
    hi[-1, 1] = cutoff[-1]
    hi[-1, 2] = torch.nextafter(cutoff[-1], torch.full_like(cutoff[-1], -torch.inf))
    hi[-1, 3] = torch.nextafter(cutoff[-1], torch.full_like(cutoff[-1], torch.inf))
    assert select() == {0: 3, 1: 2, 3: 2}

    # Exercise the softcap error contract in the stock deployment kernel near
    # the cutoff and over the accepted raw-logit range, in both signs.
    extremes = torch.exp2(torch.arange(-149, 49, device=device, dtype=torch.float32))
    values = torch.cat((torch.linspace(-192, 192, 32769, device=device), cutoff, -cutoff, extremes, -extremes))
    values = values.view(1, -1)
    ideal = cm.CAP * (values.double() / cm.CAP).tanh()
    capped = values.clone()
    from sglang.kernels.ops.activation.softcap import softcap_inplace_logits

    softcap_inplace_logits(capped, cm.CAP)
    assert float((capped.double() - ideal).abs().max()) <= cm.E_SOFTCAP, "stock softcap exceeded its error bound"


def must_reject(fn, message):
    try:
        fn()
    except RuntimeError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"binding accepted {message}")


def invalid_exact_fixture(stack, x, out, ref, v_real):
    on = torch.ones(1, device=x.device, dtype=torch.int32)
    stack.screen(x, on, v_real)
    narrow = out.new_empty((len(x), stack.V - 16))
    must_reject(lambda: stack.exact(x, narrow, v_real, wide.SET), "bad exact workspace")
    misaligned = ref.new_empty(ref.numel() + 1)[1:].view_as(ref)
    must_reject(lambda: stack.screen(x, on, v_real, (misaligned, out, ref)), "misaligned float2")
    raw_logits(x, stack.entry, ref)
    expected = stock_ids(ref, v_real)
    for reason, count, tile, group in ((0, -1, 0, 3), (0, 0, 0, 3), (0, stack.capacity + 1, 0, 3),
                                       (0, 1, -1, 3), (0, 1, stack.ntiles, 3),
                                       (0, 1, 0, 0), (0, 1, 0, 4), (0, 1, 0, 1), (99, 1, 0, 3)):
        stack.state[0], stack.state[1] = reason, count
        stack.tiles[0], stack.groups[0] = tile, group
        stack.exact(x, out, v_real, wide.SET)
        ids = torch.empty(len(x), dtype=torch.int64, device=x.device)
        stack.reduce(out, stack.sparse, stack.state, stack.tiles, stack.groups, stack.partial, ids, v_real, cm.CAP)
        assert stack.state.tolist() == [cw.INVALID, count]
        assert torch.equal(ids, expected)
        check_dense(out, ref)


def check_replay(stack, base_graph, base_ids, cert_graph, cert_ids, x, out, ref, on, cases):
    stack.require_replay_stream()
    for changed, enabled, reason, count in cases:
        on.fill_(enabled)
        x.copy_(changed)
        out.fill_(7)
        base_graph.replay()
        cert_graph.replay()
        actual = tuple(stack.state.tolist())
        assert actual == (reason, count), f"M={len(x)} replay route: expected {(reason, count)}, got {actual}"
        assert torch.equal(cert_ids, base_ids), f"M={len(x)} CUDA graph replay IDs differ"
        if reason == cw.CERTIFIED:
            assert bool((out == 7).all()), "sparse replay wrote dense logits"
        else:
            check_dense(out, ref)


def adversarial(cert_ext, device):
    stack = fixture_head(cert_ext, device)
    rounding, tiny = rounding_head(cert_ext, device)
    v_real = stack.V - 3
    on = torch.ones(1, device=device, dtype=torch.int32)
    for m in FIXTURE_ROWS:
        x = fixture_inputs(stack, m, v_real)
        out = torch.empty((m, stack.V), device=device)
        ref = torch.empty_like(out)
        for three_stages in ((False, True) if m > 128 else (False,)):
            assert compare(stack, x, out, ref, on, v_real, qualify=True, three_stages=three_stages) == (cw.CERTIFIED, 4)
            rounding_fixture(rounding, tiny, m, v_real, three_stages)
        ids = stack.run(x, out, on, v_real, wide.SET)
        assert bool((ids[:16] == 3).all() & (ids[16:] == 34).all())

    cutoff_fixture(stack)
    m = 24
    x = fixture_inputs(stack, m, v_real)
    out = torch.empty((m, stack.V), device=device)
    ref = torch.empty_like(out)
    invalid_exact_fixture(stack, x, out, ref, v_real)
    unsafe = x.clone()
    for value in (400, float("nan")):
        unsafe[0, 0] = value
        assert compare(stack, unsafe, out, ref, on, v_real)[0] == cw.UNSAFE
    flat = x.clone()
    flat[0].zero_()
    assert compare(stack, flat, out, ref, on, v_real) == (cw.TOO_MANY, stack.ntiles)

    # Reuse one captured head with two slots. Overflow here is exactly four
    # live tiles, so the all-vocab guard cannot make the test pass.
    small = cw.CertWide(cert_ext, wide.EXT, stack.entry, quant.build_tables(stack.entry.weight), capacity=2)
    sparse = fixture_inputs(stack, m, v_real, tie=False)
    dense_graph, dense_ids = graph(lambda: baseline(x, stack.entry, ref, v_real))
    sparse_graph, sparse_ids = graph(lambda: small.run(x, out, on, v_real, wide.SET))
    check_replay(small, dense_graph, dense_ids, sparse_graph, sparse_ids, x, out, ref, on,
                 ((sparse, 1, cw.CERTIFIED, 2), (x.clone(), 1, cw.TOO_MANY, 4),
                  (sparse, 0, cw.OFF, 0), (sparse, 1, cw.CERTIFIED, 2)))
    print("GB10 qualification passed: all CUDA buckets, cutoff, ties, padding, malformed state, and route replay",
          flush=True)
    return stack, sparse


def baseline(x, entry, ref, v_real):
    raw_logits(x, entry, ref)
    return stock_ids(ref, v_real)


def graph(fn):
    # Stock CUDA argmax caches its workspace per stream. Warm the stream
    # that will actually be captured, including the Triton specializations.
    ambient = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(ambient)
    with torch.cuda.stream(side):
        fn()
    ambient.wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        result = fn()
    return g, result


def cold_ms(g, flush, reps):
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for start, end in events:
        flush.sum()  # 256 MiB read, above GB10's L2. The scalar output is unused.
        start.record()
        g.replay()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in events)


def dspark_fixture(stack, sparse):
    """Capture the pinned DSpark static epilogue with the real tensor handoff."""
    from inkling_cert import patches
    from inkling_vcap import patches as vcap
    from inkling_wide import patches as wide_patches
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.models.inkling2 import Inkling2LogitsProcessor
    from sglang.srt.speculative.dspark_components import dspark_verify as verify
    from sglang.srt.speculative.spec_tp_sync import SpecTpSync

    device = sparse.device
    m, stride, v_real = len(sparse), 8, stack.V - 3
    bs = m // stride
    x = sparse.clone()
    mixed = fixture_inputs(stack, m, v_real)
    out = torch.empty((m, stack.V), device=device)
    ref = torch.empty_like(out)
    on = torch.ones(1, dtype=torch.int32, device=device)
    cert = SimpleNamespace(wide=stack, on=on)
    mode = ForwardMode.TARGET_VERIFY
    runner = SimpleNamespace(model_runner=SimpleNamespace(is_draft_worker=False), ragged_verify_mode=False)
    head = SimpleNamespace(weight=stack.entry.weight)
    lp = SimpleNamespace(config=SimpleNamespace(vocab_size=v_real), vocab_size=stack.V,
                         use_fp32_lm_head=True, shard_softcap=cm.CAP)
    meta = SimpleNamespace(forward_mode=mode, next_token_logits_buffer=out)
    forward = SimpleNamespace(forward_mode=mode, batch_size=bs,
                              input_ids=torch.zeros(m, dtype=torch.int64, device=device),
                              seq_lens=torch.full((bs,), 32, dtype=torch.int32, device=device),
                              req_pool_indices=torch.arange(bs, dtype=torch.int64, device=device),
                              out_cache_loc=torch.arange(m, dtype=torch.int64, device=device),
                              positions=torch.arange(m, dtype=torch.int64, device=device))
    tp_sync = SpecTpSync(SimpleNamespace(world_size=1))

    patched_head = patches.make_get_logits(Inkling2LogitsProcessor._get_logits)
    with (mock.patch.dict(patches.STATE, cert=cert, disabled=None, mode="1", cfg={"wide": True}, patched=[]),
          mock.patch.dict(wide_patches.HEADS, {head.weight.data_ptr(): stack.entry}),
          mock.patch.object(wide.SET, "splitk", {"head": 1}),
          mock.patch.object(wide, "IN_SCOPE", [True])):
        patches.patch_verify(verify)
        for cutoff in (False, True):
            if cutoff:
                # Use the same ordinary tail and device cutoff as serving with vcap.
                vcap.patch_verify(verify)
            with mock.patch.multiple(vcap.STATE, enabled=cutoff, capping=cutoff):
                epilogue = verify.DsparkVerifyEpilogue(
                    max_bs=bs, verify_num_draft_tokens=stride, device=device,
                    tp_sync=tp_sync, fused_accept_tail=True,
                )
                epilogue.begin_static_step(bs, True)
                if cutoff:
                    epilogue.verify_lens_buf.fill_(4)

                def serve(certified):
                    if certified and torch.cuda.is_current_stream_capturing():
                        logits = patched_head(lp, x, head, meta)
                        assert getattr(logits, "_inkling_certwide_ids", None) is not None
                    else:
                        baseline(x, stack.entry, ref, v_real)
                        logits = ref
                    epilogue.capture_hook(runner, LogitsProcessorOutput(next_token_logits=logits, hidden_states=x),
                                          forward, m)

                base_graph, _ = graph(lambda: serve(False))
                cert_graph, _ = graph(lambda: serve(True))
                if cutoff:
                    assert vcap.STATE.static_accept_recorded > 0
                else:
                    assert epilogue._serves_accept_tail(bs)
                fields = ("correct_len_buf", "bonus_buf", "cap_trim_lens_buf", "commit_lens_buf",
                          "new_seq_lens_buf", "out_tokens_buf")
                stack.require_replay_stream()
                for changed, enabled, reason, count in (
                    (sparse, 1, cw.CERTIFIED, 2), (mixed, 0, cw.OFF, 0), (mixed, 1, cw.CERTIFIED, 4)
                ):
                    x.copy_(changed)
                    on.fill_(enabled)
                    candidates = forward.input_ids.view(bs, stride)
                    expected_ids = baseline(x, stack.entry, ref, v_real).view(bs, stride)
                    candidates[:, 1:] = expected_ids[:, :-1]
                    candidates[0, 2] = 42  # Exercise a mismatch and a matching prefix in the same graph.
                    base_graph.replay()
                    expected = [getattr(epilogue, key).clone() for key in fields]
                    if cutoff:
                        assert bool((epilogue.cap_trim_lens_buf[1:] > 0).all()), "cutoff was not exercised"
                    for key in fields:
                        getattr(epilogue, key).fill_(-1)
                    out.fill_(7)
                    cert_graph.replay()
                    assert stack.state.tolist() == [reason, count], "DSpark graph did not take the expected route"
                    for key, want in zip(fields, expected):
                        assert torch.equal(getattr(epilogue, key), want), f"DSpark accept differs in {key}"
                    if reason == cw.CERTIFIED:
                        assert bool((out == 7).all()), "DSpark sparse replay wrote dense logits"
                    else:
                        check_dense(out, ref)
                del base_graph, cert_graph
    print("DSpark static epilogue graph passed with fused and cutoff-aware ordinary accept", flush=True)


EXTENSIONS = {
    "pico": (pico_build, "INKLING_PICO12_CACHE"),
    "wide": (wide_build, "INKLING_WIDE_CACHE"),
    "cert": (cert_build, "INKLING_CERT_HEAD_CACHE"),
}


def build_extensions(output):
    paths = {}
    for name, (builder, cache_env) in EXTENSIONS.items():
        print(f"building or loading {name} extension", flush=True)
        cache = os.environ.get(cache_env) or (os.environ.get("INKLING_CERT_CACHE") if name == "cert" else None)
        ext = builder.load(cache_dir=cache)
        paths[name] = {"module": ext.__name__, "path": ext.__file__}
        print(f"{name} {builder.INFO['tag']} ready in {builder.INFO['seconds']:.1f} s", flush=True)
    output.write_text(json.dumps(paths))


def prebuild(timeout):
    # Reuse the exact extension files the worker loaded. The benchmark process
    # never enters another cpp_extension/ninja lock after the timeout.
    with tempfile.TemporaryDirectory(prefix="certwide-") as directory:
        output = Path(directory) / "extensions.json"
        print(f"prebuilding CUDA extensions (timeout {timeout:g} s)", flush=True)
        start = time.monotonic()
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--build-output", str(output)],
                                start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise RuntimeError("extension build timed out; inspect the build log and cache lock") from exc
        if code:
            raise RuntimeError(f"extension build exited with status {code}")
        paths = json.loads(output.read_text())
    print(f"extension build stage finished in {time.monotonic() - start:.1f} s", flush=True)
    extensions = {}
    for key, info in paths.items():
        spec = importlib.util.spec_from_file_location(info["module"], info["path"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        extensions[key] = module
    pico.EXT, wide.EXT = extensions["pico"], extensions["wide"]
    return extensions["cert"]


class DeviceMemory:
    def __init__(self, device):
        self.device = device
        self.packed = 0
        self.argmax = 0
        self.peak_accounted = 0
        self.peak_driver = self.driver_used()
        self.start_driver = self.peak_driver

    def driver_used(self):
        free, total = torch.cuda.mem_get_info(self.device)
        return total - free

    @contextmanager
    def stage(self, label, external_scratch=0):
        print(f"{label} ...", flush=True)
        start = time.monotonic()
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        yield
        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.max_memory_allocated(self.device)
        reserved = torch.cuda.max_memory_reserved(self.device)
        external = self.packed + self.argmax + external_scratch
        driver = self.driver_used()
        self.peak_driver = max(self.peak_driver, driver)
        self.peak_accounted = max(self.peak_accounted, reserved + external)
        print(f"{label} finished in {time.monotonic() - start:.2f} s; "
              f"torch peak allocated/reserved {allocated / 1e9:.3f}/{reserved / 1e9:.3f} GB, "
              f"external bound {external / 1e9:.3f} GB, device used {driver / 1e9:.3f} GB", flush=True)
        assert self.peak_accounted < 10e9, "benchmark allocations exceeded the 10 GB device memory budget"
        assert self.peak_driver - self.start_driver < 10e9, "device memory grew by more than 10 GB"


def profile_routes(directory, stack, x, out, on, v_real, sparse, overflow, three_stages):
    directory.mkdir(parents=True, exist_ok=True)
    m = len(x)
    for label, inputs in (("sparse", sparse), ("overflow", overflow)):
        x.copy_(inputs)
        torch.cuda.synchronize(x.device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function(f"certwide_M{m}_{label}"):
                stack.run(x, out, on, v_real, wide.SET, three_stages=three_stages)
            torch.cuda.synchronize(x.device)
        prof.export_chrome_trace(str(directory / f"M{m}-{label}.json"))
        wide_kernels = [event for event in prof.events()
                        if event.device_type == torch.autograd.DeviceType.CUDA and "k_wide" in event.name]
        if len(wide_kernels) != 2:
            raise RuntimeError(f"M={m} {label}: profiling did not capture both guarded wide launches")
        dense, compact = sorted(wide_kernels, key=lambda event: event.time_range.start)
        print(f"M={m:3} {label} kernel trace: dense={dense.time_range.elapsed_us():.2f} us, "
              f"compact={compact.time_range.elapsed_us():.2f} us", flush=True)


def gpu_check(w, batches, v_real, reps, cert_ext, memory, three_stages, profile_dir):
    tiles = (len(w) // 8) * (cm.K // 32)
    # The allocator sees the tiled input but BF16X output and device headers
    # use cudaMalloc. The decoder checker runs after headers are released.
    with memory.stage("pack real head", external_scratch=max(tiles * 4, 8)):
        entry = packed_head(w, "lm_head")
        memory.packed = entry.nbytes
    with memory.stage("build INT6 tables and allocate workspace"):
        stack = cw.CertWide(cert_ext, wide.EXT, entry, quant.build_tables(w))
        on = torch.ones(1, device=w.device, dtype=torch.int32)
        flush = torch.ones(256 * 2**20 // 4, dtype=torch.float32, device=w.device)
    screen_bytes = sum(t.numel() * t.element_size() for t in (stack.q, stack.sc, stack.radius))
    print(f"screen {screen_bytes / 1e6:.2f} MB; dense packed head {entry.nbytes / 1e6:.2f} MB", flush=True)

    # SGLang holds up to 16,384 int64 argmax slots per stream. Bound the
    # ambient stream and each baseline capture stream, even if unsplit.
    memory.argmax = (1 + len(ROWS)) * 16384 * 8
    for m in ROWS:
        regs, local, shared, blocks = cert_ext.screen_wide_info(w.device.index, m, three_stages)
        print(f"M={m:3} screen: {regs} registers/thread, {local} local bytes/thread, "
              f"{shared // 1024} KiB dynamic shared memory, at most {blocks} CTAs/SM", flush=True)
        x = batches[0][:m].clone()
        out = torch.empty((m, len(w)), dtype=torch.float32, device=w.device)
        ref = torch.empty_like(out)
        status, counts, timed = Counter(), [], []
        with memory.stage(f"M={m} compare and qualify"):
            for i, batch in enumerate(batches):
                reason, count = compare(stack, batch[:m], out, ref, on, v_real, three_stages=three_stages)
                assert reason != cw.INVALID, f"M={m} compaction produced invalid state"
                status[REASON[reason]] += 1
                if reason in (cw.CERTIFIED, cw.TOO_MANY):
                    counts.append(count)
                if reason == cw.CERTIFIED:
                    timed.append((count, i))
            print(f"M={m:3} union {describe(counts)}; {dict(status)}", flush=True)
            assert timed, f"M={m} zero real-head batches certified"
            choices = sorted(timed)
            median, p90 = choices[len(choices) // 2], choices[int(0.9 * (len(choices) - 1))]
            sparse = batches[median[1]][:m]
            assert compare(stack, sparse, out, ref, on, v_real,
                           qualify=True, three_stages=three_stages) == (cw.CERTIFIED, median[0])
            overflow = sparse.clone()
            overflow[0].zero_()
            overflow_reason, overflow_count = compare(stack, overflow, out, ref, on, v_real, three_stages=three_stages)
            assert overflow_reason == cw.TOO_MANY, f"M={m} flat row did not take overflow route"

        with memory.stage(f"M={m} capture"):
            x.copy_(sparse)
            base_graph, base_ids = graph(lambda: baseline(x, entry, ref, v_real))
            cert_graph, cert_ids = graph(lambda: stack.run(x, out, on, v_real, wide.SET, three_stages=three_stages))
            screen_graph, _ = graph(lambda: stack.screen(x, on, v_real, three_stages=three_stages))

        with memory.stage(f"M={m} replay and cold timing"):
            check_replay(stack, base_graph, base_ids, cert_graph, cert_ids, x, out, ref, on,
                         ((sparse, 1, cw.CERTIFIED, median[0]), (overflow, 1, cw.TOO_MANY, overflow_count),
                          (sparse, 0, cw.OFF, 0), (sparse, 1, cw.CERTIFIED, median[0])))
            for label, (count, i) in (("median union", median), ("p90 union", p90)):
                x.copy_(batches[i][:m])
                base = cold_ms(base_graph, flush, reps)
                cert = cold_ms(cert_graph, flush, reps)
                assert stack.state.tolist() == [cw.CERTIFIED, count]
                screen = cold_ms(screen_graph, flush, reps)
                print(f"M={m:3} {label:12} tiles={count:5} wide+stock={base:.3f} ms "
                      f"certwide={cert:.3f} ms screen+select={screen:.3f} ms delta={base - cert:+.3f} ms", flush=True)
            x.copy_(overflow)
            base = cold_ms(base_graph, flush, reps)
            cert = cold_ms(cert_graph, flush, reps)
            assert stack.state.tolist() == [cw.TOO_MANY, overflow_count]
            screen = cold_ms(screen_graph, flush, reps)
            print(f"M={m:3} overflow     tiles={overflow_count:5} wide+stock={base:.3f} ms "
                  f"certwide={cert:.3f} ms screen+select={screen:.3f} ms delta={base - cert:+.3f} ms", flush=True)
        if profile_dir is not None:
            with memory.stage(f"M={m} profile sparse and overflow"):
                profile_routes(profile_dir, stack, x, out, on, v_real, sparse, overflow, three_stages)
        del base_graph, base_ids, cert_graph, cert_ids, screen_graph, ref, out
        torch.cuda.empty_cache()

    print(f"peak accounted benchmark device memory {memory.peak_accounted / 1e9:.3f} GB "
          f"(torch reserved + BF16X + encoder/argmax workspace), "
          f"max synchronized device use {memory.peak_driver / 1e9:.3f} GB "
          f"(initial {memory.start_driver / 1e9:.3f} GB)", flush=True)
    print("certwide GPU check PASS", flush=True)


def main():
    if not __debug__:
        raise RuntimeError("the GPU check requires Python assertions")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("/model"))
    parser.add_argument("--hidden", type=Path, help="directory of captured *.npz")
    parser.add_argument("--synthetic", action="store_true", help="use peaked inputs synthesized against the real head")
    parser.add_argument("--v-real", type=int, default=200058)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--cpu-count", action="store_true")
    parser.add_argument("--fixture-only", action="store_true", help="small GPU correctness fixtures, without model weights")
    parser.add_argument("--build-timeout", type=float, default=900, help="seconds allowed for extension prebuild")
    parser.add_argument("--screen-stages", type=int, choices=(2, 3), default=2, help="activation stages above 128 rows")
    parser.add_argument("--profile-dir", type=Path, help="export per-kernel traces for sparse and overflow paths")
    parser.add_argument("--build-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.build_output is not None:
        build_extensions(args.build_output)
        return
    if args.cpu_count and (args.fixture_only or args.profile_dir is not None):
        parser.error("--fixture-only and --profile-dir require CUDA")
    if args.steps < 1 or args.reps < 20:
        parser.error("require at least one batch and 20 timing repetitions")
    if not np.isfinite(args.build_timeout) or args.build_timeout <= 0:
        parser.error("--build-timeout must be positive and finite")
    if not args.cpu_count and not torch.cuda.is_available():
        parser.error("a CUDA-enabled GB10 and the SGLang image are required")
    device = torch.device("cpu" if args.cpu_count else f"cuda:{torch.cuda.current_device()}")
    if not args.cpu_count:
        if torch.cuda.get_device_capability(device) != (12, 1):
            parser.error("the tensor-core qualification requires SM121")
        print(f"qualifying {torch.cuda.get_device_name(device)}, SM121, torch {torch.__version__}, "
              f"CUDA {torch.version.cuda}, screen stages above 128 rows: {args.screen_stages}", flush=True)
        cert_ext = prebuild(args.build_timeout)
        memory = DeviceMemory(device)
        with memory.stage("qualify small GPU and DSpark fixtures", external_scratch=4 * 2**20):
            wide.EXT.prepare(device.index)
            fixture, sparse = adversarial(cert_ext, device)
            dspark_fixture(fixture, sparse)
            del fixture, sparse
        if args.fixture_only:
            print("certwide GPU fixtures PASS", flush=True)
            return
        torch.cuda.empty_cache()
        with memory.stage("load BF16 model head"):
            w = load_head(args.model, device)
    else:
        w = load_head(args.model, device)
    if w.dtype != torch.bfloat16 or w.shape[1] != cm.K or len(w) % 16 or not 0 < args.v_real <= len(w):
        raise ValueError(f"expected a BF16 [16n, {cm.K}] head and a valid --v-real")
    candidates = (ROOT / "hid", ROOT.parent / "cert" / "hid")
    capture_path = args.hidden or next((p for p in candidates if p.is_dir()), None)
    captures = [] if args.synthetic else load_capture(capture_path)
    if args.hidden and not captures and not args.synthetic:
        parser.error(f"no captures found under {args.hidden}")
    if captures:
        print(f"{sum(len(x) for x in captures)} captured oracle rows from {len(captures)} transcripts, {capture_path}")
        batches = [x.to(device) for x in captured_batches(captures, args.steps)]
    else:
        print("synthetic peaked inputs against the real head; flat and out-of-range inputs are exercised separately")
        batches = list(synthetic_batches(w, args.steps, args.v_real))
    if args.cpu_count:
        cpu_counts(w, batches, args.v_real)
    else:
        gpu_check(w, batches, args.v_real, args.reps, cert_ext, memory, args.screen_stages == 3, args.profile_dir)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
