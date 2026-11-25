"""CPU tests of the certified head under tensor parallelism (certtp.py, certtp_math.py, ops.Cert.run_tp, the TP branch
of patches.py): 4 simulated vocab shards, no GPU, no sglang. From overlay/:
    python3 -m pytest -q inkling_cert/tests/test_certtp.py

The reference is the stock TP pipeline (pico's shards, the stock softcap per shard, the full gather, the padding mask,
the lowest-index argmax). Every certified result must equal it, by both resolutions (the stock gather of the certified
shards, and the stock deferred argmax with the real-vocabulary limit), and every rank must take the same decision.
"""

import math
import os
import sys
import threading
import types
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from inkling_cert import cert_math as cm  # noqa: E402
from inkling_cert import certtp, ops, patches  # noqa: E402
from inkling_cert import certtp_math as ctm  # noqa: E402

K = cm.K
WORLD, N = 4, 768                 # 48 tiles per shard
V = WORLD * N
V_REAL = V - 37                   # the last shard: 731 real ids (not a tile multiple), 37 padded


def head(V, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((V, K)) * (1 + 0.5 * rng.standard_normal((V, 1)) ** 2)
    W[rng.random((V, K)) < 0.002] *= 6
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return cm.bf16_round(W.astype(np.float32))


def toward(W, ids, scale=220.0, noise=0.0, seed=0):
    """Rows x along the sum of the given head rows (one list of ids per row), ||x|| = scale."""
    rng = np.random.default_rng(seed)
    X = np.stack([W[list(i)].astype(np.float64).sum(0) for i in ids])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    if noise:
        X += noise * rng.standard_normal(X.shape) / math.sqrt(K)
    X *= scale / np.linalg.norm(X, axis=1, keepdims=True)
    return cm.bf16_round(X.astype(np.float32))


def peaked(W, M, seed=0, peak=0.25):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((M, K))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    for m in range(M):
        tgt = rng.integers(0, V_REAL, size=2)
        X[m] = X[m] * math.sqrt(1 - 2 * peak ** 2) + peak * W[tgt[0]] + peak * rng.random() * W[tgt[1]]
    X *= 220.0 / np.linalg.norm(X, axis=1, keepdims=True)
    return cm.bf16_round(X.astype(np.float32))


def exact_logits(x, W):
    return (np.asarray(x, np.float64) @ np.asarray(W, np.float64).T).astype(np.float32)


def shards_of(W):
    return [ctm.Shard(W, r, N, V_REAL) for r in range(WORLD)]


# ---- 1. the arithmetic model ------------------------------------------------------------------------------------


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.W = head(V, seed=21)
        cls.shards = shards_of(cls.W)

    def check(self, x, W=None, shards=None, s_err=None, on=None, bad=None, certified=None):
        W = self.W if W is None else W
        shards = self.shards if shards is None else shards
        l = exact_logits(x, W)
        ref = ctm.stock_gathered_token(l, V_REAL)
        # the stock deferral, limited to the real vocabulary, is the stock token (certtp.patch_deferral)
        np.testing.assert_array_equal(ctm.stock_deferred_token(l, V_REAL, WORLD), ref)
        res = ctm.tp_step(x, shards, l, V_REAL, on=on, bad=bad, s_err=s_err)
        np.testing.assert_array_equal(res["gathered"], ref)
        np.testing.assert_array_equal(res["deferred"], ref)
        self.assertEqual(len(set(res["certified"])), 1, "ranks disagree on whether the step is certified")
        if res["certified"][0]:
            for thr in res["thresholds"][1:]:
                np.testing.assert_array_equal(thr, res["thresholds"][0])
        else:  # every rank serves its full shard
            for r, sh in enumerate(shards):
                np.testing.assert_array_equal(res["bufs"][r], l[:, sh.offset:sh.offset + N])
        if certified is not None:
            self.assertEqual(res["certified"][0], certified)
        return res, ref

    def test_peaked_rows(self):
        for seed in range(6):
            for M in (8, 16):
                res, _ = self.check(peaked(self.W, M, seed=seed), certified=True)
                tiles = [int(m.sum()) for m in res["masks"]]
                self.assertLess(max(tiles), N // 16 // 2, tiles)

    def test_winner_on_every_shard_and_at_the_vocab_end(self):
        ids = [[5], [N - 1], [N], [2 * N + 17], [3 * N], [V_REAL - 1], [V_REAL - 16], [3 * N + 700]]
        res, ref = self.check(toward(self.W, ids, scale=55.0), certified=True)
        np.testing.assert_array_equal(ref, [i[0] for i in ids])

    def test_adversarial_screen_error(self):
        """Screen errors at the accumulation term, in the worst direction for each id: the winner low, others high."""
        for seed in (3, 4):
            x = peaked(self.W, 8, seed=seed)
            l = exact_logits(x, self.W)
            w = ctm.stock_gathered_token(l, V_REAL)
            nv = np.concatenate([sh.nv for sh in self.shards]).astype(np.float64)
            xn = np.linalg.norm(x.astype(np.float64), axis=1)
            err = cm.C_ACC * nv[None, :] * xn[:, None] * 0.999
            sign = np.where(np.arange(V)[None, :] == w[:, None], -1.0, 1.0)
            self.check(x, s_err=sign * err)

    def test_near_ties_across_shards(self):
        pairs = [(3, 2 * N + 5), (N + 1, 3 * N + 2), (100, N + 100), (2 * N + 40, V_REAL - 3), (7, 3 * N + 640)]
        for k, (a, b) in enumerate(pairs):
            for noise in (0.0, 1e-3, 1e-2):
                for scale in (60.0, 90.0, 150.0, 400.0):
                    self.check(toward(self.W, [[a, b]] * 8, scale=scale, noise=noise, seed=k))

    def test_exact_ties_across_shards_lowest_id(self):
        W = self.W.copy()
        a = N + 7
        for b in (2 * N + 5, 3 * N + 9, N + 300):  # later shards and the same shard
            W[b] = W[a]
        shards = shards_of(W)
        x = np.repeat(cm.bf16_round((W[a] * 55.0).astype(np.float32))[None], 8, 0)  # logit ~55, as served
        _, ref = self.check(x, W=W, shards=shards, certified=True)
        self.assertTrue((ref == a).all())
        # the tie can also start on a later shard than the lowest id: the lowest rank still wins
        W2 = self.W.copy()
        W2[5] = W2[3 * N + 11]
        x2 = np.repeat(cm.bf16_round((W2[5] * 55.0).astype(np.float32))[None], 8, 0)
        _, ref2 = self.check(x2, W=W2, shards=shards_of(W2), certified=True)
        self.assertTrue((ref2 == 5).all())

    def test_softcap_tie_across_shards_needs_delta(self):
        """A lower id on shard 0 whose raw logit is just below the shard-2 winner's but equal after the softcap: the
        stock token is the lower id. Intervals as tight as the argument allows (lo of the shard-2 id and hi of the
        shard-0 id at the exact logits): only delta(T, H) keeps the shard-0 tile, on another rank than T's."""
        x = peaked(self.W, 1, seed=19)
        l = exact_logits(x, self.W) * np.float32(0.5)
        w, other = 2 * N + 5, 7
        tested = 0
        for big in (45.0, 60.0, 90.0, 120.0):
            l2 = l.copy()
            l2[0, w] = big
            cand = np.float32(big)
            while cm.softcap(np.array([cand]))[0] == cm.softcap(np.array([np.float32(big)]))[0]:
                cand = np.nextafter(cand, np.float32(-np.inf))
            cand = np.nextafter(cand, np.float32(np.inf))
            if cand == np.float32(big):
                continue
            tested += 1
            l2[0, other] = cand
            self.assertEqual(int(ctm.stock_gathered_token(l2, V_REAL)[0]), other)
            lo, hi = l2 - np.float32(1e-3), l2 + np.float32(1e-3)
            lo[0, w], hi[0, other] = l2[0, w], l2[0, other]
            lo[:, V_REAL:] = hi[:, V_REAL:] = -np.inf
            bounds = [(lo[:, r * N:(r + 1) * N], hi[:, r * N:(r + 1) * N]) for r in range(WORLD)]
            res = ctm.tp_step(x, self.shards, l2, V_REAL, bounds=bounds)
            self.assertEqual(set(res["certified"]), {True})
            self.assertEqual(int(res["gathered"][0]), other)
            self.assertEqual(int(res["deferred"][0]), other)
            self.assertTrue(res["masks"][0][other // 16])
        self.assertGreater(tested, 0)

    def test_saturated_softcap_falls_back_everywhere(self):
        """Logits ~3000: outside the range test, every rank computes its whole shard (and softcap ties resolve as
        in the full head)."""
        x = toward(self.W, [[11, 2 * N + 3]] * 4 + [[N + 5]] * 4, scale=3000.0)
        self.check(x, certified=False)

    def test_padding_never_wins_with_negative_logits(self):
        """All real logits negative, padded rows zero (logit 0): the stock mask keeps the padding out, and so do the
        certified head and the real-vocabulary limit; the image's deferral without the limit would pick a padded id."""
        rng = np.random.default_rng(5)
        u = rng.standard_normal(K)
        u /= np.linalg.norm(u)
        W = 0.6 * u[None, :] + 0.8 * rng.standard_normal((V, K)) / math.sqrt(K)
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        W[V_REAL:] = 0.0
        W = cm.bf16_round(W.astype(np.float32))
        x = np.repeat(cm.bf16_round((-40.0 * u).astype(np.float32))[None], 8, 0)
        l = exact_logits(x, W)
        self.assertLess(float(l[:, :V_REAL].max()), 0.0)
        _, ref = self.check(x, W=W, shards=shards_of(W))
        self.assertTrue((ref < V_REAL).all())
        self.assertTrue((ctm.stock_deferred_token(l, V_REAL, WORLD, clamp=False) >= V_REAL).all())

    def test_large_padded_rows_never_raise_the_threshold(self):
        x = peaked(self.W, 8, seed=9)
        W = self.W.copy()
        W[V_REAL:] = cm.bf16_round((x[0] / np.linalg.norm(x[0]) * 3.0).astype(np.float32))  # logits ~660
        res, _ = self.check(x, W=W, shards=shards_of(W), certified=True)
        self.assertLess(int(res["masks"][3].sum()), N // 16)

    def test_one_rank_off_or_non_finite_falls_back_everywhere(self):
        x = peaked(self.W, 8, seed=13)
        for r in range(WORLD):
            on = [True] * WORLD
            on[r] = False
            self.check(x, on=on, certified=False)
            bad = [False] * WORLD
            bad[r] = True
            self.check(x, bad=bad, certified=False)

    def test_nan_from_a_peer_falls_back(self):
        ht = np.zeros((2, N // 16), np.float32)
        recv = np.zeros((WORLD, 2, 2), np.float32)
        recv[2, 1, 0] = np.nan
        mask, certified, _ = ctm.select_tp(ht, recv)
        self.assertFalse(certified)
        self.assertTrue(mask.all())

    def test_global_threshold_is_what_limits_the_work(self):
        """With each shard's own T (no exchange), a shard without the winner computes far more tiles: the reason
        for the (T, H) all-gather (real rows: 2,668 of 3,141 tiles on the slowest rank vs 27, see README)."""
        x = peaked(self.W, 8, seed=17)
        l = exact_logits(x, self.W)
        res = ctm.tp_step(x, self.shards, l, V_REAL)
        local = []
        for sh in self.shards:
            lo, hi = sh.screen(x)
            lt, ht = ctm.tile_max(lo), ctm.tile_max(hi)
            mask, _, _ = ctm.select_tp(ht, ctm.send_th(lt, ht)[None])
            local.append(int(mask.sum()))
        self.assertLessEqual(max(int(m.sum()) for m in res["masks"]), max(local))


# ---- 2. ops.Cert.run_tp with 4 ranks in threads and a barrier all-gather ---------------------------------------


class FakeExt:
    """cert_ext's TP entry points on the numpy model of one shard (the arithmetic of the kernels)."""

    def __init__(self, shard):
        self.sh = shard

    def screen_tp(self, x, q, sc, rt, nv, Vs, v_real, cn, lt, ht, send, on, flags, sdbg, bdbg):
        m = x.shape[0]
        assert v_real == self.sh.v and Vs == N and send.shape == (m, 2)
        if int(on[0]) == 0:
            send.fill_(float("inf"))
            return
        lo, hi = self.sh.screen(x.float().numpy())
        if not (np.isfinite(lo[:, :v_real]).all() and np.isfinite(hi[:, :v_real]).all()):
            flags[1] = 1
        lt_, ht_ = ctm.tile_max(lo), ctm.tile_max(hi)
        lt[:m] = torch.from_numpy(lt_)
        ht[:m] = torch.from_numpy(ht_)
        send.copy_(torch.from_numpy(ctm.send_th(lt_, ht_, True, bool(flags[1]))))

    def select_tp(self, recv, world, m, ht, Vs, on, flags, mask, stats, cap, two_e, dmax):
        assert recv.shape == (world * m, 2) and recv.is_contiguous()
        msk, certified, _ = ctm.select_tp(ht[:m].numpy(), recv.view(world, m, 2).numpy(), bool(on[0]), bool(flags[1]))
        mask.copy_(torch.from_numpy(msk.astype(np.uint8)))
        flags[2] = int(certified)
        stats[1 if certified else (4 if int(on[0]) == 0 else 2)] += 1

    def exact(self, x2, buf, Vs, k, mask, out, wk, flags):
        l = exact_logits(x2.float().numpy(), self.sh.W)
        out.copy_(torch.from_numpy(cm.certified_buffer(l, mask.numpy().astype(bool))))
        flags[1] = 0


def make_cert(shard):
    z16 = torch.zeros((cm.NGROUPS, N), dtype=torch.float16)
    tables = {"q": torch.zeros(1, dtype=torch.uint8), "sc": z16, "rt": z16, "nv": torch.zeros(N), "eps": 0.0}
    pack = types.SimpleNamespace(n=N, k=K, buf=torch.zeros(1, dtype=torch.uint8), nbytes=0)
    return ops.Cert(FakeExt(shard), pack, tables, 4, torch.device("cpu"))


class BarrierGroup:
    """A dim-0 all-gather across threads; ranks that disagree on the collective break the barrier."""

    def __init__(self, world):
        self.world = world
        self.slots = [None] * world
        self.barrier = threading.Barrier(world, timeout=20)
        self.log = [[] for _ in range(world)]

    def gather(self, r, out, inp):
        self.log[r].append(tuple(inp.shape))
        self.slots[r] = inp.clone()
        self.barrier.wait()
        if len({tuple(s.shape) for s in self.slots}) != 1:
            raise AssertionError("ranks disagree on the collective")
        out.copy_(torch.cat(self.slots, 0))
        self.barrier.wait()


def torch_softcap(t, cap):
    t.copy_(torch.from_numpy(cm.softcap(t.numpy())))


class ThreadedRanks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.W = head(V, seed=33)
        cls.shards = shards_of(cls.W)

    def run_ranks(self, x, on=(1, 1, 1, 1), check=False):
        group = BarrierGroup(WORLD)
        certs = [make_cert(sh) for sh in self.shards]
        for r, c in enumerate(certs):
            c.enable_tp(WORLD, r, r * N, lambda out, inp, r=r: group.gather(r, out, inp))
            c.set_on(on[r])
        xt = torch.from_numpy(x).to(torch.bfloat16)
        m = xt.shape[0]
        outs, toks, errors = [None] * WORLD, [None] * WORLD, []

        def rank(r):
            try:
                c, sh = certs[r], self.shards[r]
                out = torch.empty((m, N), dtype=torch.float32)
                if check:
                    full = lambda xx, oo: oo.copy_(torch.from_numpy(exact_logits(xx.float().numpy(), sh.W)))  # noqa
                    c.run_check_tp(xt, out, c.on, sh.v, cm.CAP, full, torch_softcap)
                else:
                    c.run_tp(xt, out, c.on, sh.v, cm.CAP)
                outs[r] = out.clone()
                y = cm.softcap(out.numpy())
                loc = torch.from_numpy(ctm.local_argmax(y, sh.v, sh.offset).astype(np.float32))
                g = torch.empty((WORLD * m, 2), dtype=torch.float32)
                group.gather(r, g, loc)  # the stock deferred_argmax's [M, 2] all-gather
                toks[r] = ctm.merge_argmax(g.view(WORLD, m, 2).numpy().astype(np.float64))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                group.barrier.abort()

        threads = [threading.Thread(target=rank, args=(r,)) for r in range(WORLD)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        self.assertEqual(errors, [])
        for r in range(1, WORLD):
            self.assertEqual(group.log[r], group.log[0])
            np.testing.assert_array_equal(toks[r], toks[0])
        return certs, outs, toks[0], group

    def test_certified_step_equals_stock(self):
        for seed in range(3):
            x = peaked(self.W, 8, seed=seed)
            certs, outs, tok, group = self.run_ranks(x)
            l = exact_logits(x, self.W)
            np.testing.assert_array_equal(tok, ctm.stock_gathered_token(l, V_REAL))
            self.assertEqual({int(c.flags[2]) for c in certs}, {1})
            self.assertEqual(group.log[0], [(8, 2), (8, 2)])   # (T, H), then the deferred argmax: two [M, 2] gathers

    def test_sixteen_rows_and_cross_shard_ties(self):
        x = np.concatenate([peaked(self.W, 8, seed=7), toward(self.W, [[2, 2 * N + 2]] * 8, noise=1e-3)])
        _, _, tok, _ = self.run_ranks(x)
        np.testing.assert_array_equal(tok, ctm.stock_gathered_token(exact_logits(x, self.W), V_REAL))

    def test_one_rank_off_gives_full_shards_everywhere(self):
        x = peaked(self.W, 8, seed=4)
        l = exact_logits(x, self.W)
        certs, outs, tok, _ = self.run_ranks(x, on=(1, 1, 0, 1))
        for r in range(WORLD):
            np.testing.assert_array_equal(outs[r].numpy(), l[:, r * N:(r + 1) * N])
            self.assertEqual(int(certs[r].flags[2]), 0)
        np.testing.assert_array_equal(tok, ctm.stock_gathered_token(l, V_REAL))

    def test_check_mode_counts_no_mismatch(self):
        x = peaked(self.W, 8, seed=5)
        certs, outs, _, group = self.run_ranks(x, check=True)
        l = exact_logits(x, self.W)
        for r, c in enumerate(certs):
            np.testing.assert_array_equal(outs[r].numpy(), l[:, r * N:(r + 1) * N])   # served: pico's full shard
            self.assertEqual(c.stats[5:8].tolist(), [8, 0, 1])                         # 8 rows, 0 mismatches, 1 step
        self.assertEqual(group.log[0], [(8, 2), (8, 4), (8, 2)])


# ---- 3. the patched head: collectives independent of the rank's own outcome ----------------------------------


class Mode:
    def __init__(self, verify):
        self.verify = verify

    def is_target_verify(self):
        return self.verify


class LP:
    shard_softcap = 30.0
    logit_scale = None
    do_tensor_parallel_all_gather = True
    do_tensor_parallel_all_gather_dp_attn = False

    def __init__(self):
        self.config = types.SimpleNamespace(vocab_size=V_REAL)


def fake_pico(weight, pack, W):
    pp = types.ModuleType("inkling_pico12.patches")
    pp.STATE = {"levers": ("lmhead",)}
    pp.HEADS = {weight.data_ptr(): pack}
    po = types.ModuleType("inkling_pico12.ops")
    po.SET = types.SimpleNamespace(wk=4)
    po.rows_of = lambda x: x.numel() // x.shape[-1]
    po.eligible = lambda x, p: 0 < po.rows_of(x) <= 16
    po.out_ok = lambda out, m, p, dt, dev: tuple(out.shape) == (m, p.n)
    po.as_rows = lambda x, k: x.reshape(-1, k)
    po.gemv = lambda x, p, out: out.copy_(torch.from_numpy(exact_logits(x.float().numpy(), W)))
    return pp, po


class TpHeadTest(unittest.TestCase):
    """One rank at a time; the (recorded) collectives of every rank-local outcome must be the same list."""

    @classmethod
    def setUpClass(cls):
        cls.W = head(V, seed=41)
        cls.shards = shards_of(cls.W)

    def setUp(self):
        self.saved_mods = {k: sys.modules.get(k) for k in ("inkling_pico12.patches", "inkling_pico12.ops")}
        self.saved_state = dict(patches.STATE)
        self.saved_cap, self.saved_fns = patches._capturing, list(patches._FNS)
        patches._FNS[:] = [torch_softcap, lambda t: t.argmax(1)]

    def tearDown(self):
        for k, v in self.saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        patches.STATE.clear()
        patches.STATE.update(self.saved_state)
        patches._capturing = self.saved_cap
        patches._FNS[:] = self.saved_fns

    def rank_calls(self, r, outcome, mode, capturing, rows=8, verify=True):
        """Collectives (input shapes) of one head call on rank r with outcome eligible | disabled | no_pack | bypass
        (SGLang's dispatch never calls our head function, e.g. sm12x_fp32_lm_head_applies false -> torch.mm)."""
        sh = self.shards[r]
        w = torch.from_numpy(sh.W).to(torch.bfloat16)
        cert = make_cert(sh)
        pack = cert.pack
        calls = []

        def gather(out, inp):
            calls.append(tuple(inp.shape))
            out.view(WORLD, inp.shape[0], -1).copy_(inp.unsqueeze(0).expand(WORLD, *inp.shape))

        cert.enable_tp(WORLD, r, r * N, gather)
        cert.set_on(1)
        pp, po = fake_pico(w if outcome != "no_pack" else torch.zeros(1), pack, sh.W)
        sys.modules["inkling_pico12.patches"], sys.modules["inkling_pico12.ops"] = pp, po
        info = {"world": WORLD, "rank": r, "n": N, "offset": r * N, "v_real": sh.v, "real": V_REAL}
        patches.STATE.update(mode=mode, cfg={"log_every": 0, "cache_dir": "/tmp", "tp": True}, cert=cert, tp=info,
                             disabled="test" if outcome == "disabled" else None, log=lambda m: None)
        patches._capturing = lambda x: capturing
        inner_calls = []

        def inner(x, weight, out):
            inner_calls.append(x.shape[0])
            return torch.from_numpy(exact_logits(x.float().numpy(), sh.W))

        inner._pico12 = True
        head_fn = patches.make_lm_head(inner)

        def compute(lp, h, lm, embedding_bias=None, *, out=None):  # Inkling2LogitsProcessor._compute_lm_head
            if outcome == "bypass":
                return torch.from_numpy(exact_logits(h.float().numpy(), sh.W))
            return head_fn(h, w, out)

        compute_fn = patches.make_compute_lm_head(compute)
        get_logits = patches.make_get_logits(lambda lp, h, lm, meta: compute_fn(lp, h, lm, None, out=None))
        x = torch.from_numpy(peaked(self.W, rows, seed=r)).to(torch.bfloat16)
        out = get_logits(LP(), x, None, types.SimpleNamespace(forward_mode=Mode(verify)))
        return calls, inner_calls, out, cert

    def test_collectives_do_not_depend_on_the_rank_outcome(self):
        for mode, want in (("1", [(8, 2)]), ("check", [(8, 2), (8, 4)])):
            for r in range(WORLD):
                for outcome in ("eligible", "disabled", "no_pack", "bypass"):
                    calls, inner_calls, out, _ = self.rank_calls(r, outcome, mode, capturing=True)
                    self.assertEqual(calls, want, (mode, r, outcome))
                    self.assertEqual(inner_calls, [8] if outcome in ("disabled", "no_pack") else [], (mode, r, outcome))
                    self.assertIsNone(patches.CTX["tp_gather"])

    def test_no_collective_outside_capture_or_above_16_rows_or_off_verify(self):
        for outcome in ("eligible", "disabled", "no_pack", "bypass"):
            calls, _, _, _ = self.rank_calls(1, outcome, "1", capturing=False)
            self.assertEqual(calls, [])
            calls, _, _, _ = self.rank_calls(1, outcome, "1", capturing=True, rows=24)
            self.assertEqual(calls, [])
            calls, _, _, _ = self.rank_calls(1, outcome, "1", capturing=True, verify=False)
            self.assertEqual(calls, [])

    def test_eager_first_call_warms_without_collective(self):
        calls, inner_calls, out, cert = self.rank_calls(2, "eligible", "1", capturing=False)
        self.assertEqual((calls, inner_calls), ([], []))  # the warm call ran the kernels (full, local exchange)
        l = exact_logits(peaked(self.W, 8, seed=2), self.shards[2].W)
        np.testing.assert_array_equal(out.numpy(), l)

    def test_head_reached_outside_the_hook_never_gathers(self):
        """A captured verify head call that make_compute_lm_head does not own (no collective owed) serves pico."""
        sh = self.shards[0]
        w = torch.from_numpy(sh.W).to(torch.bfloat16)
        calls, inner_calls, _, cert = self.rank_calls(0, "eligible", "1", capturing=True)
        inner = lambda x, weight, out: inner_calls.append("pico") or torch.zeros((x.shape[0], N))  # noqa: E731
        inner._pico12 = True
        head_fn = patches.make_lm_head(inner)
        get_logits = patches.make_get_logits(lambda lp, h, lm, meta: head_fn(h, w, None))
        x = torch.from_numpy(peaked(self.W, 8, seed=0)).to(torch.bfloat16)
        get_logits(LP(), x, None, types.SimpleNamespace(forward_mode=Mode(True)))
        self.assertEqual(len(calls), 1)            # only the first (hooked) call gathered
        self.assertEqual(inner_calls[-1], "pico")

    def test_captured_certified_output(self):
        calls, inner_calls, out, cert = self.rank_calls(3, "eligible", "1", capturing=True)
        self.assertEqual(int(cert.flags[2]), 1)  # with every "peer" equal to this rank: certified on the shard
        self.assertTrue(bool(torch.isinf(out).any()))


# ---- 4. the deferral's real-vocabulary limit -------------------------------------------------------------------


class DeferralLimitTest(unittest.TestCase):
    def make(self, rank, real, padded, width):
        calls = []

        def softcap_row_argmax(logits, cap, valid_width, col_offset):
            calls.append((valid_width, col_offset))
            z = logits[:, :valid_width]
            v, i = z.max(dim=1)
            return torch.stack([v, (i + col_offset).float()], 1)

        lp_mod = types.SimpleNamespace(
            softcap_row_argmax=softcap_row_argmax,
            get_parallel=lambda: types.SimpleNamespace(tp_group=types.SimpleNamespace(rank_in_group=rank)))

        class StockLP:
            final_logit_softcapping = None

            def __init__(self):
                self.vocab_size = padded
                self.config = types.SimpleNamespace(vocab_size=real)

            def deferred_local_argmax(self, logits):  # the image's: padded width
                group = lp_mod.get_parallel().tp_group
                shard_width = logits.shape[1]
                col_offset = group.rank_in_group * shard_width
                valid_width = max(0, min(shard_width, self.vocab_size - col_offset))
                return lp_mod.softcap_row_argmax(logits, self.final_logit_softcapping, valid_width, col_offset)

            def finish_deferred_gather(self, logits):
                return logits.repeat(1, WORLD)[:, : self.vocab_size].float()

        self.assertTrue(certtp.patch_deferral(StockLP, lp_mod, lambda m: None))
        self.assertFalse(certtp.patch_deferral(StockLP, lp_mod, lambda m: None))  # idempotent
        return StockLP(), calls

    def test_local_argmax_stops_at_the_real_vocab(self):
        lp, calls = self.make(rank=3, real=V_REAL, padded=V, width=N)
        logits = torch.full((2, N), -5.0)
        logits[:, V_REAL - 3 * N] = 1.0   # first padded id of the last shard: the image's width would pick it
        logits[:, 10] = 0.5
        out = lp.deferred_local_argmax(logits)
        self.assertEqual(calls[-1], (V_REAL - 3 * N, 3 * N))
        self.assertEqual(out[:, 1].tolist(), [3 * N + 10] * 2)
        lp0, calls0 = self.make(rank=0, real=V_REAL, padded=V, width=N)
        lp0.deferred_local_argmax(torch.zeros((1, N)))
        self.assertEqual(calls0[-1], (N, 0))

    def test_full_gather_masks_the_padding(self):
        lp, _ = self.make(rank=0, real=V_REAL, padded=V, width=N)
        out = lp.finish_deferred_gather(torch.ones((2, N)))
        self.assertEqual(out.shape, (2, V))
        self.assertTrue(bool(torch.isinf(out[:, V_REAL:]).all()) and bool((out[:, :V_REAL] == 1).all()))

    def test_unpadded_vocab_full_width(self):
        lp, calls = self.make(rank=1, real=V, padded=V, width=N)
        lp.deferred_local_argmax(torch.zeros((1, N)))
        self.assertEqual(calls[-1], (N, N))


# ---- 5. prepare(): TP arming is agreed by every rank -----------------------------------------------------------


class PrepareTpTest(unittest.TestCase):
    """Every rank reaches certtp.agree (the real function; torch.distributed.all_reduce simulates the peers), whatever
    its local outcome, and all ranks arm or disarm together."""

    def setUp(self):
        import torch.distributed as dist

        self.dist = dist
        self.keys = ("sglang", "sglang.srt", "sglang.srt.runtime_context", "inkling_pico12.patches",
                     "inkling_pico12.ops", patches.UNQUANT)
        self.saved_mods = {k: sys.modules.get(k) for k in self.keys}
        self.saved_state = dict(patches.STATE)
        self.saved_ar, self.saved_load, self.saved_fns = dist.all_reduce, patches.build.load, list(patches._FNS)
        self.saved_env = {k: os.environ.get(k) for k in patches.OBSERVER_ENVS + ("SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER",)}
        for k in self.saved_env:
            os.environ.pop(k, None)
        patches._FNS[:] = [torch_softcap, lambda t: t.argmax(1)]
        self.logs = []
        self.W = head(V, seed=51)

    def tearDown(self):
        for k, v in self.saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k, v in self.saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        patches.STATE.clear()
        patches.STATE.update(self.saved_state)
        self.dist.all_reduce, patches.build.load = self.saved_ar, self.saved_load
        patches._FNS[:] = self.saved_fns

    def prepare(self, r, peers=None, start=None, cfg_tp=True, mode="1", load=None, pico_coded=True):
        """peers: (ok, mode, defer) of the other 3 ranks (default: prepared, mode 1, no deferral). Returns the
        vectors this rank contributed to the agreement (empty: it never voted)."""
        sh = ctm.Shard(self.W, r, N, V_REAL)
        w = torch.from_numpy(sh.W).to(torch.bfloat16)
        group = types.SimpleNamespace(world_size=WORLD, rank_in_group=r, cpu_group="gloo",
                                      all_gather_into_tensor=lambda out, inp: None)
        rc = types.ModuleType("sglang.srt.runtime_context")
        rc.get_parallel = lambda: types.SimpleNamespace(tp_size=WORLD, tp_group=group)
        sys.modules.update({"sglang": types.ModuleType("sglang"), "sglang.srt": types.ModuleType("sglang.srt"),
                            "sglang.srt.runtime_context": rc})
        pack = types.SimpleNamespace(n=N, k=K, buf=torch.zeros(1, dtype=torch.uint8), nbytes=0)
        pp, po = fake_pico(w if pico_coded else torch.zeros(1), pack, sh.W)
        sys.modules["inkling_pico12.patches"], sys.modules["inkling_pico12.ops"] = pp, po
        unq = types.ModuleType(patches.UNQUANT)
        unq.sm12x_fp32_lm_head = lambda *a: None
        unq.sm12x_fp32_lm_head._inkling_cert = True
        sys.modules[patches.UNQUANT] = unq
        peers = [(1, 1, 0)] * (WORLD - 1) if peers is None else peers
        sent = []

        def all_reduce(t, op=None, group=None):
            assert op == self.dist.ReduceOp.MIN and group == "gloo"
            sent.append(t.tolist())
            for ok, m, d in peers:
                t.copy_(torch.minimum(t, torch.tensor([ok, m, -m, d, -d], dtype=t.dtype)))

        self.dist.all_reduce = all_reduce
        patches.build.load = load or (lambda cache: FakeExt(sh))
        idx = types.SimpleNamespace(org_vocab_start_index=r * N if start is None else start, num_org_elements=N,
                                    num_added_elements_padded=0)
        lm_head = types.SimpleNamespace(weight=w, tp_size=WORLD, shard_indices=idx)
        lp = types.SimpleNamespace(use_fp32_lm_head=True, do_tensor_parallel_all_gather=True,
                                   do_tensor_parallel_all_gather_dp_attn=False, use_attn_tp_group=False,
                                   use_tp_lm_head_all_to_all=False, final_logit_softcapping=None, logit_scale=None,
                                   vocab_size=V, config=types.SimpleNamespace(vocab_size=V_REAL))
        model = types.SimpleNamespace(llm=types.SimpleNamespace(lm_head=lm_head, logits_processor=lp))
        patches.STATE.update(mode=mode, cfg={"log_every": 0, "cache_dir": "/tmp", "tp": cfg_tp, "wide": False},
                             cert=None, tp=None, disabled=None, log=self.logs.append)
        patches.prepare(model)
        return sent

    def disarmed(self, text):
        self.assertIsNone(patches.STATE["tp"])
        self.assertIsNone(patches.STATE["cert"])
        self.assertIn(text, patches.STATE["disabled"])

    def test_armed_when_every_rank_agrees(self):
        for r in range(WORLD):
            sent = self.prepare(r)
            self.assertEqual(sent, [[1, 1, -1, 0, 0]])
            tp = patches.STATE["tp"]
            self.assertEqual((tp["rank"], tp["offset"], tp["v_real"]), (r, r * N, min(N, V_REAL - r * N)))
            self.assertIsNotNone(patches.STATE["cert"])
            self.assertEqual(patches.STATE["cert"].world, WORLD)
            self.assertIsNone(patches.STATE["disabled"])
            self.assertEqual(int(patches.STATE["cert"].stats.sum()), 0)

    def test_a_peer_that_failed_disarms_every_rank(self):
        self.assertEqual(self.prepare(1, peers=[(1, 1, 0), (0, 1, 0), (1, 1, 0)]), [[1, 1, -1, 0, 0]])
        self.disarmed("a rank could not prepare its shard")

    def test_peers_with_another_mode_or_deferral_disarm(self):
        self.prepare(1, peers=[(1, 1, 0), (1, 2, 0), (1, 1, 0)])
        self.disarmed("INKLING_CERT_HEAD")
        os.environ["SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER"] = "1"
        self.assertEqual(self.prepare(2, mode="check", peers=[(1, 2, 1), (1, 2, 0), (1, 2, 1)]), [[1, 2, -2, 1, -1]])
        self.disarmed("SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER")

    def test_layout_refusal_still_votes(self):
        self.assertEqual(self.prepare(2, start=0), [[0, 1, -1, 0, 0]])
        self.disarmed("vocab shard layout")

    def test_local_build_or_pico_failure_votes_no_instead_of_exiting(self):
        def broken(cache):
            raise RuntimeError("nvcc failed")

        self.assertEqual(self.prepare(3, load=broken), [[0, 1, -1, 0, 0]])
        self.disarmed("local preparation failed")
        self.assertEqual(self.prepare(0, pico_coded=False), [[0, 1, -1, 0, 0]])  # a CertFatal, also caught
        self.disarmed("not coded the head shard")

    def test_observer_flag_votes_no(self):
        os.environ[patches.OBSERVER_ENVS[0]] = "1"
        self.assertEqual(self.prepare(1), [[0, 1, -1, 0, 0]])
        self.disarmed(patches.OBSERVER_ENVS[0])

    def test_without_inkling_cert_tp_the_old_disable(self):
        self.assertEqual(self.prepare(0, cfg_tp=False), [])
        self.assertEqual(patches.STATE["disabled"], f"tensor parallel size {WORLD} (TP1 only)")

    def test_config_parse(self):
        import inkling_cert

        self.assertTrue(inkling_cert.config_from_env({"INKLING_CERT_TP": "1"})["tp"])
        self.assertFalse(inkling_cert.config_from_env({})["tp"])
        with self.assertRaises(RuntimeError):
            inkling_cert.config_from_env({"INKLING_CERT_TP": "check"})


# ---- 6. NaN: the deferred argmax equals torch.argmax over the gathered row --------------------------------------


class DeferredNanTest(unittest.TestCase):
    """The stock non-deferred accept is torch.argmax over the gathered, padding-masked row: the first NaN wins. The
    patched deferral (local first NaN, NaN-first merge) must give the same id, NaNs anywhere, ties included."""

    def lp_for(self, rank, gathered_locals=None):
        def softcap_row_argmax(logits, cap, valid_width, col_offset):  # the image kernel: NaNs never win
            z = logits[:, :valid_width].clone()
            z[torch.isnan(z)] = float("-inf")
            v, i = z.max(dim=1)
            return torch.stack([v, (i + col_offset).float()], 1)

        def all_gather_into_tensor(out, inp):
            out.copy_(torch.cat(gathered_locals, 0))

        group = types.SimpleNamespace(rank_in_group=rank, world_size=WORLD, all_gather_into_tensor=all_gather_into_tensor)
        lp_mod = types.SimpleNamespace(softcap_row_argmax=softcap_row_argmax,
                                       get_parallel=lambda: types.SimpleNamespace(tp_group=group))

        class LPC:
            final_logit_softcapping = None
            vocab_size = V
            config = types.SimpleNamespace(vocab_size=V_REAL)

            def deferred_local_argmax(self, logits):
                raise AssertionError("replaced")

            def deferred_argmax(self, logits):
                raise AssertionError("replaced")

            def finish_deferred_gather(self, logits):
                return logits

        certtp.patch_deferral(LPC, lp_mod, lambda m: None)
        return LPC()

    def deferred(self, full):
        locals_ = [self.lp_for(r).deferred_local_argmax(full[:, r * N:(r + 1) * N].contiguous()) for r in range(WORLD)]
        ids = self.lp_for(0, locals_).deferred_argmax(full[:, :N].contiguous())
        np.testing.assert_array_equal(ids.numpy(), certtp.merge_shards(torch.stack(locals_)).numpy())
        return ids

    def stock(self, full):
        y = full.clone()
        y[:, V_REAL:] = float("-inf")
        return torch.argmax(y, dim=-1)

    def test_nan_placements(self):
        g = torch.Generator().manual_seed(3)
        cases = [[], [(1, 5)], [(3, 2)], [(0, 1), (2, 9)], [(3, 700)], [(2, 0), (2, 1)], [(3, 740)]]  # (rank, col)
        for nan_at in cases:
            for tie in (False, True):
                full = torch.randn((4, V), generator=g)
                if tie:  # an exact tie of the maximum across shards (lowest id must win)
                    full[:, 2 * N + 3] = full[:, 10] = 50.0
                for row in range(4):
                    for r, c in nan_at[row % max(1, len(nan_at)):] if nan_at else []:
                        full[row, r * N + c] = float("nan")
                full[3, V_REAL + 2] = float("nan")  # padding: masked by the stock path, outside the real width here
                np.testing.assert_array_equal(self.deferred(full).numpy(), self.stock(full).numpy(), str(nan_at))

    def test_all_nan_row_and_all_minus_inf_row(self):
        full = torch.randn((2, V))
        full[0] = float("nan")
        full[1] = float("-inf")
        np.testing.assert_array_equal(self.deferred(full).numpy(), self.stock(full).numpy())


if __name__ == "__main__":
    unittest.main()
