"""CPU tests of inkling_dfbytes' hot plane under TP: 4 simulated vocab shards against the TP1 hot8 / hot16 result.

Run: uv run --no-project --python 3.12 --with torch --with numpy python tests/dfbytes/test_dfbytes_tp.py

The shard layout is SGLang's VocabParallelEmbedding._get_indices (copied below for the org-vocab case). On CPU the
overlay's hot_logits runs inkling_draftbytes.quant.linear_ref (the FP8 spec the Triton kernel implements), and
radix_topk is torch.topk (the image's own CPU fallback), so the TP1 reference and the shards use the overlay's code.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, OVERLAY)

import torch  # noqa: E402

import inkling_dfbytes as dfb  # noqa: E402
from inkling_dfbytes import patches as p  # noqa: E402

R2_FILE = os.path.join(OVERLAY, "inkling_draftbytes", "hotvocab", "inkling2-mini-hot-r2.json")
K = 16


def pad_vocab_size(v, to):
    return ((v + to - 1) // to) * to


class ShardIndices(types.SimpleNamespace):
    @property
    def num_org_elements(self):
        return self.org_vocab_end_index - self.org_vocab_start_index


class FakeHead:
    """The attributes of SGLang's ParallelLMHead the overlay reads (no added vocab)."""

    def __init__(self, full_weight, org_vocab_size, tp_size, tp_rank, padding=64, group=None):
        self.tp_size = tp_size
        self.org_vocab_size = org_vocab_size
        self.num_embeddings = org_vocab_size
        if pad_vocab_size(org_vocab_size, padding) % tp_size:
            padding *= tp_size
        self.org_vocab_size_padded = pad_vocab_size(org_vocab_size, padding)
        self.num_embeddings_padded = pad_vocab_size(self.org_vocab_size_padded, padding)
        self.shard_indices = self._get_indices(self.num_embeddings_padded, self.org_vocab_size_padded,
                                               self.num_embeddings, self.org_vocab_size, tp_rank, tp_size)
        per = self.num_embeddings_padded // tp_size
        self.weight = None if full_weight is None else full_weight[tp_rank * per:(tp_rank + 1) * per]
        if tp_size > 1:
            self.tp_group = group

    @classmethod
    def _get_indices(cls, vocab_size_padded, org_vocab_size_padded, vocab_size, org_vocab_size, tp_rank, tp_size):
        per = org_vocab_size_padded // tp_size  # vocab_range_from_global_vocab_size
        start, end = tp_rank * per, (tp_rank + 1) * per
        return ShardIndices(org_vocab_start_index=min(start, org_vocab_size),
                            org_vocab_end_index=min(end, org_vocab_size))


class Group:
    """A TP group over threads: all_gather concatenates every rank's tensor in rank order; every call is logged."""

    def __init__(self, world):
        self.world = world
        self.barrier = threading.Barrier(world)
        self.slots = [None] * world
        self.logs = [[] for _ in range(world)]

    def view(self, rank):
        group = self

        class View:
            def all_gather(self, t, dim=-1):
                group.logs[rank].append(("all_gather", t.dtype, tuple(t.shape)))
                group.slots[rank] = t.clone()
                group.barrier.wait()
                out = torch.cat(group.slots, dim=dim)
                group.barrier.wait()
                return out

        return View()


def write_hot(ids, tmp):
    path = os.path.join(tmp, "hot.json")
    with open(path, "w") as f:
        json.dump([int(i) for i in ids], f)
    return path


def model_ns(valid):
    return types.SimpleNamespace(valid_vocab_size=valid, candidate_selector=types.SimpleNamespace(top_k=K))


def tp1_topk(hot, hidden):
    """The TP1 hot8 computation of patches.compute_candidates (hot_logits -> radix_topk -> ids)."""
    logits = p.hot_logits(hidden.to(torch.bfloat16), hot.pack)
    vals, idx = torch.topk(logits, K, dim=-1)
    return hot.ids64.index_select(0, idx.reshape(-1).long()).view(idx.shape), vals, logits


def tp_topk(planes, hidden):
    """Each shard's hot_local_topk, the rank-major K-wide gather, then gather_topk's merge (NCCL branch)."""
    local = [p.hot_local_topk(hidden, hot, K, lambda s, k: torch.topk(s, k, dim=-1)) for hot in planes]
    for vals, ids in local:
        assert tuple(vals.shape) == tuple(ids.shape) == (hidden.shape[0], K)  # exactly K per rank
    gathered_vals = torch.cat([v.float() for v, _ in local], dim=-1)
    gathered_ids = torch.cat([i for _, i in local], dim=-1)
    top_vals, sel = torch.topk(gathered_vals, K, dim=-1)
    return torch.gather(gathered_ids, -1, sel).long(), top_vals


class Base(unittest.TestCase):
    opts = ("hot8",)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.saved = dict(dfb.STATE)
        dfb.STATE.update(opts=self.opts, hot_n=0, hot_file=None, hot_built=0)
        self.saved_log, p._log = p._log, (lambda msg: None)
        p._draftbytes()  # import inkling_draftbytes once (CPU)

    def tearDown(self):
        p._log = self.saved_log
        dfb.STATE.clear()
        dfb.STATE.update(self.saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, weight, vocab, valid, ranked, hot_n, tp=4):
        dfb.STATE.update(hot_file=write_hot(ranked, self.tmp), hot_n=hot_n)
        model = model_ns(valid)
        tp1 = p.build_hot(model, FakeHead(weight, vocab, 1, 0))
        planes = [p.build_hot(model, FakeHead(weight, vocab, tp, r)) for r in range(tp)]
        return tp1, planes

    def check_equal(self, tp1, planes, hidden):
        ids1, vals1, logits1 = tp1_topk(tp1, hidden)
        ids4, vals4 = tp_topk(planes, hidden)
        # values: identical (the TP1 vals are BF16; the gathered ones the same values in FP32)
        self.assertTrue(torch.equal(vals1.float(), vals4), (vals1, vals4))
        self.assertTrue(torch.isfinite(vals4).all())  # no pad row ever wins
        pos = {int(t): j for j, t in enumerate(tp1.ids64.tolist())}
        for row in range(hidden.shape[0]):
            got = ids4[row].tolist()
            self.assertEqual(len(set(got)), K)
            # every id is a TP1 hot id whose TP1 plane logit is the reported value (ties may pick other ids)
            for t, v in zip(got, vals4[row].tolist()):
                self.assertIn(t, pos)
                self.assertEqual(float(logits1[row, pos[t]]), v)
            # no tie at the K-th place: the same id set
            srt = torch.sort(logits1[row].float(), descending=True).values
            if len(srt) == K or srt[K - 1] > srt[K]:
                self.assertEqual(set(got), set(ids1[row].tolist()))
        return ids1, ids4


class R2Split(Base):
    def test_r2_split_inkling_shapes(self):
        """The r2 hot list on Inkling-2-Mini's head (201,024 padded, 200,058 real) at TP4."""
        from inkling_draftbytes import patches as dbp

        limit = min(201024, 200058)
        ids = dbp.load_hot_ids(R2_FILE, 65536, limit)
        self.assertEqual(len(ids), 65536)
        heads = [FakeHead(None, 201024, 4, r) for r in range(4)]
        got = [p.shard_ranges(h, limit) for h in heads]
        ranges = got[0][0]
        self.assertTrue(all(g[0] == ranges for g in got))  # every rank derives the same list
        self.assertEqual([g[1] for g in got], [0, 1, 2, 3])
        self.assertEqual(ranges, [(0, 50256), (50256, 100512), (100512, 150768), (150768, 200058)])
        splits = [p.split_hot(ids, ranges, r, K) for r in range(4)]
        self.assertEqual(splits[0][1], [18769, 12739, 9438, 24590])
        self.assertTrue(all(s[2] == 0 for s in splits))  # no shard below K: no pad
        union = sorted(t for s in splits for t in s[0])
        self.assertEqual(union, ids)  # the shards' union is the TP1 hot set


class Equivalence(Base):
    # padded storage 4032 at TP4 (4 x 1008); the last shard [3024, 4032) is clamped to [3024, 3990): 966 real rows
    V, VALID, H, ROWS = 4000, 3990, 128, 4032

    def weight(self, seed):
        g = torch.Generator().manual_seed(seed)
        w = torch.randn((self.ROWS, self.H), generator=g) * (0.5 + torch.rand((self.ROWS, 1), generator=g))
        return w.to(torch.bfloat16)

    def hidden(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        return (torch.randn((n, self.H), generator=g) * 3).to(torch.bfloat16)

    def test_random_hot_sets(self):
        for seed in range(4):
            g = torch.Generator().manual_seed(100 + seed)
            ranked = torch.randperm(self.V, generator=g)[:2500].tolist()  # includes ids >= VALID (dropped)
            tp1, planes = self.build(self.weight(seed), self.V, self.VALID, ranked, 1500)
            self.assertEqual(sum(h.n_real for h in planes), tp1.n)
            self.assertTrue(all(h.pad == 0 for h in planes))
            self.assertTrue(all(int(i) < self.VALID for i in tp1.ids64))
            for n in (1, 7, 56):
                self.check_equal(tp1, planes, self.hidden(n, seed * 10 + n))

    def test_fp8_rows_identical_to_tp1(self):
        """Per-row scales: each shard row's FP8 bytes and scale equal the TP1 plane's for the same id."""
        ranked = list(range(0, self.V, 3))
        tp1, planes = self.build(self.weight(7), self.V, self.VALID, ranked, 1300)
        pos = {int(t): j for j, t in enumerate(tp1.ids64.tolist())}
        for hot in planes:
            for j in range(hot.n_real):
                i = pos[int(hot.ids64[j])]
                self.assertTrue(torch.equal(hot.pack.q[j].view(torch.uint8), tp1.pack.q[i].view(torch.uint8)))
                self.assertTrue(torch.equal(hot.pack.scale[j], tp1.pack.scale[i]))

    def test_short_shard_refused_on_every_rank(self):
        """A shard with fewer than K hot rows: every rank refuses alike (counts come from the head's config, so the
        refusal is rank-invariant) instead of padding with sentinel rows that could tie with real -inf scores."""
        # shard 1 [1008, 2016): no hot id; shard 2 [2016, 3024): 5; shards 0 and 3: many
        ranked = [2020, 2400, 2600, 3000, 3023] + list(range(0, 1008, 2)) + list(range(3024, self.V))
        with self.assertRaises(dfb.DfbytesFatal):
            self.build(self.weight(3), self.V, self.VALID, ranked, 2000)
        for ranked in (list(range(16)), [1, 2, 3, 4, 1100, 1101, 1102, 1103, 2100, 2101, 2102, 2103, 3500, 3501, 3502,
                                         3503]):
            dfb.STATE.update(hot_file=write_hot(ranked, self.tmp), hot_n=16)
            from inkling_draftbytes import patches as dbp
            from inkling_draftbytes import _opts, quant

            ids = dbp.load_hot_ids(dfb.STATE["hot_file"], 16, self.VALID)
            w = self.weight(11)
            for r in range(4):
                with self.assertRaises(dfb.DfbytesFatal):
                    p._build_hot_tp(FakeHead(w, self.V, 4, r), FakeHead(w, self.V, 4, r).weight, ids, self.VALID,
                                    K, quant, _opts.parse_mode("row", "row"))


class Hot16(Equivalence):
    opts = ("hot16",)

    def test_fp8_rows_identical_to_tp1(self):
        """hot16: each shard row is the TP1 plane's BF16 row."""
        ranked = list(range(0, self.V, 3))
        tp1, planes = self.build(self.weight(7), self.V, self.VALID, ranked, 1300)
        pos = {int(t): j for j, t in enumerate(tp1.ids64.tolist())}
        for hot in planes:
            idx = torch.tensor([pos[int(t)] for t in hot.ids64[:hot.n_real]], dtype=torch.long)
            self.assertTrue(torch.equal(hot.pack.q[:hot.n_real], tp1.pack.q.index_select(0, idx)))


class Wiring(Base):
    """The patched compute_candidates on 4 threads (one per rank) against TP1: same output on every rank, and the
    collectives each rank issued are the stock tail's (two all-gathers: fp32 vals then int64 ids, [N, K] each)."""

    V, VALID, H, ROWS = 4000, 3990, 128, 4032

    def make_classes(self):
        class Selector:
            top_k = K

        class Sel:
            lm_head = None
            merge_candidates_fused = False
            draft_config = types.SimpleNamespace(output_multiplier=0.5, final_logit_softcapping=30.0)

            def __init__(self):
                self.candidate_selector = Selector()

            def load_weights(self, weights):
                return None

            def compute_candidates(self, hidden):
                raise AssertionError("stock path taken")

            def _transform_unary_logits(self, logits):  # the image's transform
                logits = logits.float()
                logits.mul_(self.draft_config.output_multiplier)
                cap = self.draft_config.final_logit_softcapping
                return logits.div_(cap).tanh_().mul_(cap)

        fused_calls = []
        mod = types.SimpleNamespace(
            DFlashDraftModel=Sel, DFlash2DraftModel=Sel, radix_topk=lambda s, k: torch.topk(s, k, dim=-1),
            _fused_candidate_all_gather=lambda g, v, i: fused_calls.append(tuple(v.shape)),  # None: NCCL path
            get_parallel=lambda: types.SimpleNamespace(tp_size=4))
        p.patch_dflash(mod)
        return Sel, fused_calls

    def test_threads(self):
        from inkling_draftbytes import quant

        Sel, fused_calls = self.make_classes()
        g = torch.Generator().manual_seed(5)
        w = (torch.randn((self.ROWS, self.H), generator=g)).to(torch.bfloat16)
        ranked = torch.randperm(self.V, generator=g)[:2500].tolist()
        dfb.STATE.update(hot_file=write_hot(ranked, self.tmp), hot_n=1500)
        group = Group(4)
        models = []
        for r in range(5):  # 0..3: TP4 ranks; 4: TP1
            m = Sel()
            m.valid_vocab_size = self.VALID
            m.lm_head = FakeHead(w, self.V, 1, 0) if r == 4 else FakeHead(w, self.V, 4, r, group=group.view(r))
            m._idf_hot = p.build_hot(m, m.lm_head)  # pre-built (the worker does it before capture)
            models.append(m)
        hidden = (torch.randn((24, self.H), generator=g) * 3).to(torch.bfloat16)
        out = [None] * 4
        errors = []

        def run(r):
            try:
                out[r] = models[r].compute_candidates(hidden)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                group.barrier.abort()

        orig_cuda, orig_logits = torch.Tensor.is_cuda, p.hot_logits
        try:
            torch.Tensor.is_cuda = property(lambda self: True)  # the hot path is CUDA-gated
            p.hot_logits = lambda h, pack: quant.linear_ref(h, pack)  # w8_linear's CPU spec, without its CUDA gate
            threads = [threading.Thread(target=run, args=(r,)) for r in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
            ref_ids, ref_vals = models[4].compute_candidates(hidden)
        finally:
            torch.Tensor.is_cuda = orig_cuda
            p.hot_logits = orig_logits
        self.assertEqual(errors, [])
        stock_log = [("all_gather", torch.float32, (24, K)), ("all_gather", torch.int64, (24, K))]
        self.assertEqual(group.logs, [stock_log] * 4)
        self.assertEqual(fused_calls, [(24, K)] * 4)  # the image's fused gather was offered first, as stock
        for ids, vals in out:
            self.assertTrue(torch.equal(ids, out[0][0]) and torch.equal(vals, out[0][1]))  # rank-identical
            self.assertTrue(torch.equal(vals, ref_vals))
            self.assertEqual(ids.dtype, torch.int64)
        hot1 = models[4]._idf_hot
        logits1 = quant.linear_ref(hidden, hot1.pack)
        pos = {int(t): j for j, t in enumerate(hot1.ids64.tolist())}
        ids4, vals4 = out[0]
        for row in range(hidden.shape[0]):
            got = ids4[row].tolist()
            self.assertEqual(len(set(got)), K)
            ref = models[4]._transform_unary_logits(logits1[row, [pos[t] for t in got]])  # KeyError: not a hot id
            self.assertTrue(torch.equal(ref, vals4[row]))
            srt = torch.sort(logits1[row].float(), descending=True).values
            if srt[K - 1] > srt[K]:  # no tie at the K-th place: the TP1 ids
                self.assertEqual(set(got), set(ref_ids[row].tolist()))

    def test_worker_builds_under_tp(self):
        built = []
        orig = p._ensure_hot
        p._ensure_hot = lambda model, head: built.append(head)
        try:
            class W:
                def _maybe_build_draft_sampler(self):
                    return "built"

            mod = types.SimpleNamespace(DFlashWorkerV2=W, unwrap_lora_layer=lambda m: m,
                                        get_parallel=lambda: types.SimpleNamespace(tp_size=4))
            p.patch_worker(mod)
            head = FakeHead(None, self.V, 4, 2, group=object())
            for tp_size in (4, 2):  # 2: the head is sharded over another group (4 != 2), left to the stock path
                mod.get_parallel = (lambda n: (lambda: types.SimpleNamespace(tp_size=n)))(tp_size)
                w = W()
                w.draft_model = types.SimpleNamespace(candidate_selector=object())
                w._target_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(
                    model=types.SimpleNamespace(lm_head=head)))
                self.assertEqual(w._maybe_build_draft_sampler(), "built")
            self.assertEqual(built, [head])
        finally:
            p._ensure_hot = orig


if __name__ == "__main__":
    unittest.main()
