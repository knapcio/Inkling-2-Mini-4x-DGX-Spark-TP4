"""CPU tests of the static-pool binding (KV12S_POOL=static)."""

import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from inkling_kv12s import IncompatibleImage
from inkling_kv12s import patch as hooks
from inkling_kv12s import static
from inkling_kv12s.storage import PAGE, Layout


def server_args(**kw):
    base = dict(max_running_requests=48, disable_overlap_schedule=False, max_prefill_tokens=8192,
                chunked_prefill_size=8192, context_length=262144)
    base.update(kw)
    return SimpleNamespace(**base)


def kvc(local=(0, 2), layers=3):
    hf = SimpleNamespace(local_layer_ids=list(local), num_hidden_layers=layers)
    return SimpleNamespace(is_draft_worker=False, spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
                           spec_aux_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=hf)))


def runtime(args):
    return {"sglang.srt.runtime_context": SimpleNamespace(get_server_args=lambda: args),
            "sglang.srt.mem_cache.kv_cache_configurator": SimpleNamespace(max_speculative_num_draft_tokens=lambda: 8)}


class SizingTests(unittest.TestCase):
    def _configurator(self):
        class HybridSWAPoolConfigurator:
            def __init__(self, kvc):
                self._full_layers_num = 8
                self._draft_swa_layers_num = 0
                self._draft_swa_full_layers_num = 0
                self._draft_full_layers_num = 0
                self._full_per_token = 8 * 128 * 2 * 2
                self._swa_per_token = 8 * 128 * 2 * 2
                self._swa_layers_num = 40
                self._swa_full_tokens_ratio = 0.037
                self._draft_cell_size = 3 * 4096
                self._draft_swa_cell_size = 0
                self._draft_follows_swa = False
                self._fused_full_entry = None
                self._recompute_cell_size()

            def _full_cell_bytes(self):
                return self._full_per_token * (self._full_layers_num + self._draft_full_layers_num)

            def _recompute_cell_size(self):
                self._cell_size = (self._full_cell_bytes()
                                   + self._swa_full_tokens_ratio
                                   * (self._swa_per_token * self._swa_layers_num + self._draft_swa_cell_size)
                                   + self._draft_cell_size)

            def calculate_pool_sizes(self, available_bytes, page_size):
                return int(available_bytes // self._cell_size) // page_size * page_size

        return HybridSWAPoolConfigurator

    def test_target_cell_is_the_packed_envelope_and_fixed_bytes_come_first(self):
        cls = self._configurator()
        stock_target = cls(SimpleNamespace(is_draft_worker=False))
        with patch.dict(sys.modules, runtime(server_args())):
            hooks._register_static_sizing(SimpleNamespace(HybridSWAPoolConfigurator=cls))
            target = cls(kvc())
            draft = cls(SimpleNamespace(is_draft_worker=True))
        self.assertEqual(target._full_cell_bytes(), 8 * 3108 + 1)  # + the per-page failure flags
        self.assertEqual(draft._full_cell_bytes(), 8 * 4096)
        self.assertAlmostEqual(stock_target._cell_size - target._cell_size,
                               8 * (4096 - 3108) - 1 + 2 * 4096 * (1 - 0.037), places=6)
        fixed = static.fixed_bytes(48, 8, overlap=True)
        self.assertAlmostEqual(fixed / 2**30, 2.667, places=2)
        budget = 75_000_000_000
        self.assertEqual(target.calculate_pool_sizes(budget, PAGE),
                         int((budget - fixed) // target._cell_size) // PAGE * PAGE)
        self.assertEqual(draft.calculate_pool_sizes(budget, PAGE), int(budget // draft._cell_size) // PAGE * PAGE)

    def test_draft_window_lanes_are_priced_per_window_token(self):
        cls = self._configurator()
        stock = cls(SimpleNamespace(is_draft_worker=False))
        with patch.dict(sys.modules, runtime(server_args())):
            hooks._register_static_sizing(SimpleNamespace(HybridSWAPoolConfigurator=cls))
            target = cls(kvc())
            draft = cls(SimpleNamespace(is_draft_worker=True))
            with self.assertRaises(IncompatibleImage):  # same bytes, one window layer
                cls(kvc(local=[0]))
        lanes = 2 * 4096
        self.assertAlmostEqual(stock._cell_size - target._cell_size,
                               8 * (4096 - 3108) - 1 + lanes - 0.037 * lanes, places=6)
        self.assertEqual((target._draft_cell_size, target._draft_swa_cell_size), (4096, lanes))
        self.assertEqual(draft._draft_cell_size, 3 * 4096)

    def test_an_unexpected_draft_pool_is_refused(self):
        cls = self._configurator()

        class Other(cls):
            def __init__(self, kvc):
                super().__init__(kvc)
                self._draft_cell_size = 2 * 4096

        with patch.dict(sys.modules, runtime(server_args())):
            hooks._register_static_sizing(SimpleNamespace(HybridSWAPoolConfigurator=Other))
            with self.assertRaises(IncompatibleImage):
                Other(kvc())

    def test_an_unbounded_request_count_is_refused(self):
        cls = self._configurator()
        with patch.dict(sys.modules, runtime(server_args(max_running_requests=None))):
            hooks._register_static_sizing(SimpleNamespace(HybridSWAPoolConfigurator=cls))
            with self.assertRaises(IncompatibleImage):
                cls(kvc())


class ReleaseTests(unittest.TestCase):
    def test_full_page_frees_release_the_packed_pages(self):
        released, cleared = [], []

        class PagedTokenToKVPoolAllocator:
            def _release_page_ids(self, *page_ids):
                self.freed = torch.cat([ids.reshape(-1) for ids in page_ids])

            def clear(self):
                pass

        hooks._register_static_release(SimpleNamespace(PagedTokenToKVPoolAllocator=PagedTokenToKVPoolAllocator))
        full, swa = PagedTokenToKVPoolAllocator(), PagedTokenToKVPoolAllocator()
        full._inkling_kv12s_view = SimpleNamespace(release_pages=lambda ids: released.append(ids.tolist()),
                                                   release_all=lambda: cleared.append(True))
        full._release_page_ids(torch.tensor([3, 4]), torch.tensor([[9]]))
        swa._release_page_ids(torch.tensor([5]))
        full.clear()
        swa.clear()
        self.assertEqual(released, [[3, 4, 9]])
        self.assertEqual(full.freed.tolist(), [3, 4, 9])
        self.assertEqual(cleared, [True])


class BuildTests(unittest.TestCase):
    def _module(self):
        built = {}

        class MHATokenToKVPool:
            pass

        class KVCacheConfigurator:
            def _build_token_to_kv_pool(self, *, sizes, is_dsa_model, is_compressed_kv_model, req_to_token_pool,
                                        state=None):
                return self._build_hybrid_swa_kv_pool(full_max_total_num_tokens=sizes.full,
                                                      swa_max_total_num_tokens=sizes.swa,
                                                      mha_pool_class=MHATokenToKVPool)

            def _build_hybrid_swa_kv_pool(self, *, full_max_total_num_tokens, swa_max_total_num_tokens,
                                          mha_pool_class):
                return ("stock", full_max_total_num_tokens)

            def _build_token_to_kv_pool_allocator(self, *, sizes, token_to_kv_pool, is_compressed_kv_model,
                                                  req_to_token_pool, token_to_kv_pool_allocator):
                return self.allocator

        def swa_pool(**kwargs):
            built.update(kwargs)
            return SimpleNamespace(full_kv_pool=SimpleNamespace(is_inkling_kv12s=True, page_size=PAGE))

        module = SimpleNamespace(
            KVCacheConfigurator=KVCacheConfigurator, MHATokenToKVPool=MHATokenToKVPool, SWAKVPool=swa_pool,
            get_parallel=lambda: SimpleNamespace(attn_tp_size=1, attn_dcp_size=1),
            get_spec=lambda: SimpleNamespace(speculative_algorithm="DSPARK"),
            get_exec=lambda: SimpleNamespace(features=SimpleNamespace(enable_memory_saver=False)),
            max_speculative_num_draft_tokens=lambda: 8,
        )
        return module, built

    def _configurator(self, module, *, draft=False):
        conf = module.KVCacheConfigurator()
        conf.is_draft_worker = draft
        conf.is_hybrid_swa_compress = False
        conf.kv_cache_dtype_str = "auto"
        conf.kv_cache_dtype = torch.bfloat16
        conf.post_capture_kv_active = False
        conf.pool_page_size = PAGE
        conf.device = "cpu"
        conf.model_config = SimpleNamespace(get_num_kv_heads=lambda tp, dcp: 8, head_dim=128, v_head_dim=128)
        conf.layer_info = SimpleNamespace(swa_attention_layer_ids=[0, 1], full_attention_layer_ids=[5, 11])
        return conf

    def test_target_full_sub_pool_is_packed_and_the_draft_stays_stock(self):
        module, built = self._module()
        sizes = SimpleNamespace(full=4096, swa=1024, max_running_requests=48)
        with patch.dict(sys.modules, runtime(server_args())), patch.object(static, "pool_class", lambda: "Packed"):
            hooks._register_static_pools(module)
            target = self._configurator(module)
            pool = target._build_token_to_kv_pool(sizes=sizes, is_dsa_model=False, is_compressed_kv_model=False,
                                                  req_to_token_pool=None)
            draft = self._configurator(module, draft=True)
            stock = draft._build_token_to_kv_pool(sizes=sizes, is_dsa_model=False, is_compressed_kv_model=False,
                                                  req_to_token_pool=None)
        self.assertEqual(stock, ("stock", 4096))
        self.assertEqual(built["full_kv_pool_class"], "Packed")
        self.assertEqual((built["size"], built["size_swa"], built["page_size"]), (4096, 1024, PAGE))
        self.assertEqual(built["full_kv_pool_kwargs"]["inkling"],
                         {"running": 48, "verify_rows": 8, "overlap": True, "prefill_tokens": 8192})
        self.assertTrue(pool.full_kv_pool.is_inkling_kv12s)

    def test_more_requests_than_the_sizing_charged_are_refused(self):
        module, _ = self._module()
        sizes = SimpleNamespace(full=4096, swa=1024, max_running_requests=64)
        with patch.dict(sys.modules, runtime(server_args())), patch.object(static, "pool_class", lambda: "Packed"):
            hooks._register_static_pools(module)
            with self.assertRaises(IncompatibleImage):
                self._configurator(module)._build_token_to_kv_pool(
                    sizes=sizes, is_dsa_model=False, is_compressed_kv_model=False, req_to_token_pool=None)

    def test_the_full_allocator_is_tagged_with_the_packed_view(self):
        module, _ = self._module()

        class PagedTokenToKVPoolAllocator:
            page_size = PAGE

        paged = {"sglang.srt.mem_cache.allocator.paged":
                 SimpleNamespace(PagedTokenToKVPoolAllocator=PagedTokenToKVPoolAllocator)}
        view = SimpleNamespace(is_inkling_kv12s=True, page_size=PAGE)
        pool = SimpleNamespace(full_kv_pool=view)
        with patch.dict(sys.modules, paged):
            hooks._register_static_pools(module)
            conf = self._configurator(module)
            conf.allocator = SimpleNamespace(full_attn_allocator=PagedTokenToKVPoolAllocator())
            allocator = conf._build_token_to_kv_pool_allocator(
                sizes=None, token_to_kv_pool=pool, is_compressed_kv_model=False, req_to_token_pool=None,
                token_to_kv_pool_allocator=None)
            self.assertIs(allocator.full_attn_allocator._inkling_kv12s_view, view)
            draft = self._configurator(module, draft=True)
            draft.allocator = SimpleNamespace(full_attn_allocator=PagedTokenToKVPoolAllocator())
            other = draft._build_token_to_kv_pool_allocator(
                sizes=None, token_to_kv_pool=pool, is_compressed_kv_model=False, req_to_token_pool=None,
                token_to_kv_pool_allocator=None)
            self.assertFalse(hasattr(other.full_attn_allocator, "_inkling_kv12s_view"))

    def test_view_lookup_covers_both_bindings(self):
        unified = SimpleNamespace(_arena=SimpleNamespace(_inkling_kv12s="arena view"))
        packed = SimpleNamespace(full_kv_pool=SimpleNamespace(is_inkling_kv12s=True))
        stock = SimpleNamespace(full_kv_pool=SimpleNamespace())
        self.assertEqual(static.view_of(unified), "arena view")
        self.assertIs(static.view_of(packed), packed.full_kv_pool)
        self.assertIsNone(static.view_of(stock))


@unittest.skipUnless(importlib.util.find_spec("triton") and os.getenv("TRITON_INTERPRET") == "1",
                     "requires Triton and TRITON_INTERPRET=1")
class StaticPoolTests(unittest.TestCase):
    """The packed full sub-pool on a stub MHATokenToKVPool base, with the interpreted writer."""

    def setUp(self):
        class MHATokenToKVPool:
            def __init__(self, *, size, page_size, dtype, head_num, head_dim, layer_num, device,
                         enable_memory_saver, allocation_label=None, enable_kv_cache_copy=False):
                self.size, self.page_size, self.dtype, self.store_dtype = size, page_size, dtype, dtype
                self.head_num, self.head_dim, self.v_head_dim = head_num, head_dim, head_dim
                self.layer_num, self.device = layer_num, device
                self.post_capture_active = self.use_hnd = self.is_quantized_kv_cache = False
                self._create_buffers()
                self.model_layer_ids = tuple(range(layer_num))

        def unwrap_write_loc(loc_info):
            return getattr(loc_info, "loc", loc_info), None, None

        stub = ModuleType("sglang.srt.mem_cache.memory_pool")
        stub.MHATokenToKVPool, stub.unwrap_write_loc = MHATokenToKVPool, unwrap_write_loc
        self.modules = patch.dict(sys.modules, {"sglang.srt.mem_cache.memory_pool": stub, **runtime(server_args())})
        self.modules.start()
        from inkling_kv12s import pressure

        self.addCleanup(self.modules.stop)
        self.addCleanup(lambda: hooks._PREFIX_SCRATCH.clear())
        self.addCleanup(lambda: pressure.GUARD.clear())
        with patch.dict(os.environ, {"INKLING_KV12S_PREFILL_SCRATCH_PAGES": "4"}):
            self.pool = static.make_pool_class()(
                size=6 * PAGE, page_size=PAGE, dtype=torch.bfloat16, head_num=2, head_dim=128, layer_num=8,
                device="cpu", enable_memory_saver=False,
                inkling={"running": 2, "verify_rows": 8, "overlap": False, "prefill_tokens": 256})

    def test_writes_reads_and_release(self):
        pool = self.pool
        self.assertEqual(pool.num_pages, 7)
        self.assertEqual(pool.raw.numel(), 7 * 8 * Layout(2).size)
        g = torch.Generator().manual_seed(3)
        k = torch.randn((PAGE + 20, 2, 128), generator=g).to(torch.bfloat16)
        v = torch.randn((PAGE + 20, 2, 128), generator=g).to(torch.bfloat16)
        loc = torch.arange(2 * PAGE, 3 * PAGE + 20)
        layer = SimpleNamespace(layer_id=3)
        pool.set_kv_buffer(layer, loc, k, v)
        pool.set_kv_buffer(None, loc, v, k, layer_id_override=5)
        got = pool.layers[3].read_rows(loc.tolist())
        self.assertTrue(torch.equal(got.view(torch.int16), torch.stack((k, v), 1).view(torch.int16)))
        got = pool.layers[5].read_rows(loc.tolist())
        self.assertTrue(torch.equal(got.view(torch.int16), torch.stack((v, k), 1).view(torch.int16)))
        self.assertEqual(pool.layers[3].flags[2].item(), 1)          # sealed
        self.assertGreaterEqual(pool.layers[3].flags[3].item(), 2)   # open tail, staged
        self.assertEqual(int(pool.layers[3].stage_locks.sum()), 1)
        self.assertEqual(int(pool.layers[0].flags[2:4].sum()), 0)    # other visits untouched
        with self.assertRaises(RuntimeError):
            pool.get_key_buffer(0)
        pool.release_pages(torch.tensor([2, 3], dtype=torch.int32))  # stock request rows are int32
        for lane in pool.layers:
            self.assertEqual(lane.flags[2:4].tolist(), [0, 0])
            self.assertEqual(int(lane.stage_locks.sum()), 0)
        pool.set_kv_buffer(layer, loc, v, k)                         # the pages' next owner
        got = pool.layers[3].read_rows(loc.tolist())
        self.assertTrue(torch.equal(got.view(torch.int16), torch.stack((v, k), 1).view(torch.int16)))
        pool.release_all()
        self.assertEqual(int(pool.layers[3].flags[:7].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(importlib.util.find_spec("triton") and os.getenv("TRITON_INTERPRET") == "1",
                     "requires Triton and TRITON_INTERPRET=1")
class CapturedScratchTests(unittest.TestCase):
    """PrefixScratch.gather_captured (device metadata, fixed grid) against the eager gather."""

    def _layer(self):
        from inkling_kv12s.storage import Layer

        layer = Layer.standalone(10, 1)
        g = torch.Generator().manual_seed(5)
        rows = 9 * PAGE
        k = torch.randn((rows, 1, 128), generator=g).to(torch.bfloat16)
        v = torch.randn((rows, 1, 128), generator=g).to(torch.bfloat16)
        layer.write(list(range(PAGE, PAGE + rows - 40)), k[:rows - 40], v[:rows - 40])
        return layer

    def test_fits_matches_the_eager_gather_and_overflow_gathers_nothing(self):
        from inkling_kv12s.attention import PrefixScratch

        layer = self._layer()
        # Two requests sharing page 2; request 1 ends inside its third page; a padded third row.
        table = torch.tensor([[2, 1, 6, 0], [2, 3, 5, 0], [0, 0, 0, 0]], dtype=torch.int32)
        seq = torch.tensor([3 * PAGE, 2 * PAGE + 17, 0], dtype=torch.int32)
        q = torch.zeros((8, 2, 128), dtype=torch.bfloat16)
        cu_q = torch.tensor([0, 4, 8, 8], dtype=torch.int32)
        eager = PrefixScratch(16, 1, "cpu")
        ek, ev, etable = eager.gather(layer, q, table, seq, cu_q)
        captured = PrefixScratch(16, 1, "cpu")
        ck, cv, ctable, fits = captured.gather_captured(layer, table, seq)
        self.assertTrue(bool(fits))
        for b in range(2):
            for c in range((int(seq[b]) + PAGE - 1) // PAGE):
                n = min(PAGE, int(seq[b]) - c * PAGE)
                es, cs = int(etable[b, c]), int(ctable[b, c])
                self.assertTrue(torch.equal(ek[es, :n].view(torch.int16), ck[cs, :n].view(torch.int16)))
                self.assertTrue(torch.equal(ev[es, :n].view(torch.int16), cv[cs, :n].view(torch.int16)))
                self.assertTrue(torch.equal(ck[cs, n:].view(torch.int16), torch.zeros_like(ck[cs, n:]).view(torch.int16)))
        self.assertEqual(ctable[:2, :3].tolist(), [[1, 2, 3], [4, 5, 6]])  # no dedup; slot 0 is the sink
        self.assertEqual(int(ctable[2].abs().sum()), 0)
        # 7 pages + sink > 6 slots: nothing gathered, the caller reads the packed store.
        small = PrefixScratch(6, 1, "cpu")
        small.buffer.fill_(77)
        _, _, stable, fits = small.gather_captured(layer, table, seq)
        self.assertFalse(bool(fits))
        self.assertEqual(int(stable.abs().sum()), 0)
        self.assertTrue(bool((small.buffer == 77).all()))


class DFlashTests(unittest.TestCase):
    def test_dflash_verify_is_settled_and_extends_are_not(self):
        settled = []
        lane = SimpleNamespace(settle=lambda ids, commit: settled.append((ids.tolist(), commit.tolist())))
        view = SimpleNamespace(is_inkling_kv12s=True, owned_layers=(lane, lane))
        pool = SimpleNamespace(full_kv_pool=view)

        class DFlashWorkerV2:
            block_size = 4
            _hosts_draft = True

            def __init__(self):
                self.target_worker = SimpleNamespace(model_runner=SimpleNamespace(token_to_kv_pool=pool))
                self._draft_verify_out_cache_loc_buf = torch.arange(3 * 4).reshape(3, 4) + 256

            def forward_batch_generation(self, batch, on_publish=None, grammar_barrier=None, pp_proxy_tensors=None):
                # prepare_for_verify switches the batch to TARGET_VERIFY, an extend mode.
                batch.forward_mode = SimpleNamespace(is_extend=lambda: True, is_idle=lambda: False)
                return SimpleNamespace(accept_lens=torch.tensor([2, 1], dtype=torch.int32))

        def batch(extend=False, idle=False):
            mode = SimpleNamespace(is_extend=lambda: extend, is_idle=lambda: idle)
            return SimpleNamespace(forward_mode=mode, is_extend_in_batch=False, reqs=[])

        hooks._register_dflash(SimpleNamespace(DFlashWorkerV2=DFlashWorkerV2))
        worker = DFlashWorkerV2()
        with patch("inkling_kv12s.pressure.GUARD", []):
            worker.forward_batch_generation(batch())
            worker.forward_batch_generation(batch(extend=True))
            worker.forward_batch_generation(batch(idle=True))
        rows = [[256, 257, 258, 259], [260, 261, 262, 263]]
        self.assertEqual(settled, [(rows, [2, 1]), (rows, [2, 1])])

    def test_dflash_is_accepted_only_on_static_pools(self):
        memory = SimpleNamespace(enable_flexkv=False, radix_cache_backend=None)
        with patch.object(hooks, "_POOL", "static"):
            hooks._check_cache_selectors(memory, "DFLASH", None)
            with self.assertRaises(IncompatibleImage):
                hooks._check_cache_selectors(memory, "EAGLE", None)
        with patch.object(hooks, "_POOL", "unified"), self.assertRaises(IncompatibleImage):
            hooks._check_cache_selectors(memory, "DFLASH", None)

    def test_a_dflash_draft_ring_keeps_stock_pricing(self):
        cls = SizingTests._configurator(SizingTests())
        ring = SimpleNamespace(is_draft_worker=False, spec_algorithm=SimpleNamespace(is_dspark=lambda: False))

        class Ring(cls):
            def __init__(self, kvc):
                super().__init__(kvc)
                self._draft_cell_size = 0  # the ring is a fixed charge
                self._recompute_cell_size()

        with patch.dict(sys.modules, runtime(server_args())):
            hooks._register_static_sizing(SimpleNamespace(HybridSWAPoolConfigurator=Ring))
            target = Ring(ring)
        self.assertEqual((target._draft_cell_size, target._draft_swa_cell_size), (0, 0))
        self.assertEqual(target._full_cell_bytes(), 8 * 3108 + 1)
