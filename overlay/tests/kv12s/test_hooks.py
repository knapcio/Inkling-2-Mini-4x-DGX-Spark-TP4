"""CPU tests of the lifetime and retraction hook boundaries."""

from dataclasses import dataclass, replace
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from inkling_kv12s import IncompatibleImage
from inkling_attn.patch import use_visit
from inkling_kv12s import patch as hooks
from inkling_kv12s.patch import _check_cache_selectors, _register_lifetime, _register_snapshot, dispatch_packed_fa4
from inkling_kv12s.storage import DIM, EMPTY, PACK_BATCH, PAGE, CapacityError, Layer, Layout


class HookTests(unittest.TestCase):
    def setUp(self):
        layer = Layer.standalone(8, 1)
        view = SimpleNamespace(name="full", device=layer.device, layers=(layer,),
                               layer_num=1, head_num=1, head_dim=DIM, dtype=torch.bfloat16)
        arena = SimpleNamespace(_inkling_kv12s=view)
        self.layer, self.arena = layer, arena
        loc = list(range(PAGE, 2 * PAGE + 11))
        values = torch.arange(len(loc) * DIM, dtype=torch.int32).to(torch.int16).view(-1, 1, DIM)
        values = values.view(torch.bfloat16)
        self.layer.write(loc, values, values.flip(0))
        self.locations = torch.tensor(loc)
        self.expected = self.layer.read_rows(loc).view(torch.int16)

    def test_snapshot_dispatch_preserves_other_bands_and_remaps_global_rows(self):
        calls = []

        def copy_to_cpu(allocator, indices, *, state_indices, request_row):
            calls.append(("copy", [band.sub_pool_name for band in allocator.members], state_indices, request_row))
            return SimpleNamespace(bands={"swa": "unchanged"})

        def restore_from_cpu(allocator, snapshot, indices, *, state_indices, request_row):
            calls.append(("restore", [band.sub_pool_name for band in allocator.members], state_indices, request_row))
            self.assertEqual(snapshot.bands["swa"], "unchanged")

        module = SimpleNamespace(
            __name__="snapshot", copy_to_cpu=copy_to_cpu, restore_from_cpu=restore_from_cpu,
            BandSnapshot=SimpleNamespace, torch=torch,
            _physical=lambda band, indices, by_unit: indices + band.offset,
            current_platform=SimpleNamespace(synchronize=lambda: None),
        )
        _register_snapshot(module)
        full = SimpleNamespace(sub_pool_name="full", offset=PAGE)
        swa = SimpleNamespace(sub_pool_name="swa")
        allocator = SimpleNamespace(members=[full, swa], arena=self.arena, roles=SimpleNamespace(owner="full"),
                                    full_attn_allocator=full)
        virtual = self.locations - full.offset
        saved = module.copy_to_cpu(allocator, virtual, state_indices=7, request_row=9)
        self.assertEqual(saved.bands["full"].entries[0].device.type, "cpu")
        self.layer.release([1, 2])
        full.offset += 3 * PAGE
        module.restore_from_cpu(allocator, saved, virtual, state_indices=8, request_row=10)
        actual = self.layer.read_rows((virtual + full.offset).tolist()).view(torch.int16)
        self.assertTrue(torch.equal(actual, self.expected))
        self.assertEqual(calls, [("copy", ["swa"], 7, 9), ("restore", ["swa"], 8, 10)])

    def test_restore_preflights_every_visit_before_other_bands(self):
        calls = []

        def copy_to_cpu(allocator, indices, *, state_indices, request_row):
            self.fail("unexpected CPU snapshot")

        def restore_from_cpu(allocator, snapshot, indices, *, state_indices, request_row):
            calls.append((state_indices, request_row))

        module = SimpleNamespace(
            __name__="snapshot", copy_to_cpu=copy_to_cpu, restore_from_cpu=restore_from_cpu,
            torch=torch, _physical=lambda band, indices, by_unit: indices,
            current_platform=SimpleNamespace(synchronize=lambda: None),
        )
        _register_snapshot(module)
        visits = tuple(Layer.standalone(PACK_BATCH + 3, 1, overflow_heads=1) for _ in range(2))
        self.arena._inkling_kv12s.layers = visits
        full = SimpleNamespace(sub_pool_name="full")
        allocator = SimpleNamespace(members=[full], arena=self.arena, roles=SimpleNamespace(owner="full"),
                                    full_attn_allocator=full)
        locations = torch.arange(2 * PAGE, (PACK_BATCH + 3) * PAGE)
        entries = torch.full((len(locations), 2, 2, 1, DIM), 0x3F80, dtype=torch.int16)
        entries[-PAGE:, 1, 0].fill_(0x7FC2)
        wide, plain = entries[-PAGE:, 1, 0].view(torch.bfloat16), entries[-PAGE:, 1, 1].view(torch.bfloat16)
        visits[1].write(list(range(PAGE, 2 * PAGE)), wide, plain)
        snapshot = SimpleNamespace(bands={
            "full": SimpleNamespace(positions=torch.arange(len(locations)), entries=entries.view(torch.bfloat16)),
        })
        with self.assertRaisesRegex(CapacityError, "overflow"):
            module.restore_from_cpu(allocator, snapshot, locations, state_indices=8, request_row=10)
        self.assertFalse(calls)
        for layer in visits:
            self.assertTrue((layer.flags[2:] == EMPTY).all())
            self.assertFalse(layer.open)
        visits[1].release([1])
        module.restore_from_cpu(allocator, snapshot, locations, state_indices=8, request_row=10)
        self.assertEqual(calls, [(8, 10)])
        for i, layer in enumerate(visits):
            self.assertTrue(torch.equal(layer.read_rows(locations.tolist()).view(torch.int16), entries[:, i]))

    def test_startup_allows_dspark_and_rejects_external_readers(self):
        memory = SimpleNamespace(enable_flexkv=False, radix_cache_backend=None)
        _check_cache_selectors(memory, None, None)
        _check_cache_selectors(memory, "DSPARK", None)
        for config, algorithm, disagg, error in (
            (SimpleNamespace(enable_flexkv=True, radix_cache_backend=None), None, None, "FlexKV"),
            (SimpleNamespace(enable_flexkv=False, radix_cache_backend="custom"), None, None, "explicit backend"),
            (memory, "EAGLE", None, "linear DSpark"),
            (memory, None, "decode", "PD decode restore"),
        ):
            with self.assertRaisesRegex(IncompatibleImage, error):
                _check_cache_selectors(config, algorithm, disagg)

    def test_shared_draft_is_priced_and_bound_to_its_own_lane(self):
        @dataclass(frozen=True, kw_only=True)
        class MHA:
            name: str
            layer_num: int
            grow_direction: str
            head_num: int
            head_dim: int
            v_head_dim: int
            store_dtype: torch.dtype
            kv_cache_dtype: torch.dtype = torch.bfloat16
            is_mxfp8: bool = False
            draft_region: object = None

        class DraftRegion:
            lane_num = token_replicas = 1
            head_num, head_dim = 2, DIM

            def resolved_kv_dtype(self):
                return torch.bfloat16

            def resolved_v_head_dim(self):
                return DIM

        @dataclass(frozen=True)
        class Pool:
            name: str
            spec: MHA
            parts: tuple = ()

        @dataclass(frozen=True)
        class Plan:
            sub_pools: tuple
            derived_parts: tuple = ()

            def owner(self):
                return self.sub_pools[0]

        draft = DraftRegion()
        owner = Pool("full", MHA(name="full", layer_num=8, grow_direction="down", head_num=8,
                                 head_dim=DIM, v_head_dim=DIM, store_dtype=torch.bfloat16, draft_region=draft))
        window = Pool("window", owner.spec)

        def plan_kv_storage(decls, *, page_size, fused_draft=None, speculative_num_draft_tokens=None, derived_parts=()):
            return Plan((owner, window))

        def _fixed_terms(plan, inp, running, strategy):
            return ()

        def _derived_terms(plan, chain_bytes):
            return ()

        sizing = SimpleNamespace(_fixed_terms=_fixed_terms, _derived_terms=_derived_terms, FixedAllocation=SimpleNamespace)
        planner = SimpleNamespace(plan_kv_storage=plan_kv_storage, MHASubPoolSpec=MHA, DenseDraftRegion=DraftRegion)
        fake_msgspec = SimpleNamespace(structs=SimpleNamespace(replace=replace))
        with patch.dict(sys.modules, msgspec=fake_msgspec), patch.object(hooks, "PackedSpec", None):
            hooks._register_plan(planner, 32, 1024)
            hooks._register_sizing(sizing)
            token = hooks._PLAN_VERIFY_ROWS.set(8)
            try:
                plan = planner.plan_kv_storage((), page_size=PAGE, fused_draft=object())
            finally:
                hooks._PLAN_VERIFY_ROWS.reset(token)
            spec = plan.owner().spec
            self.assertIs(spec.draft_region, draft)
            self.assertIs(plan.sub_pools[1], window)
            # The MTP draft lane stays raw BF16 (page-major) inside the envelope.
            self.assertEqual(spec.entry_bytes() * PAGE,
                             8 * Layout(8).size + 2 * PAGE * 2 * DIM * 2)
            self.assertEqual(spec.lane_overflow_heads(), (32,) * 8)
            serial = sizing._fixed_terms(plan, None, 48, SimpleNamespace(overlap=False))
            overlap = sizing._fixed_terms(plan, None, 48, SimpleNamespace(overlap=True))
            self.assertGreater(overlap[0].bytes, serial[0].bytes)
            self.assertEqual(overlap[1].bytes, serial[1].bytes)
            chain = spec.entry_bytes() * PAGE * 17 + 123
            # No BF16 FA4 gather workspace unless INKLING_KV12S_FA4_WORKSPACE=1.
            self.assertEqual(sizing._derived_terms(plan, chain)[0].bytes, 0)

            bound = object()
            arena = self.arena
            arena._inkling_kv12s.draft_view = lambda a, s, mapping: bound if mapping == {0: 0} else None
            module = SimpleNamespace(DraftMHAView=lambda **kwargs: kwargs["host_spec"],
                                     DRAFT_BINDERS={"dense": lambda ctx, binding: "mixed"})
            hooks._register_draft_binding(module)
            args = dict(arena=arena, layer_lanes={0: 0}, page_size=PAGE)
            self.assertIs(module.DraftMHAView(host_spec=spec, **args), bound)
            self.assertEqual(module.DraftMHAView(host_spec=SimpleNamespace(name="window"), **args).name,
                          "window")

    def test_dspark_settles_physical_windows_after_draft_commit(self):
        issued = []
        virtual = torch.tensor([[PAGE, PAGE + 1], [2 * PAGE - 1, 3 * PAGE]])
        physical = virtual + PAGE
        accept = torch.tensor([1, 2], dtype=torch.int32)
        runner_ids = object()

        class Lane:
            def settle(self, locations, lengths):
                self_outer.assertTrue(torch.equal(locations, physical))
                self_outer.assertIs(lengths, accept)
                issued.append("settle")

        self_outer = self
        iteration = SimpleNamespace(write_ids=lambda ids: physical if ids is runner_ids else None)
        self.arena._inkling_kv12s.owned_layers = (Lane(), Lane())

        def alloc_verify_window(*, batch, bs, device, verify_num_draft_tokens, block_pos_offsets,
                                model_runner, seq_lens_cpu, bonus=None, anchor_ids=None):
            return SimpleNamespace(iteration_ids=iteration, verify_cache_loc_2d=virtual)

        class DSparkWorkerV2:
            def _forward_decode(self, batch, on_publish, grammar_barrier=None, coordination_plan=None):
                module.alloc_verify_window(batch=batch, bs=2, device="cpu", verify_num_draft_tokens=2,
                                           block_pos_offsets=None, model_runner=None, seq_lens_cpu=None)
                issued.extend(("verify", "draft prefix commit"))
                return SimpleNamespace(accept_lens=accept)

            def alloc_memory_pool(self, memory_pool_config=None, req_to_token_pool=None,
                                  token_to_kv_pool_allocator=None):
                issued.append(("alloc", self._static_epilogue_needs_draft_fold))

        module = SimpleNamespace(DSparkWorkerV2=DSparkWorkerV2, alloc_verify_window=alloc_verify_window)
        hooks._register_dspark(module)
        worker = DSparkWorkerV2()
        worker.target_worker = SimpleNamespace(model_runner=SimpleNamespace(
            runner_ids=runner_ids, token_to_kv_pool=SimpleNamespace(_arena=self.arena)))
        result = worker._forward_decode(None, None)
        self.assertIs(result.accept_lens, accept)
        self.assertEqual(issued, ["verify", "draft prefix commit", "settle", "settle"])
        # The unified FULL translator fails stock's static fold predicate:
        # the epilogue keeps the captured accept and commits the draft eagerly.
        worker._static_epilogue_needs_draft_fold = True
        worker.alloc_memory_pool()
        self.assertEqual(issued[-1], ("alloc", False))

    def test_overlap_settlement_recycles_only_after_stock_fence(self):
        order = []
        pages = torch.tensor([2, 3])
        vacated = object()

        def recycle(ids, *, recycle=True):
            self.assertIs(ids, pages)
            order.append("recycle" if recycle else "invalidate moved source")

        view = self.arena._inkling_kv12s
        view.device_owned = True
        view.owned_layers = (SimpleNamespace(release=recycle),)

        class Band:
            sub_pool_name = "full"
            _pending_until_retire = True
            _free_phys_pages = pages

            def _set_free_pages(self, current, *, host):
                self._free_phys_pages, self._host_free = current, host

            def release(self, pages, cause, *, host=None, readers=()):
                order.append("publish free list")

            def settle_pending(self, through=None, prepared_holes=None):
                order.append("wait for reader event")
                self.release(pages, "free")

            def clear(self):
                pass

            usage = SimpleNamespace(latest_forward=object())

            def _wait_on_stream(self, event):
                order.append("wait for moved source readers")

            def _move_pages(self, src_pages, dst_pages):
                order.append("copy envelope")

        band_module = SimpleNamespace(Band=Band, Release=SimpleNamespace(VACATED=vacated))
        hooks._register_lifetime(band_module)
        hooks._register_move(band_module)
        band = Band()
        band.arena = self.arena
        band.settle_pending(prepared_holes=[1, 2, 3])
        self.assertEqual(band._host_free, [1, 2, 3])
        band._move_pages(pages, torch.tensor([4, 5]))
        band.release(pages, vacated)
        self.assertEqual(order, ["wait for reader event", "recycle", "publish free list",
                                 "wait for moved source readers", "copy envelope",
                                 "invalidate moved source", "publish free list"])

    def test_stock_fa4_gets_original_split_bias_and_output_arguments(self):
        calls = []
        q = torch.full((2, 4, DIM), 1, dtype=torch.bfloat16)
        marker = q[..., :16].contiguous()
        proj = torch.eye(16, 1024, dtype=q.dtype)
        original_table = torch.tensor([[1]], dtype=torch.int32)
        mapped = torch.tensor([[0]], dtype=torch.int32)
        cache = torch.zeros((1, PAGE, 1, DIM), dtype=q.dtype)
        out = torch.empty_like(q, dtype=torch.float32)
        bias = torch.einsum("thd,de->the", marker, proj)

        def gather(layer, query, table, seq, cu_q, qmax):
            self.assertIs(layer, self.layer)
            self.assertIs(query, q)
            self.assertIs(table, original_table)
            self.assertEqual(qmax, 2)
            return cache, cache, mapped

        def stock(**kwargs):
            calls.append(kwargs)
            return out

        kwargs = dict(q=q, page_table=original_table, cache_seqlens=torch.tensor([PAGE], dtype=torch.int32),
                      cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32), max_seqlen_q=2, rel_bias=bias,
                      causal=True, softmax_scale=0.1, window_size=(-1, -1), num_splits=3, pack_gqa=True, out=out)
        with patch.dict(sys.modules, {"inkling_kv12s.attention": SimpleNamespace(gather_for_fa4=gather)}), \
                patch.object(hooks, "_ALLOW_UNBOUNDED_GATHER", True):
            self.assertIs(dispatch_packed_fa4(stock, self.layer, kwargs), out)
            with use_visit(marker, proj):
                explicit = dict(kwargs, rel_bias=marker, max_seqlen_k=PAGE)
                self.assertIs(dispatch_packed_fa4(stock, self.layer, explicit), out)
        self.assertIs(calls[0]["rel_bias"], bias)
        self.assertTrue(torch.equal(calls[1]["rel_bias"], bias))
        self.assertNotIn("max_seqlen_k", kwargs)
        self.assertEqual(calls[0]["max_seqlen_k"], self.layer.pages * PAGE)
        self.assertEqual(calls[1]["max_seqlen_k"], PAGE)
        for call in calls:
            self.assertEqual(call["num_splits"], 3)
            self.assertIs(call["pack_gqa"], True)
            self.assertIs(call["out"], out)
            self.assertIs(call["page_table"], mapped)
            self.assertIs(call["k_cache"], cache)
            self.assertIs(call["v_cache"], cache)

    def test_scheduler_retraction_releases_and_reprefills_without_host_backup(self):
        root = os.environ.get("INKLING_ATTN_SGLANG_ROOT")
        if not root:
            self.skipTest("set INKLING_ATTN_SGLANG_ROOT to the SGLang package source")
        layer = self.layer

        class Band:
            sub_pool_name = "full"

            def release(self, pages, cause, *, host=None, readers=()):
                self.reusable.extend(pages.tolist())

            def settle_pending(self, through=None, prepared_holes=None):
                pass

            def clear(self):
                pass

        _register_lifetime(SimpleNamespace(Band=Band))
        band = Band()
        band.arena = self.arena
        band.reusable = []
        req = SimpleNamespace(kv=SimpleNamespace(holds_kv=True, holds_mamba=False), resumable_by_reprefill=True)

        def release(req_arg, tree_cache, *, checkpoint):
            self.assertIs(req_arg, req)
            self.assertFalse(checkpoint)
            tree_cache.release(torch.tensor([1, 2]), "retract")
            req.kv.holds_kv = False

        def unexpected_backup(*args):
            self.fail("ordinary scheduler retraction must re-prefill")

        # Execute the image's scheduler decision with only its release and
        # transport dependencies replaced by a CPU fixture.
        imports = {
            "sglang.srt.environ": SimpleNamespace(envs=None),
            "sglang.srt.managers.req_types": SimpleNamespace(FINISH_ABORT=None),
            "sglang.srt.managers.scheduler_components.exit_ledger": SimpleNamespace(record_exit=None),
            "sglang.srt.managers.utils": SimpleNamespace(make_abort_req=None),
            "sglang.srt.mem_cache.base_prefix_cache": SimpleNamespace(CacheRequestOutcome=None),
            "sglang.srt.mem_cache.common": SimpleNamespace(backup_kv_cache=unexpected_backup,
                                                          discard_kv_cache_backup=None, release_kv_cache=release),
            "sglang.srt.runtime_context": SimpleNamespace(get_disagg=lambda: SimpleNamespace(disaggregation_mode="null")),
        }
        source = Path(root) / "srt/managers/scheduler_components/request_retirer.py"
        with patch.dict(sys.modules, imports):
            mod = runpy.run_path(str(source))
        retirer = mod["RequestRetirer"](tree_cache=band, beam_coordinator=None, hisparse_coordinator=None,
                                         uocr_runtime=None, send_abort_output=None, stream_output=None)
        self.assertTrue(retirer.retract(req, offload_kv=True))
        self.assertEqual(band.reusable, [1, 2])
        self.assertEqual(layer.flags[band.reusable].tolist(), [EMPTY, EMPTY])
        self.assertFalse(layer.open or layer.spilled)
        prefill_locs = [page * PAGE + row for page, count in zip(band.reusable[::-1], (PAGE, 11))
                        for row in range(count)]
        layer.write(prefill_locs, self.expected[:, 0].view(torch.bfloat16), self.expected[:, 1].view(torch.bfloat16))
        self.assertTrue(torch.equal(layer.read_rows(prefill_locs).view(torch.int16), self.expected))

    def test_prepared_free_list_branch_recycles_external_ownership(self):
        class Pending:
            def __len__(self):
                return 1

            def oldest(self, n):
                return [torch.tensor([1])], (), ()

        layer = self.layer
        fixture = self

        class Band:
            sub_pool_name = "full"

            def release(self, pages, cause, *, host=None, readers=()):
                fixture.assertTrue((layer.flags[pages] == EMPTY).all())

            def clear(self):
                pass

            def settle_pending(self, through=None, prepared_holes=None):
                fixture.assertEqual(layer.flags[1].item(), EMPTY)
                self._pending_until_retire = ()

        _register_lifetime(SimpleNamespace(Band=Band))
        band = Band()
        band.arena = self.arena
        band._pending_until_retire = Pending()
        self.assertIn(1, self.layer.spilled)
        self.assertIn(2, self.layer.open)
        band.settle_pending(prepared_holes=[1])
        self.assertNotIn(1, self.layer.spilled)
        band.release(torch.tensor([2]), "free")
        self.assertNotIn(2, self.layer.open)


if __name__ == "__main__":
    unittest.main()
