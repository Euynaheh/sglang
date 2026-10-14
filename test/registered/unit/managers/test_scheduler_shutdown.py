"""CPU regressions for coordinated DP scheduler shutdown."""

import inspect
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers import scheduler as scheduler_module  # noqa: E402
from sglang.srt.managers.io_struct import ShutdownReq  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class _Scheduler:
    """Run the real event loops and metadata gather without a model or CUDA."""

    handle_shutdown = Scheduler.handle_shutdown
    _complete_shutdown = Scheduler._complete_shutdown
    is_disable_overlap_for_batch = Scheduler.is_disable_overlap_for_batch
    require_mlp_sync = True
    _engine_paused = False
    is_generation = False
    enable_unified_memory = False

    def __init__(self, rank, first_rank, first_iteration, delayed):
        self.rank = rank
        self.shutdown_iteration = first_iteration + (
            delayed if rank != first_rank else 0
        )
        self.iteration = -1
        self._shutdown_requested = False
        self.gracefully_exit = False
        self.running_batch = None
        self.last_batch = None
        self.group = SimpleNamespace(cpu_group=dist.group.WORLD, device="cpu")
        self.forwards = 0
        self.results = 0

    def ingest_requests(self):
        self.iteration += 1
        assert self.iteration < 20, "shutdown did not converge"
        if self.iteration == self.shutdown_iteration:
            self.handle_shutdown(ShutdownReq())

    def _batch(self, mode):
        return SimpleNamespace(
            forward_mode=mode,
            is_extend_in_batch=False,
            batch_size=lambda: 1,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            copy=lambda: self._batch(mode),
        )

    def get_next_batch_to_run(self, **kwargs):
        # Stand in for a prefill-delayer collective BEFORE the real MLP gather.
        value = torch.tensor([self.iteration])
        gathered = torch.empty(dist.get_world_size(), dtype=torch.int64)
        dist.all_gather_into_tensor(gathered, value)
        assert gathered.tolist() == [self.iteration] * dist.get_world_size()

        batch = (
            self._batch(ForwardMode.DECODE)
            if self.rank == 0 and self.iteration < 2
            else None
        )
        batch = dp_attn.prepare_mlp_sync_batch_raw(
            batch,
            model_runner=SimpleNamespace(
                spec_algorithm=SpeculativeAlgorithm.NONE,
                model_config=object(),
            ),
            dp_size=dist.get_world_size(),
            attn_tp_size=1,
            attn_cp_size=1,
            tp_group=self.group,
            get_idle_batch=lambda: self._batch(ForwardMode.IDLE),
            disable_cuda_graph=True,
            require_mlp_tp_gather=True,
            disable_overlap_schedule=False,
            offload_tags=set(),
            shutdown_requested=self._shutdown_requested,
            on_shutdown=self._complete_shutdown,
        )
        return SimpleNamespace(running_batch=None, batch_to_run=batch)

    def run_batch(self, batch):
        self.forwards += 1

    def process_batch_result(self, batch, result):
        self.results += 1

    def _apply_war_barrier(self):
        pass

    def on_idle(self):
        pass


def _shutdown_worker(rank, init_file):
    dist.init_process_group(
        "gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    parallel = SimpleNamespace(
        dp_size=2,
        tp_group=SimpleNamespace(active_ranks_cpu=torch.ones(2, dtype=torch.int32)),
    )
    tbo = Mock()
    tbo.prepare_all_gather.side_effect = lambda batch: (
        False,
        batch.forward_mode.value if batch else ForwardMode.IDLE.value,
    )
    tbo.compute_output.return_value = (None, None)
    try:
        with (
            patch.object(scheduler_module, "get_parallel", return_value=parallel),
            patch.object(dp_attn, "get_parallel", return_value=parallel),
            patch.object(dp_attn, "world_dp_gather_enabled", return_value=False),
            patch.object(dp_attn, "check_cuda_graph_backend", return_value=False),
            patch.object(dp_attn, "TboDPAttentionPreparer", return_value=tbo),
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
            envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.override(False),
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(False),
        ):
            for loop in ("normal", "overlap"):
                for first_rank in (0, 1):
                    for delayed in (0, 1, 5):
                        for repeat in range(5):
                            scheduler = _Scheduler(rank, first_rank, 3, delayed)
                            with patch.object(
                                dist,
                                "all_gather_into_tensor",
                                wraps=dist.all_gather_into_tensor,
                            ) as gather:
                                inspect.unwrap(
                                    getattr(Scheduler, "event_loop_" + loop)
                                )(scheduler)
                            assert scheduler.gracefully_exit
                            assert scheduler.iteration == 3
                            # Two existing gathers per iteration, no shutdown gather.
                            assert gather.call_count == 8
                            assert scheduler.forwards == scheduler.results == 2
                            if loop == "overlap":
                                assert not scheduler.result_queue
                            assert not torch.cuda.is_initialized()
                            dist.barrier()
        print(f"rank={rank}: 60 shutdown cases PASS, CUDA uninitialized", flush=True)
    finally:
        dist.destroy_process_group()


class TestSchedulerShutdown(CustomTestCase):
    def test_staggered_shutdown_normal_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = mp.spawn(
                _shutdown_worker,
                args=(str(Path(tmp) / "store"),),
                nprocs=2,
                join=False,
            )
            deadline = time.monotonic() + 120
            try:
                while not context.join(timeout=1):
                    self.assertLess(time.monotonic(), deadline, "worker timed out")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)

    def test_local_shutdown_without_metadata_gather(self):
        for require_sync, paused, dp_size, skip in (
            (False, False, 4, False),
            (True, True, 4, False),
            (True, False, 1, False),
            (True, False, 4, True),
        ):
            with self.subTest(
                require_sync=require_sync, paused=paused, dp_size=dp_size, skip=skip
            ):
                scheduler = _Scheduler(0, 0, 0, 0)
                scheduler.require_mlp_sync = require_sync
                scheduler._engine_paused = paused
                with (
                    patch.object(
                        scheduler_module,
                        "get_parallel",
                        return_value=SimpleNamespace(dp_size=dp_size),
                    ),
                    envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(skip),
                ):
                    scheduler.handle_shutdown(ShutdownReq())
                    scheduler.handle_shutdown(ShutdownReq())
                self.assertTrue(scheduler._shutdown_requested)
                self.assertTrue(scheduler.gracefully_exit)

    def test_dp_shutdown_waits_for_metadata_gather(self):
        scheduler = _Scheduler(0, 0, 0, 0)
        with (
            patch.object(
                scheduler_module,
                "get_parallel",
                return_value=SimpleNamespace(dp_size=2),
            ),
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
        ):
            scheduler.handle_shutdown(ShutdownReq())
        self.assertTrue(scheduler._shutdown_requested)
        self.assertFalse(scheduler.gracefully_exit)


class TestShutdownMetadata(CustomTestCase):
    @staticmethod
    def info(requested):
        return dp_attn.MLPSyncBatchInfo(
            dp_size=3,
            tp_size=1,
            cp_size=1,
            num_tokens=0,
            num_tokens_for_logprob=0,
            can_run_decode_cuda_graph=True,
            can_run_prefill_cuda_graph=False,
            is_extend_in_batch=False,
            local_can_run_tbo=False,
            local_forward_mode=ForwardMode.IDLE.value,
            shutdown_requested=requested,
        )

    def test_shutdown_vote_ignores_inactive_ranks(self):
        for use_all_reduce in (False, True):
            for active_vote in (False, True):
                with self.subTest(all_reduce=use_all_reduce, active_vote=active_vote):
                    info = self.info(False)
                    expected = torch.stack(
                        [
                            self.info(False)._get_local_tensor("cpu"),
                            self.info(active_vote)._get_local_tensor("cpu"),
                            self.info(True)._get_local_tensor("cpu"),
                        ]
                    )
                    group = SimpleNamespace(active_ranks_cpu=torch.tensor([1, 1, 0]))

                    def gather(output, *args, **kwargs):
                        output.copy_(expected.reshape(output.shape))

                    with (
                        patch.object(
                            dp_attn,
                            "get_parallel",
                            return_value=SimpleNamespace(tp_group=group),
                        ),
                        patch.object(dist, "get_rank", return_value=0),
                        patch.object(
                            dist, "all_gather_into_tensor", side_effect=gather
                        ),
                        patch.object(dist, "all_reduce", side_effect=gather),
                    ):
                        info.all_gather("cpu", object(), use_all_reduce=use_all_reduce)
                    self.assertEqual(info.shutdown_requested, active_vote)
                    self.assertEqual(info.global_num_tokens, [0, 0, 0])

    def test_local_and_fallback_shutdown_votes(self):
        info = self.info(True)
        info.finalize_local()
        self.assertTrue(info.shutdown_requested)
        self.assertEqual(info.tp0_info_cpu[0, 8], 1)
        self.assertEqual(info._get_fallback_tensor("cpu")[8], 0)


if __name__ == "__main__":
    unittest.main()
