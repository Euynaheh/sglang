from types import SimpleNamespace

import pytest
import torch

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.layers.moe import flashinfer_mega_moe, mega_moe
from sglang.srt.layers.moe.flashinfer_mega_moe import _view_byte_dtype
from sglang.srt.server_args import ServerArgs


def _resolve_dummy(**kwargs):
    args = ServerArgs(
        model_path="dummy",
        tp_size=4,
        dp_size=4,
        enable_dp_attention=True,
        ep_size=4,
        moe_runner_backend="flashinfer_megamoe",
        **kwargs,
    )
    args.resolve_once()
    return resolved_view(args)


def test_flashinfer_megamoe_resolution_contract():
    cfg = _resolve_dummy()
    assert cfg.moe_a2a_backend == "megamoe"
    assert cfg.disable_shared_experts_fusion
    assert cfg.flashinfer_megamoe_max_num_tokens == 8192


@pytest.mark.parametrize("moe_a2a_backend", ["deepep", "mori"])
def test_flashinfer_megamoe_rejects_external_dispatch(moe_a2a_backend):
    with pytest.raises(ValueError, match="owns dispatch and combine"):
        _resolve_dummy(moe_a2a_backend=moe_a2a_backend)


def test_flashinfer_megamoe_rejects_eplb():
    with pytest.raises(ValueError, match="disable EPLB"):
        _resolve_dummy(enable_eplb=True)


def test_flashinfer_megamoe_rejects_tbo():
    with pytest.raises(ValueError, match="does not support.*two-batch-overlap"):
        _resolve_dummy(enable_two_batch_overlap=True)


def test_mxfp4_weight_view_is_zero_copy():
    storage = torch.arange(16, dtype=torch.uint8)
    viewed = _view_byte_dtype(storage, torch.float4_e2m1fn_x2)
    assert viewed.dtype == torch.float4_e2m1fn_x2
    assert viewed.data_ptr() == storage.data_ptr()


@pytest.mark.parametrize("num_tokens", [0, 2])
def test_flashinfer_megamoe_forwards_int32_topk_ids(monkeypatch, num_tokens):
    hidden_states = torch.zeros((num_tokens, 4), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[1, 2], [3, 0]], dtype=torch.int32)[:num_tokens]
    topk_weights = torch.ones((num_tokens, 2), dtype=torch.float32)
    topk_output = SimpleNamespace(topk_ids=topk_ids, topk_weights=topk_weights)
    moe = SimpleNamespace(
        config=SimpleNamespace(
            num_experts_per_tok=2, hidden_size=4, moe_intermediate_size=8
        ),
        experts=SimpleNamespace(
            should_fuse_routed_scaling_factor_in_topk=True,
            moe_runner_config=SimpleNamespace(swiglu_limit=None),
        ),
        gate=lambda hidden_states, forward_batch: torch.empty((num_tokens, 4)),
        is_hash=False,
        layer_id=0,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        routed_scaling_factor_fused_in_topk=True,
        mega_shared_l1_weights=None,
        mega_shared_l2_weights=None,
        topk=lambda *args, **kwargs: topk_output,
    )
    captured = {}

    def fake_run_flashinfer_megamoe(
        layer,
        hidden_states,
        *,
        topk_ids,
        topk_weights,
        compile_tokens_per_rank,
    ):
        captured["topk_ids"] = topk_ids
        captured["compile_tokens_per_rank"] = compile_tokens_per_rank
        return torch.zeros_like(hidden_states)

    monkeypatch.setattr(
        mega_moe,
        "get_moe_runner_backend",
        lambda: SimpleNamespace(is_flashinfer_megamoe=lambda: True),
    )
    monkeypatch.setattr(mega_moe, "get_dp_global_num_tokens", lambda: [0, 3, 1, 0])
    monkeypatch.setattr(mega_moe, "_get_mega_moe_max_tokens_per_rank", lambda: 8)
    monkeypatch.setattr(
        flashinfer_mega_moe,
        "run_flashinfer_megamoe",
        fake_run_flashinfer_megamoe,
    )

    mega_moe._run_mega_routed(moe, hidden_states, None, None, num_tokens)

    assert captured["topk_ids"].dtype == torch.int32
    assert captured["topk_ids"].shape == (num_tokens, 2)
    if num_tokens:
        assert captured["topk_ids"] is topk_ids
    assert captured["compile_tokens_per_rank"] == 3


@pytest.mark.parametrize("capacity", [2048, 4095, 4096, 8192])
def test_flashinfer_megamoe_budget_uses_resolved_dp_chunk(capacity):
    from sglang.srt.arg_groups.mega_moe_hook import validate_mega_moe_token_budget

    args = ServerArgs(
        model_path="dummy",
        tp_size=4,
        dp_size=4,
        enable_dp_attention=True,
        ep_size=4,
        moe_runner_backend="flashinfer_megamoe",
        chunked_prefill_size=16384,
        flashinfer_megamoe_max_num_tokens=capacity,
    )
    args.resolve_once()
    # Dummy models skip hardware/model hooks. Reproduce the declaration
    # handle_data_parallelism makes before this validator runs.
    from sglang.srt.arg_groups.overrides import declare_resolution

    declare_resolution(args, "_handle_data_parallelism", chunked_prefill_size=4096)
    assert resolved_view(args).chunked_prefill_size == 4096
    if capacity < 4096:
        with pytest.raises(ValueError, match="required_per_rank=4096"):
            validate_mega_moe_token_budget(args, "DeepseekV4ForCausalLM")
    else:
        validate_mega_moe_token_budget(args, "DeepseekV4ForCausalLM")
