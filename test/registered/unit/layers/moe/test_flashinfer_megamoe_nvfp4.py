from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.moe import flashinfer_mega_moe as adapter_module
from sglang.srt.layers.moe.flashinfer_mega_moe import (
    FlashInferMegaMoEAdapter,
    finalize_flashinfer_megamoe_weights,
)


@pytest.fixture
def api(monkeypatch):
    import flashinfer

    monkeypatch.setattr(
        flashinfer,
        "fp4_quantize",
        Mock(side_effect=AssertionError("adapter must use fused FlashInfer staging")),
    )
    fake = SimpleNamespace(
        PrequantizedMoEWeights=SimpleNamespace,
        BootstrapConfig=SimpleNamespace,
        FleetParams=SimpleNamespace,
        MegaConfig=SimpleNamespace,
        MoEEpTensors=SimpleNamespace,
        Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig=Mock(side_effect=SimpleNamespace),
        Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig=Mock(side_effect=SimpleNamespace),
        MoEEpLayer=Mock(return_value=SimpleNamespace(stage_inputs=Mock())),
    )
    monkeypatch.setattr(FlashInferMegaMoEAdapter, "_api", staticmethod(lambda: fake))
    monkeypatch.setattr(
        adapter_module,
        "get_parallel",
        lambda: SimpleNamespace(attn_tp_size=1, moe_tp_size=1),
    )
    monkeypatch.setattr(
        adapter_module,
        "get_moe_ep_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=1, device_group=object()),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return fake


def make_layer():
    layer = torch.nn.Module()
    layer.num_experts = 4
    layer.num_local_experts = 2
    layer.num_fused_shared_experts = 0
    layer.moe_ep_rank = 1
    layer.intermediate_size_per_partition = 32
    layer.moe_runner_config = SimpleNamespace(
        top_k=2,
        hidden_size=32,
        swiglu_limit=10.0,
        is_gated=True,
        activation="silu",
    )
    for name, value in {
        "w13_weight": torch.zeros((2, 64, 16), dtype=torch.uint8),
        "w2_weight": torch.zeros((2, 32, 16), dtype=torch.uint8),
        "w13_weight_scale": torch.ones((2, 64, 2), dtype=torch.float8_e4m3fn),
        "w2_weight_scale": torch.ones((2, 32, 2), dtype=torch.float8_e4m3fn),
        "w13_weight_scale_2": torch.tensor([[2.0, 2.0], [3.0, 3.0]]),
        "w2_weight_scale_2": torch.tensor([5.0, 6.0]),
        "w13_input_scale": torch.full((4, 2), 4.0),
        "w2_input_scale": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "g1_alphas": torch.tensor([8.0, 12.0]),
        "g1_alphas_up": torch.tensor([8.0, 12.0]),
        "g2_alphas": torch.tensor([15.0, 24.0]),
        "w13_input_scale_quant": torch.tensor(0.25),
        "w2_input_scale_quant": torch.tensor([1.0 / 3.0, 0.25]),
    }.items():
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    return layer


def test_nvfp4_weights_and_epilogue_scales(api):
    layer = make_layer()
    original = layer.w13_weight
    finalize_flashinfer_megamoe_weights(layer, max_num_tokens=64, nvfp4=True)
    args = api.MoEEpLayer.call_args.kwargs
    assert args["weights"].w13.dtype == torch.float4_e2m1fn_x2
    assert args["weights"].w13.data_ptr() == original.data_ptr()
    assert args["weights"].w13_scale.dtype == torch.float8_e4m3fn
    assert args["backend"].megakernel.input_norm_const == 0.25
    assert args["backend"].quantize_input
    assert args["backend"].megakernel.gate_up_clamp == 10.0
    api.Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig.assert_called_once()
    api.Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig.assert_not_called()
    assert layer.w13_weight is None and layer.w2_weight is None
    assert layer.w13_weight_scale is None and layer.w2_weight_scale is None
    assert layer._mega_moe_weights_built
    adapter = layer.flashinfer_megamoe_adapter
    for tokens in (2, 0, 1):
        hidden = torch.zeros((tokens, 32), dtype=torch.bfloat16)
        ids = torch.zeros((tokens, 2), dtype=torch.int32)
        output = torch.empty((64, 32), dtype=torch.bfloat16)
        adapter.stage_into(
            hidden,
            torch.ones((tokens, 2)),
            ids,
            output,
            compile_tokens_per_rank=16,
        )
        tensors = adapter.layer.stage_inputs.call_args.args[0]
        assert tensors.topk_ids is ids
        assert tensors.hidden_states is hidden
        assert tensors.hidden_states.dtype == torch.bfloat16
        assert not hasattr(tensors, "scales")
        torch.testing.assert_close(tensors.fc1_alpha, torch.tensor([8.0, 12.0]))
        torch.testing.assert_close(tensors.fc2_alpha, torch.tensor([15.0, 24.0]))
        torch.testing.assert_close(
            tensors.fc1_norm_const, torch.tensor([1.0 / 3.0, 0.25])
        )


@pytest.mark.parametrize("gated,activation", [(False, "silu"), (True, "gelu")])
def test_nvfp4_rejects_unsupported_activation(api, gated, activation):
    layer = make_layer()
    layer.moe_runner_config.is_gated = gated
    layer.moe_runner_config.activation = activation
    with pytest.raises(ValueError, match="gated SiLU"):
        finalize_flashinfer_megamoe_weights(layer, max_num_tokens=64, nvfp4=True)
    api.MoEEpLayer.assert_not_called()
    assert layer.w13_weight is not None


def test_nvfp4_rejects_distinct_gate_up_alpha(api):
    layer = make_layer()
    layer.g1_alphas_up[0] = 9.0
    with pytest.raises(ValueError, match="matching gate/up"):
        finalize_flashinfer_megamoe_weights(layer, max_num_tokens=64, nvfp4=True)
    api.MoEEpLayer.assert_not_called()
    assert layer.w13_weight is not None


def test_nvfp4_modelopt_reuses_scale_processing(api, monkeypatch):
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.layers.quantization import modelopt_quant

    monkeypatch.setattr(
        modelopt_quant,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.FLASHINFER_MEGAMOE,
    )
    monkeypatch.setattr(
        modelopt_quant,
        "get_moe_a2a_backend",
        lambda: SimpleNamespace(is_megamoe=lambda: True),
    )
    monkeypatch.setattr(modelopt_quant, "_use_nvfp4_dispatch", lambda: False)
    monkeypatch.setattr(
        modelopt_quant,
        "get_exec",
        lambda: SimpleNamespace(
            moe=SimpleNamespace(flashinfer_megamoe_max_num_tokens=64)
        ),
    )
    method = SimpleNamespace(
        enable_flashinfer_trtllm_moe=False,
        enable_flashinfer_cutlass_moe=False,
        enable_flashinfer_cutedsl_moe=False,
        _is_cutedsl_v2_standard=False,
        quant_config=SimpleNamespace(use_per_token_activation=False),
    )
    layer = make_layer()
    modelopt_quant.ModelOptNvFp4FusedMoEMethod.process_weights_after_loading(
        method, layer
    )
    scales = layer.flashinfer_megamoe_adapter._scale_tensors
    torch.testing.assert_close(scales["fc1_alpha"], torch.tensor([8.0, 12.0]))
    torch.testing.assert_close(scales["fc2_alpha"], torch.tensor([15.0, 24.0]))
    torch.testing.assert_close(
        scales["fc1_norm_const"], torch.tensor([1.0 / 3.0, 0.25])
    )


def test_w4a8_backend_and_scale_format_unchanged(api):
    adapter = FlashInferMegaMoEAdapter(
        num_experts=4,
        num_local_experts=2,
        top_k=2,
        hidden_size=32,
        intermediate_size=32,
        max_num_tokens=64,
        activation_clamp=10.0,
    )
    adapter.finalize_weights(
        torch.zeros((2, 64, 16), dtype=torch.uint8),
        torch.zeros((2, 32, 16), dtype=torch.uint8),
        torch.full((2, 64, 1), 127, dtype=torch.uint8),
        torch.full((2, 32, 1), 127, dtype=torch.uint8),
    )
    args = api.MoEEpLayer.call_args.kwargs
    assert args["weights"].w13_scale.dtype == torch.float8_e8m0fnu
    api.Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig.assert_called_once()
    api.Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig.assert_not_called()
    assert adapter._scale_tensors == {}


@pytest.mark.parametrize(
    "sm120,per_token,w4a16,match",
    [
        (False, False, False, "requires SM120"),
        (True, True, False, "per-tensor"),
        (True, False, True, "W4A16"),
    ],
)
def test_nvfp4_rejects_unsupported_modes(monkeypatch, sm120, per_token, w4a16, match):
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.layers.quantization import modelopt_quant

    monkeypatch.setattr(
        modelopt_quant,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.FLASHINFER_MEGAMOE,
    )
    monkeypatch.setattr(
        modelopt_quant,
        "get_moe_a2a_backend",
        lambda: SimpleNamespace(is_megamoe=lambda: True),
    )
    monkeypatch.setattr(
        modelopt_quant, "get_platform", lambda: SimpleNamespace(is_sm120=sm120)
    )
    method = SimpleNamespace(
        quant_config=SimpleNamespace(use_per_token_activation=per_token, is_w4a16=w4a16)
    )
    with pytest.raises(ValueError, match=match):
        modelopt_quant.ModelOptNvFp4FusedMoEMethod.create_moe_runner(
            method, torch.nn.Module(), SimpleNamespace()
        )
