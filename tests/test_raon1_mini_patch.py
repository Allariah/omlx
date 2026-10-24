# SPDX-License-Identifier: Apache-2.0
"""Tests for the vendored Raon-1 mini (``raon1_mini``) mlx-lm patch."""

import importlib
import json
import sys

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten


def _minimal_config(**overrides):
    config = {
        "model_type": "raon1_mini",
        "architectures": ["Raon1MiniForCausalLM"],
        "model_file": "raon_mlx_model.py",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "head_dim": 32,
        "vocab_size": 128,
        "intermediate_size": 64,
        "rms_norm_eps": 1e-5,
        "hidden_act": "situ",
        "first_k_dense_replace": 1,
        "num_experts": 32,
        "num_experts_per_token": 2,
        "moe_intermediate_size": 32,
        "num_shared_experts": 1,
        "q_lora_rank": 32,
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 16,
        "v_head_dim": 16,
        "rda_decay_rank": 16,
        "num_nextn_predict_layers": 0,
        "rda_config": {
            "num_heads": 2,
            "head_dim": 32,
            "conv_kernel_size": 4,
            "full_rank_gate": True,
            "gate_lower_bound": -5.0,
            "rda_layers": [1],
            "mla_layers": [2],
        },
    }
    config.update(overrides)
    return config


def _load_patch_module():
    from omlx.patches.raon1_mini import apply_raon1_mini_patch

    apply_raon1_mini_patch()
    return importlib.import_module("mlx_lm.models.raon1_mini")


def test_apply_registers_raon1_mini_module():
    module = _load_patch_module()

    assert module.__package__ == "mlx_lm.models"
    assert sys.modules["mlx_lm.models.raon1_mini"] is module
    import mlx_lm.models as models_pkg

    assert models_pkg.raon1_mini is module


def test_apply_is_idempotent():
    from omlx.patches.raon1_mini import apply_raon1_mini_patch

    _load_patch_module()
    assert apply_raon1_mini_patch() is False


def test_get_classes_resolves_raon1_mini():
    module = _load_patch_module()
    from mlx_lm.utils import _get_classes

    config = _minimal_config()
    config.pop("model_file")
    model_class, args_class = _get_classes(config)

    assert model_class is module.Model
    assert args_class is module.ModelArgs


def test_model_file_dropped_only_for_the_vendored_remote_module():
    from omlx.patches.raon1_mini import normalize_raon1_mini_model_file

    assert "model_file" not in normalize_raon1_mini_model_file(_minimal_config())
    other = _minimal_config(model_file="something_else.py")
    assert normalize_raon1_mini_model_file(other)["model_file"] == "something_else.py"
    not_raon = _minimal_config(model_type="llama")
    assert (
        normalize_raon1_mini_model_file(not_raon)["model_file"] == "raon_mlx_model.py"
    )


def test_remote_code_gate_trusts_only_the_vendored_module(tmp_path):
    from omlx.utils.model_loading import ensure_model_code_trusted

    ensure_model_code_trusted(
        _minimal_config(), model_path=tmp_path, trust_remote_code=False
    )
    with pytest.raises(ValueError, match="custom"):
        ensure_model_code_trusted(
            _minimal_config(model_file="something_else.py"),
            model_path=tmp_path,
            trust_remote_code=False,
        )


def test_pre_load_dispatch_calls_raon1_mini_patch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "omlx.patches.raon1_mini.apply_raon1_mini_patch",
        lambda: calls.append(True) or True,
    )
    (tmp_path / "config.json").write_text(json.dumps(_minimal_config()))

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(str(tmp_path))

    assert calls == [True]


def _nvfp4_experts(w, gs):
    q = [
        mx.quantize(w[e], 16, 4, mode="nvfp4", global_scale=gs[e])
        for e in range(w.shape[0])
    ]
    return mx.stack([a for a, _ in q]), mx.stack([b for _, b in q])


def _dequant_experts(wq, sq, gs):
    # MLX keeps the expert's amax as ``global_scale`` and applies the NVFP4
    # tensor-scale convention itself, so dequantize through it as well.
    return mx.stack(
        [
            mx.dequantize(
                wq[e], sq[e], group_size=16, bits=4, mode="nvfp4", global_scale=gs[e]
            )
            for e in range(wq.shape[0])
        ]
    )


def test_switch_compat_global_scale_matches_dequantized_reference():
    from omlx.patches.raon1_mini import switch_compat

    mx.random.seed(0)
    experts, out_dims, in_dims = 4, 32, 64
    dense = switch_compat.SwitchLinear(in_dims, out_dims, experts, bias=False)
    dense.weight = mx.random.normal((experts, out_dims, in_dims)) * 3.0
    quant = dense.to_quantized(group_size=16, bits=4, mode="nvfp4")

    assert quant.global_scale.shape == (experts,)
    x = mx.random.normal((3, 1, 1, in_dims))
    idx = mx.array([[0], [2], [3]], dtype=mx.uint32)
    got = quant(x, idx)
    w = _dequant_experts(quant.weight, quant.scales, quant.global_scale)
    ref = mx.stack([x[i, 0, 0] @ w[idx[i, 0]].T for i in range(3)])

    # gather_qmm rounds activations inside the kernel; scale errors are 2688x.
    err = mx.max(mx.abs(got.reshape(3, out_dims) - ref)) / mx.max(mx.abs(ref))
    assert err.item() < 1e-2


def test_switch_compat_keeps_affine_unscaled():
    from omlx.patches.raon1_mini import switch_compat

    dense = switch_compat.SwitchLinear(64, 32, 2, bias=False)
    quant = dense.to_quantized(group_size=64, bits=4, mode="affine")

    assert quant.global_scale is None
    assert quant.biases is not None


def test_nvfp4_checkpoint_loads_strictly_without_remote_code(tmp_path):
    """An NVFP4 Raon checkpoint (per-expert global scales, separate up/gate as
    exported) loads strictly through mlx-lm without importing ``model_file``,
    and its prefill + cached decode match a float model with the dequantized
    expert weights."""
    raon = _load_patch_module()
    config = _minimal_config()
    args = raon.ModelArgs.from_dict(
        {k: v for k, v in config.items() if k != "model_file"}
    )

    mx.random.seed(1)
    source = raon.Model(args)
    params = dict(tree_flatten(source.parameters()))
    for k, v in params.items():
        if "raon_mhc" in k and (".phi_" in k or k.endswith("phi_z")):
            params[k] = mx.random.normal(v.shape) * 0.02
        elif k.endswith("moe.gate.weight"):
            params[k] = mx.random.normal(v.shape) * 0.1

    ckpt = dict(params)
    reference = dict(params)
    quant = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    sw = "model.layers.1.moe.switch_mlp."
    h = args.moe_intermediate_size
    gate_up = ckpt.pop(sw + "gate_up.weight")
    up, gate = gate_up[:, :h], gate_up[:, h:]
    shared_gs = mx.max(mx.abs(gate_up), axis=(-2, -1)).astype(mx.float32)
    for name, w, gs in (
        ("up_proj", up, shared_gs),
        ("gate_proj", gate, shared_gs),
    ):
        ckpt[sw + f"{name}.weight"], ckpt[sw + f"{name}.scales"] = _nvfp4_experts(w, gs)
        ckpt[sw + f"{name}.global_scale"] = gs
        quant[sw + name] = {
            **{k: quant[k] for k in ("group_size", "bits", "mode")},
            "global_scale": True,
        }
    down = ckpt.pop(sw + "down_proj.weight")
    down_gs = mx.max(mx.abs(down), axis=(-2, -1)).astype(mx.float32)
    ckpt[sw + "down_proj.weight"], ckpt[sw + "down_proj.scales"] = _nvfp4_experts(
        down, down_gs
    )
    ckpt[sw + "down_proj.global_scale"] = down_gs
    quant[sw + "down_proj"] = {
        "group_size": 16,
        "bits": 4,
        "mode": "nvfp4",
        "global_scale": True,
    }

    reference[sw + "gate_up.weight"] = mx.concatenate(
        [
            _dequant_experts(
                ckpt[sw + "up_proj.weight"], ckpt[sw + "up_proj.scales"], shared_gs
            ),
            _dequant_experts(
                ckpt[sw + "gate_proj.weight"], ckpt[sw + "gate_proj.scales"], shared_gs
            ),
        ],
        axis=1,
    )
    reference[sw + "down_proj.weight"] = _dequant_experts(
        ckpt[sw + "down_proj.weight"], ckpt[sw + "down_proj.scales"], down_gs
    )

    mx.save_safetensors(str(tmp_path / "model.safetensors"), ckpt)
    (tmp_path / "config.json").write_text(json.dumps({**config, "quantization": quant}))
    assert not (tmp_path / "raon_mlx_model.py").exists()

    from mlx_lm.utils import load_model
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(str(tmp_path))
    loaded, loaded_config = load_model(tmp_path, strict=True)

    assert "model_file" not in loaded_config
    assert type(loaded).__module__ == "mlx_lm.models.raon1_mini"
    glu = loaded.layers[1].moe.switch_mlp
    assert glu.gate_up.global_scale.shape == (args.num_experts,)
    assert glu.down_proj.global_scale.shape == (args.num_experts,)

    ref_model = raon.Model(args)
    ref_model.update(tree_unflatten(list(reference.items())))

    tokens = mx.array([[3, 17, 42, 5, 99, 64]], dtype=mx.int32)
    got_cache, ref_cache = loaded.make_cache(), ref_model.make_cache()
    got = loaded(tokens, cache=got_cache)[:, -1]
    ref = ref_model(tokens, cache=ref_cache)[:, -1]
    assert mx.allclose(got, ref, atol=2e-3, rtol=2e-3)

    step = mx.array([[7]], dtype=mx.int32)
    got = loaded(step, cache=got_cache)[:, -1]
    ref = ref_model(step, cache=ref_cache)[:, -1]
    assert mx.allclose(got, ref, atol=2e-3, rtol=2e-3)

    assert isinstance(loaded.layers[0].self_attn.q_proj, nn.Linear)


def test_raon1_mini_is_discovered_as_llm(tmp_path):
    from omlx.model_discovery import detect_model_type

    (tmp_path / "config.json").write_text(json.dumps(_minimal_config()))

    assert detect_model_type(tmp_path) == "llm"
