"""Unit tests for flagscale.train.utils.activation_checkpoint."""

import logging

import pytest
import torch
import torch.nn as nn

from flagscale.logger import logger as fs_logger
from flagscale.train.train_config import ActivationCheckpointConfig
from flagscale.train.utils.activation_checkpoint import (
    DEFAULT_OP_SAC_SAVE_LIST,
    _build_fqn_map,
    _replace_module_by_fqn,
    _warn_opaque_autograd_functions,
    apply_activation_checkpointing,
)


@pytest.fixture(autouse=True)
def _enable_log_propagation():
    """Enable propagation so caplog can capture FlagScale logger output."""
    fs_logger.logger.propagate = True
    yield
    fs_logger.logger.propagate = False


class SimpleBlock(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(self.ln(x))


class SimpleModel(nn.Module):
    def __init__(self, num_layers=4, dim=16):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.layers = nn.ModuleList([SimpleBlock(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

    def fsdp_units(self):
        return list(self.layers)


class FakeGatedDeltaNet(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class HybridBlock(nn.Module):
    def __init__(self, dim=16, use_linear_attn=False):
        super().__init__()
        if use_linear_attn:
            self.attn = FakeGatedDeltaNet(dim)
        else:
            self.attn = nn.MultiheadAttention(dim, num_heads=2, batch_first=True)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        if isinstance(self.attn, FakeGatedDeltaNet):
            x = self.attn(x)
        else:
            x, _ = self.attn(x, x, x)
        return self.mlp(x)


class HybridModel(nn.Module):
    def __init__(self, num_layers=4, dim=16):
        super().__init__()
        self.layers = nn.ModuleList(
            [HybridBlock(dim, use_linear_attn=(i % 2 == 0)) for i in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def fsdp_units(self):
        return list(self.layers)


class TestBuildFqnMap:
    def test_simple_model(self):
        model = SimpleModel(num_layers=2)
        fqn_map = _build_fqn_map(model)

        assert fqn_map[id(model)] == ""
        assert fqn_map[id(model.embed)] == "embed"
        assert fqn_map[id(model.layers[0])] == "layers.0"
        assert fqn_map[id(model.layers[1])] == "layers.1"
        assert fqn_map[id(model.layers[0].linear)] == "layers.0.linear"

    def test_all_modules_present(self):
        model = SimpleModel(num_layers=3)
        fqn_map = _build_fqn_map(model)
        named_modules = dict(model.named_modules())
        assert len(fqn_map) == len(named_modules)


class TestReplaceModuleByFqn:
    def test_replace_top_level(self):
        model = SimpleModel(num_layers=2)
        new_embed = nn.Linear(32, 32)
        _replace_module_by_fqn(model, "embed", new_embed)
        assert model.embed is new_embed

    def test_replace_nested(self):
        model = SimpleModel(num_layers=2)
        new_block = SimpleBlock(dim=16)
        _replace_module_by_fqn(model, "layers.0", new_block)
        assert model.layers[0] is new_block

    def test_replace_deeply_nested(self):
        model = SimpleModel(num_layers=2)
        new_linear = nn.Linear(16, 16)
        _replace_module_by_fqn(model, "layers.1.linear", new_linear)
        assert model.layers[1].linear is new_linear

    def test_replaced_module_in_named_modules(self):
        model = SimpleModel(num_layers=2)
        new_block = SimpleBlock(dim=16)
        _replace_module_by_fqn(model, "layers.0", new_block)
        named = dict(model.named_modules())
        assert named["layers.0"] is new_block


class TestApplyActivationCheckpointing:
    def test_mode_none(self):
        model = SimpleModel(num_layers=4)
        original_layers = [model.layers[i] for i in range(4)]
        ac_config = ActivationCheckpointConfig(mode="none")
        apply_activation_checkpointing(model, ac_config, units=model.fsdp_units())
        for i in range(4):
            assert model.layers[i] is original_layers[i]

    def test_mode_full(self):
        model = SimpleModel(num_layers=4)
        original_layers = [model.layers[i] for i in range(4)]
        ac_config = ActivationCheckpointConfig(mode="full")
        apply_activation_checkpointing(model, ac_config, units=model.fsdp_units())
        for i in range(4):
            assert model.layers[i] is not original_layers[i]

    def test_mode_full_forward_backward(self):
        model = SimpleModel(num_layers=4)
        ac_config = ActivationCheckpointConfig(mode="full")
        apply_activation_checkpointing(model, ac_config, units=model.fsdp_units())

        x = torch.randn(2, 8, 16)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_selective_layer_every_2(self):
        model = SimpleModel(num_layers=4)
        original_layers = [model.layers[i] for i in range(4)]
        ac_config = ActivationCheckpointConfig(mode="selective", selective_ac_option="2")
        apply_activation_checkpointing(model, ac_config, units=model.fsdp_units())

        # Every 2nd layer is wrapped (layer_count 2, 4 → indices 1, 3)
        assert model.layers[0] is original_layers[0]
        assert model.layers[1] is not original_layers[1]
        assert model.layers[2] is original_layers[2]
        assert model.layers[3] is not original_layers[3]

    def test_selective_layer_every_3(self):
        model = SimpleModel(num_layers=6)
        original_layers = [model.layers[i] for i in range(6)]
        ac_config = ActivationCheckpointConfig(mode="selective", selective_ac_option="3")
        apply_activation_checkpointing(model, ac_config, units=model.fsdp_units())

        # layer_count: 1,2,3,4,5,6 → wrapped when count%3==0 → indices 2, 5
        assert model.layers[0] is original_layers[0]
        assert model.layers[1] is original_layers[1]
        assert model.layers[2] is not original_layers[2]
        assert model.layers[3] is original_layers[3]
        assert model.layers[4] is original_layers[4]
        assert model.layers[5] is not original_layers[5]

    def test_selective_layer_forward_backward(self):
        model = SimpleModel(num_layers=4)
        ac_config = ActivationCheckpointConfig(mode="selective", selective_ac_option="2")
        apply_activation_checkpointing(model, ac_config, units=model.fsdp_units())

        x = torch.randn(2, 8, 16)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_selective_op_sac(self):
        model = SimpleModel(num_layers=4)
        original_layers = [model.layers[i] for i in range(4)]
        ac_config = ActivationCheckpointConfig(mode="selective", selective_ac_option="op")
        apply_activation_checkpointing(
            model, ac_config, units=model.fsdp_units(), op_sac_save_list=DEFAULT_OP_SAC_SAVE_LIST
        )
        # All layers should be wrapped
        for i in range(4):
            assert model.layers[i] is not original_layers[i]

    def test_selective_op_sac_forward_backward(self):
        model = SimpleModel(num_layers=4)
        ac_config = ActivationCheckpointConfig(mode="selective", selective_ac_option="op")
        apply_activation_checkpointing(
            model, ac_config, units=model.fsdp_units(), op_sac_save_list=DEFAULT_OP_SAC_SAVE_LIST
        )

        x = torch.randn(2, 8, 16)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_checkpoint_patterns(self):
        model = SimpleModel(num_layers=4)
        original_layers = [model.layers[i] for i in range(4)]
        ac_config = ActivationCheckpointConfig(
            mode="full",
            checkpoint_patterns=[r"layers\.[02]$"],
        )
        apply_activation_checkpointing(model, ac_config)

        assert model.layers[0] is not original_layers[0]
        assert model.layers[1] is original_layers[1]
        assert model.layers[2] is not original_layers[2]
        assert model.layers[3] is original_layers[3]

    def test_no_targets_warns(self, caplog):
        model = SimpleModel(num_layers=4)
        ac_config = ActivationCheckpointConfig(
            mode="full",
            checkpoint_patterns=[r"nonexistent_module"],
        )
        with caplog.at_level(logging.WARNING):
            apply_activation_checkpointing(model, ac_config)
        assert "no modules matched" in caplog.text

    def test_invalid_mode_raises(self):
        model = SimpleModel(num_layers=2)
        ac_config = ActivationCheckpointConfig(mode="selective", selective_ac_option="2")
        # Manually override mode to something invalid after construction
        object.__setattr__(ac_config, "mode", "invalid")
        with pytest.raises(ValueError, match="Invalid activation checkpoint mode"):
            apply_activation_checkpointing(model, ac_config, units=model.fsdp_units())

    def test_memory_budget_without_compile_raises(self):
        model = SimpleModel(num_layers=2)
        ac_config = ActivationCheckpointConfig(mode="memory_budget", memory_budget=0.5)
        with pytest.raises(AssertionError, match="requires torch.compile"):
            apply_activation_checkpointing(model, ac_config, model_compile_enabled=False)

    def test_numerics_match_no_ac(self):
        """Verify AC doesn't change forward pass numerics."""
        torch.manual_seed(42)
        model_ref = SimpleModel(num_layers=4)
        torch.manual_seed(42)
        model_ac = SimpleModel(num_layers=4)

        ac_config = ActivationCheckpointConfig(mode="full")
        apply_activation_checkpointing(model_ac, ac_config, units=model_ac.fsdp_units())

        x = torch.randn(2, 8, 16)
        out_ref = model_ref(x)
        out_ac = model_ac(x)
        assert torch.allclose(out_ref, out_ac, atol=1e-6)

    def test_gradients_match_no_ac(self):
        """Verify AC produces identical gradients."""
        torch.manual_seed(42)
        model_ref = SimpleModel(num_layers=4)
        torch.manual_seed(42)
        model_ac = SimpleModel(num_layers=4)

        ac_config = ActivationCheckpointConfig(mode="full")
        apply_activation_checkpointing(model_ac, ac_config, units=model_ac.fsdp_units())

        x = torch.randn(2, 8, 16)

        out_ref = model_ref(x).sum()
        out_ref.backward()

        out_ac = model_ac(x).sum()
        out_ac.backward()

        for (n1, p1), (n2, p2) in zip(model_ref.named_parameters(), model_ac.named_parameters()):
            assert torch.allclose(p1.grad, p2.grad, atol=1e-6), f"Gradient mismatch at {n1}"


class TestWarnOpaqueAutogradFunctions:
    def test_warns_on_gated_delta_net(self, caplog):
        model = HybridModel(num_layers=4)
        targets = [(f"layers.{i}", model.layers[i]) for i in range(4)]

        with caplog.at_level(logging.WARNING):
            _warn_opaque_autograd_functions(targets)

        assert "opaque" in caplog.text
        assert "layers.0" in caplog.text
        assert "layers.2" in caplog.text

    def test_no_warning_for_standard_modules(self, caplog):
        model = SimpleModel(num_layers=4)
        targets = [(f"layers.{i}", model.layers[i]) for i in range(4)]

        with caplog.at_level(logging.WARNING):
            _warn_opaque_autograd_functions(targets)

        assert caplog.text == ""
