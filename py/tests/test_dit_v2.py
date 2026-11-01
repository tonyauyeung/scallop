"""AAAAAA's canonical backbone, A3 defaults, and native checkpoint handoffs."""
from copy import deepcopy
import importlib

import pytest
import torch
from torch.func import functional_call

from common import distributed, state_utils
from common.dit import FlowMapAtomDiT
from common.dit_v2 import FlowMapAtomDiTV2
from common.flow_map import TorchFlowMap, get_dit_class
from configs.aaaaaa import get_config
from launchers import learn


@pytest.mark.parametrize('system', ['ace_a_nme', 'aaa', 'ace_aaa_nme', 'aaaaaa', 'chignolin'])
@pytest.mark.parametrize('mode', ['pretrain_v', 'pretrain_flowmap', 'dlogpdt', 'hutchinson'])
def test_backbone_and_stage_defaults_are_system_scoped(monkeypatch, system, mode):
    monkeypatch.setattr('sys.argv', ['learn.py', '--cfg_path', 'configs.' + system,
                                   '--mode', mode, '--device', 'cpu'])
    cfg = learn.setup_config_dict(learn.parse_command_line_arguments())
    a3 = system in ('aaaaaa','chignolin') and mode == 'pretrain_v'
    ace_velocity = system == 'ace_a_nme' and mode == 'pretrain_v'
    assert get_dit_class(cfg.network) is (FlowMapAtomDiTV2 if system in ('aaaaaa','chignolin') else FlowMapAtomDiT)
    assert cfg.optimization.learning_rate == (5e-4 if a3 or mode == 'pretrain_flowmap' else 1e-4)
    assert cfg.optimization.schedule_type == ('cosine' if a3 else 'sqrt')
    assert cfg.optimization.warmup_fraction == (.05 if a3 else 0)
    assert cfg.optimization.total_steps == (400000 if mode == 'pretrain_v' else 100000)
    assert cfg.optimization.decay_steps == (10000 if ace_velocity or mode == 'pretrain_flowmap' else 50000)
    if a3:
        assert cfg.training.recipe == 'A3' and cfg.optimization.bs == 1024
        assert cfg.training.diag_teacher_source == 'conditional'
        assert cfg.optimization.diag_fraction == 1
        monkeypatch.setattr(distributed, 'world_size', lambda: 8)
        distributed.configure_batch(cfg)
        assert cfg.optimization.bs == 1024 and cfg.optimization.local_bs == cfg.optimization.local_diag_bs == 128
        schedule = state_utils.setup_schedule(cfg)
        assert [schedule(k) for k in (0, 20000, 400000)] == [0, 5e-4, 0]
        assert schedule(210000) == pytest.approx(2.5e-4)
    else:
        old = importlib.import_module('configs.' + system).get_config(0)
        assert cfg.optimization.bs == old.optimization.bs


def test_a3_explicit_schedule_overrides_win(monkeypatch):
    monkeypatch.setattr('sys.argv', ['learn.py', '--cfg_path', 'configs.aaaaaa',
        '--mode', 'pretrain_v', '--device', 'cpu', '--learning_rate', '0.0002',
        '--schedule_type', 'sqrt', '--warmup_fraction', '0', '--total_steps', '100',
        '--optimizer', 'radam', '--no-cuda_graphs'])
    cfg = learn.setup_config_dict(learn.parse_command_line_arguments())
    assert cfg.optimization.learning_rate == .0002
    assert cfg.optimization.schedule_type == 'sqrt' and cfg.optimization.warmup_fraction == 0
    assert cfg.optimization.total_steps == 100 and cfg.optimization.optimizer == 'radam'
    assert not cfg.training.cuda_graphs
    cfg.network.dit_version = 'misspelled'
    with pytest.raises(ValueError, match='Unknown network.dit_version'):
        TorchFlowMap(cfg.network)


@pytest.mark.parametrize('density', [False, True])
def test_canonical_shape_positions_initialization_and_muon_groups(density):
    cfg = get_config(0)
    cfg.network.predict_divergence = density
    net = TorchFlowMap(cfg.network)
    dit = net.dit
    assert isinstance(dit, FlowMapAtomDiTV2)
    assert (len(dit.blocks), dit.nhead, dit.d_model, dit.emb_s.linear1.in_features) == (6, 6, 192, 64)
    assert dit.x_embedder.in_features == 3
    phases = torch.arange(63).double()[:, None] / 10000 ** (torch.arange(96).double() / 96)
    torch.testing.assert_close(dit.pos_emb, torch.cat([phases.sin(), phases.cos()], -1)[None].float(), rtol=0, atol=0)
    for name, p in dit.named_parameters():
        if 'modulation' in name or '.linear.' in name or name.endswith('bias'):
            assert p.count_nonzero() == 0, name
    assert sum(p.numel() for p in dit.parameters()) == (5580678 if density else 4173123)
    optimizer, _ = state_utils.setup_optimizer(cfg, net)
    assert len(optimizer.parameter_names['muon']) == (32 if density else 24)
    assert len(optimizer.parameter_names['adamw']) == (66 if density else 50)
    assert 'dit.blocks.0.attn.qkv.weight' in optimizer.parameter_names['muon']
    assert 'dit.blocks.0.mlp.fc1.weight' in optimizer.parameter_names['muon']
    assert 'dit.x_embedder.weight' in optimizer.parameter_names['adamw']


@pytest.mark.parametrize('version', ['canonical_atom_dit_v2', 'aaaaaa_pretrain_v_canonical_v1', None])
def test_canonical_checkpoint_resume_teacher_and_density_handoff(tmp_path, version):
    cfg = get_config(0)
    cfg.network.predict_divergence = False
    cfg.network.dit_kwargs.update(d_model=24, nhead=3, time_embedding_dim=16)
    net = TorchFlowMap(cfg.network)
    # Nonzero outputs ensure checkpoint and derivative comparisons exercise the trunk.
    with torch.no_grad():
        for name, p in net.named_parameters():
            if 'modulation' in name or 'final_layer.linear' in name:
                p.normal_(std=.02)
    opt, schedule = state_utils.setup_optimizer(cfg, net)
    source = state_utils.TrainState(net, opt, schedule, [0.999])
    x, s, t = torch.randn(2, 63, 3), torch.tensor([.2, .4]), torch.tensor([.7, .9])
    net.calc_phi(s, t, x).square().mean().backward()
    opt.step()
    source.step = 7
    source.update_ema()
    saved = source.state_dict()
    saved['model_version'] = version
    if version == 'aaaaaa_pretrain_v_canonical_v1':
        saved['network_architecture']['aaaaaa_variant'] = 'canonical'
    path = tmp_path/'velocity.pt'
    torch.save(saved, path)
    cfg.network.load_path = str(path)
    cfg.network.reset_optimizer = False
    restored, _, _ = state_utils.setup_training_state(cfg, 'cpu')
    assert restored.step == 7 and restored.optimizer.state
    cfg.network.predict_divergence = True
    cfg.network.reset_optimizer = True
    cfg.teacher.load_path = str(path)
    loaded, density, _ = state_utils.setup_training_state(cfg, 'cpu')
    assert loaded.step == 0 and not loaded.optimizer.state
    for name, p in source.ema_params[.999].items():
        torch.testing.assert_close(loaded.ema_params[.999][name], p, rtol=0, atol=0)
    teacher = state_utils.load_teacher_params(cfg, loaded, 'cpu')
    expected = functional_call(net, source.ema_params[.999], (s, t, x), {'method': 'calc_phi'})
    actual = functional_call(density, teacher, (s, t, x), {'method': 'calc_phi'})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for candidate in (restored.net, density):
        fn = lambda tt: candidate.calc_phi(s, tt, x)
        y, j = torch.func.jvp(fn, (t,), (torch.ones_like(t),))
        yr, jr = torch.func.jvp(lambda tt: net.calc_phi(s, tt, x), (t,), (torch.ones_like(t),))
        torch.testing.assert_close(y, yr, rtol=0, atol=0)
        torch.testing.assert_close(j, jr, rtol=0, atol=0)
        grad = torch.autograd.grad(j.square().mean(), candidate.dit.blocks[0].attn.qkv.weight)[0]
        assert torch.isfinite(grad).all() and grad.norm() > 0
    eval_cfg = deepcopy(cfg)
    eval_cfg.network.predict_divergence = False
    state_utils.load_evaluation_model(eval_cfg, path, "cpu", use_ema=False)
    TorchFlowMap(eval_cfg.network)  # Sweep metadata must not leak into constructor kwargs.
    for bad_version in ('original_atom_dit_v1', 'aaaaaa_pretrain_v_packed_v1'):
        bad = {**saved, 'model_version': bad_version}
        before = deepcopy(density.state_dict())
        with pytest.raises(ValueError, match='Incompatible backbone'):
            loaded.load_state_dict(bad, load_optimizer=False)
        for name, p in density.state_dict().items():
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    original = deepcopy(cfg)
    original.network.dit_version = 'original_atom_dit_v1'
    with pytest.raises(ValueError, match='Incompatible backbone'):
        state_utils.load_evaluation_model(original, path, "cpu", use_ema=False)


# Frozen pre-simplification forward; preserve this independent loop oracle.
from typing import Optional, Tuple
from common.dit_v2 import _no_autocast
from common.se3_utils import remove_mean
def _original_v2_forward(
    self, x: torch.Tensor, ss: torch.Tensor, ts: torch.Tensor, train: bool = False,
    *, output: str = "both",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return selected (velocity, raw density); ``ts`` is the signed gap t-s."""
    if output not in ("velocity", "divergence", "both"):
        raise ValueError(f"Unknown DiT output: {output!r}")
    if output == "divergence" and not self.predict_divergence:
        raise ValueError("Divergence output requires a density head")
    want_velocity = output != "divergence"
    want_divergence = output != "velocity" and self.predict_divergence
    assert x.ndim == ss.ndim == ts.ndim == 3, "Input must have 3 dimensions"
    assert x.shape[1] == self.n_particles, "Particle dimension mismatch"
    if not self.com_aware:
        x = remove_mean(x)
    del train
    with _no_autocast(x):
        c = self.emb_s(ss.reshape(x.shape[0])) + self.emb_t(ts.reshape(x.shape[0]))
    tokens = self.x_embedder(x) + self.pos_emb.to(dtype=x.dtype)
    for block in self.blocks[:self.num_shared_layers]:
        tokens = block(tokens, c)
    velocity = density = None
    if want_velocity:
        velocity_tokens = tokens
        for block in self.blocks[self.num_shared_layers:]:
            velocity_tokens = block(velocity_tokens, c)
        velocity = self.final_layer(velocity_tokens, c)
        if not self.com_aware:
            velocity = remove_mean(velocity)
    if want_divergence:
        for block in self.div_blocks:
            tokens = block(tokens, c)
        density = self.div_final_layer(tokens, c)
        if not self.vectorized_divergence:
            density = density.sum(dim=(1, 2))
    return velocity, density


@pytest.mark.parametrize('shared', [0, 1, 2])
@pytest.mark.parametrize('output', ['velocity', 'divergence', 'both'])
def test_indexed_blocks_match_original_outputs_jvp_gradients_and_update(shared, output):
    import types
    from test_simplification import tiny_config
    cfg = tiny_config('canonical_atom_dit_v2')
    cfg.network.dit_kwargs.num_shared_layers = shared
    torch.manual_seed(52)
    net = TorchFlowMap(cfg.network).dit
    with torch.no_grad():
        for p in net.parameters(): p.add_(torch.randn_like(p) * .02)
    reference = deepcopy(net)
    reference.forward = types.MethodType(_original_v2_forward, reference)
    assert net.state_dict().keys() == reference.state_dict().keys()
    x = torch.randn(3, 22, 3)
    s, t = torch.full((3, 1, 1), .2), torch.full((3, 1, 1), .5)
    def evaluate(model):
        fn = lambda xx: tuple(v for v in model(xx, s, t, output=output) if v is not None)
        values, tangents = torch.func.jvp(fn, (x,), (torch.ones_like(x),))
        assert any(v.abs().max() > 1e-5 for v in values)
        sum(v.square().sum() for v in (*values, *tangents)).backward()
        return values, tangents
    torch.testing.assert_close(evaluate(net), evaluate(reference), rtol=0, atol=0)
    for (_, a), (_, b) in zip(net.named_parameters(), reference.named_parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None: torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    for model in (net, reference): torch.optim.SGD(model.parameters(), lr=.001).step()
    for name, value in net.state_dict().items(): assert torch.equal(value, reference.state_dict()[name])
