"""Behavior and work-count gates for the camera-ready simplification.

Config fingerprints track the current camera-ready training defaults.
Numerical expectations below use independent equations
or saved tensors, never the implementation under test as their own oracle.
"""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch

from common import distributed, state_utils, ts_eval
from common.flow_map import TorchFlowMap, _bcast
from configs.ace_a_nme import get_config
from launchers import learn

FIXTURES = json.loads(Path(__file__).with_name('fixtures').joinpath('simplification_configs.txt').read_text())
NAMES = ['ace_a_nme', 'aaa', 'ace_aaa_nme', 'aaaaaa', 'chignolin']
EXTRA = ['--batch_size', '12', '--total_steps', '17', '--network_load_path', 'student.pt',
         '--teacher_load_path', 'teacher.pt', '--resume_optimizer', '--visual_freq', '9',
         '--save_freq', '7', '--wandb_project', 'parity', '--mixed_precision', 'bf16_trunk',
         '--use_weight', '--logvar_channels', '8', '--wandb_name_suffix', '_parity', '--log_freq', '3']


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize('name', NAMES)
def test_full_current_configs_match_frozen_baseline(name, monkeypatch):
    monkeypatch.setenv('WANDB_ENTITY', 'RKOY')
    module = importlib.import_module(f'configs.{name}')
    for row in range(4):
        cfg = module.get_config(row, '/tmp/parity-data', '/tmp/parity-out')
        assert fingerprint(cfg) == FIXTURES['base'][f'{name}/{row}']
        other = module.get_config(row, '/tmp/parity-data', '/tmp/parity-out')
        cfg.network.dit_kwargs.d_model = 1
        cfg.training.ema_facs.append(.5)
        cfg.logging.n_sampling_steps.append(99)
        assert fingerprint(other) == FIXTURES['base'][f'{name}/{row}']


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('mode', ['pretrain_v', 'pretrain_flowmap', 'dlogpdt', 'hutchinson', 'cond_trace'])
@pytest.mark.parametrize('override', [False, True])
def test_full_cli_resolution_matches_baseline(name, mode, override, monkeypatch):
    monkeypatch.setenv('WANDB_ENTITY', 'RKOY')
    monkeypatch.setattr(sys, 'argv', ['learn', '--cfg_path', f'configs.{name}', '--mode', mode,
        '--device', 'cpu', '--dataset_location', '/tmp/parity-data', '--output_folder', '/tmp/parity-out',
        *(EXTRA if override else [])])
    cfg = learn.setup_config_dict(learn.parse_command_line_arguments())
    distributed.configure_batch(cfg)
    assert fingerprint(cfg) == FIXTURES['resolved'][f'{name}/{mode}/{override}']


@pytest.mark.parametrize('legacy', ['ala3_implicit', 'ala6_implicit'])
def test_retired_spellings_fail_in_fresh_import_and_clis(legacy, tmp_path):
    result = subprocess.run([sys.executable, '-c',
        f'import sys; sys.path.insert(0, {str(Path(__file__).parents[1])!r}); import configs.{legacy}'],
        capture_output=True, text=True)
    assert result.returncode != 0 and 'ModuleNotFoundError' in result.stderr
    from launchers import eval_ode_int, eval_peptide_ts
    with pytest.raises(SystemExit):
        eval_ode_int.parse_args(['--system', legacy, '--ckpt_path', 'unused.pt', '--out_dir', str(tmp_path)])
    with pytest.raises(SystemExit):
        eval_peptide_ts.parse_args(['--cfg_path', f'configs.{legacy}', '--ckpt_path', 'unused.pt',
                                   '--out_dir', str(tmp_path), '--eval_recipe', 'v1'])


def test_removed_prediction_flag_is_rejected():
    from launchers.eval_peptide_ts import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--cfg_path', 'configs.aaa', '--ckpt_path', 'unused.pt', '--out_dir', '/tmp/eval',
                    '--predict_divergence'])


@pytest.mark.parametrize('schedule', ['sqrt', 'constant', 'cosine'])
@pytest.mark.parametrize('budget', ['0', '-1', '1.5'])
def test_invalid_budgets_fail_before_startup(schedule, budget, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['learn', '--cfg_path', 'configs.aaa', '--schedule_type', schedule,
                                   '--total_steps', budget, '--device', 'cpu'])
    with pytest.raises((ValueError, SystemExit)):
        learn.setup_config_dict(learn.parse_command_line_arguments())


def tiny_config(version='original_atom_dit_v1', density=True, vectorized=True):
    cfg = get_config(0)
    cfg.network.dit_version = version
    cfg.network.dit_kwargs = dict(d_model=24, num_layers=2, num_shared_layers=1,
        num_div_layers=1, nhead=3, mlp_ratio=2, time_embedding_dim=16)
    cfg.network.predict_divergence = density
    cfg.network.vectorized_divergence = vectorized
    cfg.network.divergence_rescale = 1.
    cfg.network.divergence_mode = 'dlogpdt' if density else 'pretrain_flowmap'
    cfg.network.unscale_factor = .25
    return cfg


def saved_model(cfg, path):
    torch.manual_seed(61)
    net = TorchFlowMap(cfg.network)
    with torch.no_grad():
        for p in net.parameters(): p.add_(.02 * torch.randn_like(p))
    optimizer, schedule = state_utils.setup_optimizer(cfg, net)
    state = state_utils.TrainState(net, optimizer, schedule, [.999])
    state.step = 7
    state.best_metrics = {'ess': .4}
    with torch.no_grad():
        for value in state.ema_params[.999].values(): value.add_(.001)
    torch.save(state.state_dict(), path)
    return state


@pytest.mark.parametrize('version', ['original_atom_dit_v1', 'canonical_atom_dit_v2'])
@pytest.mark.parametrize('density,vectorized', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('ema', [False, True])
def test_inference_one_read_no_optimizer_exact_weights_and_buffers(tmp_path, monkeypatch, version, density, vectorized, ema):
    cfg = tiny_config(version, density, vectorized)
    path = tmp_path / 'model.pt'
    source = saved_model(cfg, path)
    expected = deepcopy(source.net)
    if ema:
        tensors = {**dict(expected.named_parameters()), **dict(expected.named_buffers())}
        with torch.no_grad():
            for name, value in source.ema_params[.999].items(): tensors[name].copy_(value)
    load = Mock(wraps=torch.load)
    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(state_utils, 'setup_optimizer', Mock(side_effect=AssertionError('inference optimizer')))
    net, step, fac = state_utils.load_evaluation_model(cfg, path, 'cpu', use_ema=ema,
                                                     require_saved_ema=ema, velocity_only=not density)
    assert load.call_count == 1 and step == 7 and fac == (.999 if ema else None)
    assert not net.training
    assert net.state_dict().keys() == expected.state_dict().keys()
    for name, value in state_utils._model_tensors(net).items():
        assert torch.equal(value, state_utils._model_tensors(expected)[name])
    x, s, t = torch.randn(3, 22, 3), torch.tensor([.1, .2, .3]), torch.tensor([.8, .7, .9])
    actual = net.calc_phi(s, t, x, return_div=density)
    wanted = expected.calc_phi(s, t, x, return_div=density)
    assert actual[0].abs().max() > 0 if density else actual.abs().max() > 0
    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


@pytest.mark.parametrize('defect', ['missing', 'partial', 'extra', 'shape', 'nan', 'buffer_shape'])
def test_strict_ema_rejects_corruption(tmp_path, defect):
    cfg = tiny_config()
    path = tmp_path / 'model.pt'
    saved_model(cfg, path)
    saved = torch.load(path, weights_only=False)
    ema = saved['ema_params'][.999]
    key = next(iter(saved['params']))
    if defect == 'missing': saved.pop('ema_params')
    elif defect == 'partial': ema.pop(key)
    elif defect == 'extra': ema['unknown'] = torch.zeros(1)
    elif defect == 'shape': ema[key] = torch.zeros(1)
    elif defect == 'nan': ema[key].fill_(float('nan'))
    else: ema['dit.pos_emb'] = torch.zeros(1)
    torch.save(saved, path)
    with pytest.raises(ValueError, match='EMA'):
        state_utils.load_evaluation_model(cfg, path, 'cpu', require_saved_ema=True)


@pytest.mark.parametrize('missing', [True, False])
def test_generic_ema_fallback_uses_raw_weights(tmp_path, missing):
    cfg = tiny_config()
    path = tmp_path / 'model.pt'
    state = saved_model(cfg, path)
    saved = state.state_dict()
    key = next(iter(saved['params']))
    if missing: saved.pop('ema_params')
    else: saved['ema_params'][.999].pop(key)
    torch.save(saved, path)
    net, step, fac = state_utils.load_evaluation_model(cfg, path, 'cpu')
    assert step == 7 and fac == .999
    for name, value in net.state_dict().items():
        expected = saved['params'][name] if missing or name == key else saved['ema_params'][.999][name]
        assert torch.equal(value, expected)


def test_evaluation_rejects_wrong_head_and_legacy_pickle(tmp_path):
    path = tmp_path / 'model.pt'
    velocity = tiny_config(density=False)
    saved_model(velocity, path)
    with pytest.raises(ValueError, match='no density head'):
        state_utils.load_evaluation_model(tiny_config(), path, 'cpu')
    saved_model(tiny_config(), path)
    with pytest.raises(ValueError, match='velocity checkpoint'):
        state_utils.load_evaluation_model(velocity, path, 'cpu', velocity_only=True)
    with pytest.raises(ValueError, match='native .pt'):
        state_utils.load_evaluation_model(velocity, tmp_path / 'legacy.pkl', 'cpu')


@pytest.mark.parametrize('optimizer', ['radam', 'muon_adamw'])
@pytest.mark.parametrize('stage', ['scratch', 'reset', 'resume', 'partial'])
def test_training_constructs_one_optimizer_and_preserves_reset_policy(tmp_path, monkeypatch, optimizer, stage):
    cfg = tiny_config(density=stage != 'partial')
    cfg.optimization.optimizer = optimizer
    path = tmp_path / 'model.pt'
    source = saved_model(cfg, path)
    cfg.network.predict_divergence = True
    cfg.network.reset_best_metrics = True
    cfg.network.reset_optimizer = stage != 'resume'
    cfg.network.load_path = '' if stage == 'scratch' else str(path)
    setup = Mock(wraps=state_utils.setup_optimizer)
    monkeypatch.setattr(state_utils, 'setup_optimizer', setup)
    state, net, _ = state_utils.setup_training_state(cfg, 'cpu')
    assert setup.call_count == 1
    assert state.step == (7 if stage == 'resume' else 0)
    assert state.best_metrics == (source.best_metrics if stage == 'resume' else {})
    if stage != 'scratch':
        for key, value in source.net.state_dict().items():
            assert torch.equal(net.state_dict()[key], value)
            assert torch.equal(state.ema_params[.999][key], source.ema_params[.999][key])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('kind', ['scalar', 'zero_dim', 'batch'])
def test_shared_broadcast_preserves_values_and_derivatives(dtype, kind):
    x = torch.randn(3, 2, 3, dtype=dtype, requires_grad=True)
    coefficient = 2. if kind == 'scalar' else torch.tensor(2. if kind == 'zero_dim' else [1., 2., 3.], dtype=dtype)
    if kind == 'scalar': assert _bcast(coefficient, x) is coefficient
    expected = x * (coefficient.reshape(-1, 1, 1) if kind != 'scalar' else coefficient)
    actual = x * _bcast(coefficient, x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.autograd.grad(actual.sum(), x)[0], torch.autograd.grad(expected.sum(), x)[0])


@pytest.mark.parametrize('version', ['original_atom_dit_v1', 'canonical_atom_dit_v2'])
@pytest.mark.parametrize('mode', ['pretrain_v', 'pretrain_flowmap', 'dlogpdt'])
@pytest.mark.parametrize('lift', [False, True])
def test_cpu_collection_preserves_sampling_dispatch_and_rng(version, mode, lift, tmp_path):
    cfg = tiny_config(version, mode == 'dlogpdt')
    cfg.network.divergence_mode = mode
    cfg.network.com_aware = lift
    net = saved_model(cfg, tmp_path / 'model.pt').net
    x = torch.randn(7, 22, 3)
    for steps in [1, 3]:
        rng = torch.get_rng_state()
        expected = ts_eval.sample_model(net, x, n_steps=steps, batch_size=3)
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        actual = ts_eval.sample_model(net, x, n_steps=steps, batch_size=3, output_device='cpu')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), after)
        assert actual[0].device.type == 'cpu'


@pytest.mark.parametrize('name', ['ace_a_nme', 'aaa', 'ace_aaa_nme', 'aaaaaa'])
def test_readme_commands_resolve_without_launching(name, monkeypatch):
    import re
    import shlex
    root = Path(__file__).parents[2]
    blocks = re.findall(r'```bash\n(.*?)```', (root / 'README.md').read_text(), re.S)
    commands = [block for block in blocks if 'python py/launchers/learn.py' in block]
    assert len(commands) == 3
    for mode, block in zip(('pretrain_v', 'dlogpdt', 'hutchinson'), commands):
        subprocess.run(['bash', '-n'], input=block, text=True, check=True)
        argv = shlex.split(block.replace('\\\n', ' ').replace('<system>', name), comments=True)[1:]
        monkeypatch.setattr(sys, 'argv', argv)
        args = learn.parse_command_line_arguments()
        cfg = learn.setup_config_dict(args)
        velocity = mode == 'pretrain_v'
        a3 = velocity and name in ('aaaaaa', 'chignolin')
        ace_velocity = velocity and name == 'ace_a_nme'
        assert args.mode == cfg.network.divergence_mode == mode
        assert not args.wandb and not args.require_wandb
        assert cfg.problem.ts_data_dir == 'datasets/ts'
        assert cfg.logging.output_folder == f'<run-dir>/{mode}'
        assert cfg.optimization.total_steps == (400000 if velocity else 100000)
        assert cfg.optimization.bs == 1024
        assert cfg.optimization.diag_fraction == (1 if velocity else .5)
        assert cfg.optimization.optimizer == ('radam' if ace_velocity else 'muon_adamw')
        assert cfg.optimization.learning_rate == (5e-4 if a3 else 1e-4)
        assert cfg.optimization.schedule_type == ('cosine' if a3 else 'sqrt')
        assert cfg.optimization.warmup_fraction == (.05 if a3 else 0)
        assert cfg.optimization.decay_steps == (10000 if ace_velocity else 50000)
        assert cfg.optimization.weight_decay == (0. if ace_velocity else 1e-4)
        assert cfg.optimization.clip == 10
        assert cfg.training.loss_type == 'lsd' and cfg.training.stopgrad_type == 'stopped'
        assert cfg.training.seed == 42 and cfg.training.ema_facs == [.999]
        assert cfg.training.density_ema_fac == cfg.training.diag_tmax == .999
        assert cfg.problem.com_augmentation and cfg.network.com_aware
        assert cfg.network.mixed_precision == 'off' and not cfg.network.use_bfloat16
        assert cfg.network.reset_optimizer
        assert cfg.logging.log_freq == 100
        assert cfg.logging.save_freq == cfg.logging.visual_freq == 10000
        assert cfg.logging.plot_bs == 1000
        assert cfg.logging.n_sampling_steps == [4]
        assert cfg.logging.importance_weight_clip == .002
        assert cfg.network.predict_divergence is not velocity
        if velocity:
            assert cfg.network.load_path == cfg.teacher.load_path == ''
            assert cfg.training.diag_teacher_source == 'conditional'
            assert cfg.training.offdiag_teacher_source == 'self'
        else:
            assert cfg.network.load_path == cfg.teacher.load_path == '<velocity-checkpoint.pt>'
            assert cfg.training.diag_teacher_source == cfg.training.offdiag_teacher_source == 'external'
            assert cfg.teacher.ema_fac == .999
            assert cfg.training.offdiag_time_order == 'forward'
            assert cfg.network.vectorized_divergence and cfg.network.divergence_rescale == 1
            assert cfg.network.div_loss_weighting_type == ('linear' if mode == 'dlogpdt' else 'uniform')

    from launchers import eval_peptide_ts
    evaluations = [block for block in blocks if 'python py/launchers/eval_peptide_ts.py' in block]
    assert len(evaluations) == 2
    for mode, block in zip(('dlogpdt', 'hutchinson'), evaluations):
        subprocess.run(['bash', '-n'], input=block, text=True, check=True)
        argv = shlex.split(block.replace('\\\n', ' ').replace('<system>', name), comments=True)[2:]
        args = eval_peptide_ts.parse_args(argv)
        assert args.mode == mode and args.eval_recipe == 'v1'
        assert args.steps == [{'ace_a_nme': 4, 'aaa': 4, 'ace_aaa_nme': 8,
                               'aaaaaa': 16, 'chignolin': 32}[name]]
        assert args.ckpt_path == f'<{mode}-checkpoint.pt>'
        assert args.out_dir == f'<run-dir>/eval/{mode}'
        assert args.ts_data_dir == 'datasets/ts' and args.batch_size == 2000
        assert args.n_workers == 8
        assert args.n_generate_samples == 1000000 and args.n_eval_samples == 10000
        assert args.posthoc_mala_steps == 1024 and args.mala_adapt_steps == 512
        assert args.mala_checkpoint_interval == 128 and args.logw_quantile_filter == .002
        assert args.reference_splits == ['val', 'test'] and args.wandb_project is None
        assert not args.no_ema


def test_learned_log_variance_has_nonzero_forward_and_gradient():
    cfg = tiny_config()
    cfg.network.use_weight = True
    cfg.network.logvar_channels = 8
    net = TorchFlowMap(cfg.network)
    with torch.no_grad(): net.logvar_linear.weight.normal_(std=.1)
    s, t = torch.tensor([.1, .3]), torch.tensor([.6, .9])
    result = net.calc_phi(s, t, torch.randn(2, 22, 3), return_div=True, calc_weight=True)
    logvar = result[2]
    assert logvar.abs().max() > 0
    logvar.square().sum().backward()
    assert net.logvar_linear.weight.grad.abs().max() > 0


def test_grad_norm_stays_detached_until_scalar_logging(tmp_path, monkeypatch):
    from common import logging, updates
    cfg = tiny_config(density=False)
    cfg.logging.output_folder = str(tmp_path)
    cfg.logging.visual_freq = cfg.logging.save_freq = 0
    net = torch.nn.Linear(3, 1)
    state = state_utils.TrainState(net, torch.optim.SGD(net.parameters(), lr=.01), lambda _: .01, [.999])
    def loss(*args, **kwargs): return net(torch.ones(2, 3)).sum(), {}
    value, metrics = updates.setup_train_step(cfg)(state, loss, None, None, (None,)*7, torch.Generator())
    norm = metrics['train/grad_norm']
    assert isinstance(norm, torch.Tensor) and not norm.requires_grad
    expected = float(norm)
    monkeypatch.setattr(logging, '_wandb_active', lambda: False)
    logging.log_metrics(cfg, state, 'cpu', value, metrics, 1, 0.)
    saved = json.loads((tmp_path / 'metrics.jsonl').read_text())
    assert isinstance(saved['train/grad_norm'], float) and saved['train/grad_norm'] == expected


@pytest.mark.parametrize('version', ['original_atom_dit_v1', 'canonical_atom_dit_v2'])
def test_inference_legacy_metadata_and_incompatible_architecture(tmp_path, version):
    cfg = tiny_config(version)
    cfg.network.dit_kwargs.num_shared_layers = 2
    cfg.network.dit_kwargs.num_div_layers = 0
    path = tmp_path / 'model.pt'
    source = saved_model(cfg, path)
    saved = source.state_dict()
    saved.pop('model_version')
    saved.pop('network_architecture')
    torch.save(saved, path)
    net, step, _ = state_utils.load_evaluation_model(cfg, path, 'cpu', use_ema=False)
    assert step == 7
    for key, value in source.net.state_dict().items():
        assert torch.equal(net.state_dict()[key], value)
    wrong = tiny_config('canonical_atom_dit_v2' if version == 'original_atom_dit_v1' else 'original_atom_dit_v1')
    with pytest.raises(ValueError, match='backbone'):
        state_utils.load_evaluation_model(wrong, path, 'cpu')
    saved['network_architecture'] = dict(num_layers=3, num_shared_layers=2, num_div_layers=0)
    torch.save(saved, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        state_utils.load_evaluation_model(cfg, path, 'cpu')


@pytest.mark.parametrize('name,target', [('aaa', 'ala3_implicit'), ('aaaaaa', 'ala6_implicit')])
@pytest.mark.parametrize('matching', [True, False])
def test_ode_current_name_routes_saved_config_and_checkpoint(tmp_path, monkeypatch, name, target, matching):
    from launchers import eval_ode_int
    from common import ts_systems
    cfg = importlib.import_module(f'configs.{name}').get_config(0)
    cfg.network.dit_kwargs = tiny_config().network.dit_kwargs
    cfg.network.predict_divergence = False
    cfg.network.divergence_mode = 'pretrain_v'
    cfg.network.unscale_factor = .25
    path = tmp_path / f'{target}.pt'
    source = saved_model(cfg, path)
    if not matching:
        cfg.problem.target = 'ace_a_nme'
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(cfg.to_dict()))
    monkeypatch.setattr(sys, 'argv', ['eval', '--system', name, '--training_config', str(config_path),
        '--ckpt_path', str(path), '--out_dir', str(tmp_path / 'eval'), '--device', 'cpu'])
    energy = Mock()
    monkeypatch.setattr(ts_systems, 'ParallelEnergy', Mock(return_value=energy))
    monkeypatch.setattr(ts_systems, 'ensure_data', Mock(return_value={'source': 'fixture'}))
    def run(args, resolved, cfg_path, system, paths, pool, dim):
        assert system.target == resolved.problem.target == target
        net, description, step = eval_ode_int._load_net(args, resolved, 'cpu')
        assert step == 7 and description == 'EMA 0.999'
        for key, value in state_utils._model_tensors(net).items():
            assert torch.equal(value, source.ema_params[.999][key])
    runner = Mock(side_effect=run)
    monkeypatch.setattr(eval_ode_int, '_run', runner)
    if matching:
        eval_ode_int.main()
        runner.assert_called_once()
        energy.close.assert_called_once()
    else:
        with pytest.raises(SystemExit, match='--system must match'):
            eval_ode_int.main()
        runner.assert_not_called()
        ts_systems.ensure_data.assert_not_called()
