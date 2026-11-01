"""Stage budgets, explicit overrides and the optimizer's actual cosine warmup."""
from copy import deepcopy

import pytest
import torch

from common import state_utils, updates
from configs.ace_a_nme import get_config
from launchers import learn
from test_muon_optimizer import model


@pytest.mark.parametrize('total,warmup', [(400000, 20000), (100000, 5000)])
def test_schedule_warms_up_then_decays_over_entire_stage(total, warmup):
    cfg = get_config(0)
    cfg.optimization.update(schedule_type="cosine", learning_rate=5e-4, warmup_fraction=.05)
    cfg.optimization.total_steps = total
    schedule = state_utils.setup_schedule(cfg)
    assert schedule(0) == 0
    assert schedule(warmup // 2) == pytest.approx(2.5e-4)
    assert schedule(warmup) == pytest.approx(5e-4)
    assert schedule((total + warmup) // 2) == pytest.approx(2.5e-4)
    assert schedule(total) == schedule(total + 1) == 0
    assert 0 < schedule(total - 1) < 1e-10
    # The old sqrt onset must not shorten a cosine stage.
    cfg.optimization.decay_steps = 1
    other = state_utils.setup_schedule(cfg)
    assert other(warmup) == schedule(warmup)


@pytest.mark.parametrize('mode,total', [('pretrain_v',400000),('pretrain_flowmap',100000),
                                       ('dlogpdt',100000),('hutchinson',100000)])
@pytest.mark.parametrize('config', ['ace_a_nme', 'ace_aaa_nme', 'aaa', 'aaaaaa'])
def test_resolved_stage_defaults_and_explicit_overrides(mode, total, config, monkeypatch):
    argv=['learn.py','--cfg_path','configs.'+config,'--mode',mode,'--device','cpu','--from_scratch']
    monkeypatch.setattr('sys.argv',argv)
    cfg=learn.setup_config_dict(learn.parse_command_line_arguments())
    assert cfg.optimization.total_steps == total
    a3 = config == "aaaaaa" and mode == "pretrain_v"
    ace_velocity = config == "ace_a_nme" and mode == "pretrain_v"
    lr = 5e-4 if a3 or mode == "pretrain_flowmap" else 1e-4
    assert cfg.optimization.learning_rate == lr
    assert cfg.training.stopgrad_type == ("convex" if mode == "pretrain_flowmap" else "stopped")
    assert cfg.optimization.weight_decay == (0. if ace_velocity else 1e-4)
    assert cfg.optimization.schedule_type == ('cosine' if a3 else 'sqrt')
    assert cfg.optimization.warmup_fraction == (.05 if a3 else 0)
    decay = 10000 if ace_velocity or mode == "pretrain_flowmap" else 50000
    assert cfg.optimization.decay_steps == decay
    schedule=state_utils.setup_schedule(cfg)
    if a3:
        assert schedule(0) == schedule(total) == 0
        assert schedule(20000) == lr
    else:
        assert schedule(0) == schedule(decay) == lr
        assert schedule(4*decay) == lr/2
    monkeypatch.setattr('sys.argv',argv+['--learning_rate','0.002','--weight_decay','0.003',
                        '--schedule_type','cosine','--warmup_fraction','0.1','--total_steps','2000',
                        '--stopgrad_type','stopped','--decay_steps','12345'])
    cfg=learn.setup_config_dict(learn.parse_command_line_arguments())
    schedule=state_utils.setup_schedule(cfg)
    assert cfg.optimization.weight_decay == .003
    assert cfg.optimization.decay_steps == 12345
    assert cfg.training.stopgrad_type == "stopped"
    assert schedule(100) == pytest.approx(.001)
    assert schedule(200) == pytest.approx(.002)
    assert schedule(2000) == 0


@pytest.mark.parametrize('value', [-.1, 1., float('nan'), float('inf')])
def test_bad_warmup_rejected(value):
    cfg=get_config(0);cfg.optimization.schedule_type="cosine";cfg.optimization.warmup_fraction=value
    with pytest.raises(ValueError,match='warmup_fraction'):
        state_utils.setup_schedule(cfg)


def test_cosine_schedule_drives_both_groups_and_resume_counter():
    cfg=get_config(0);cfg.optimization.total_steps=100
    cfg.optimization.update(schedule_type="cosine", learning_rate=5e-4, warmup_fraction=.05)
    cfg.optimization.adamw_lr_multiplier=.5
    cfg.training.cuda_graphs=False;cfg.network.predict_divergence=False
    net=model();opt,schedule=state_utils.setup_optimizer(cfg,net)
    state=state_utils.TrainState(net, opt, schedule, [0.999])
    state.step=2
    train=updates.setup_train_step(cfg)
    def loss(*args,**kwargs):
        return sum(p.square().sum() for p in net.parameters()),{}
    train(state,loss,None,None,(None,)*7,torch.Generator())
    assert opt.muon.param_groups[0]['lr'] == pytest.approx(2e-4)
    assert opt.adamw.param_groups[0]['lr'] == pytest.approx(1e-4)
    assert all(pg['weight_decay']==1e-4 for pg in opt.param_groups)
    saved=deepcopy(state.state_dict())
    second_net=model();second_opt,second_schedule=state_utils.setup_optimizer(cfg,second_net)
    second=state_utils.TrainState(second_net, second_opt, second_schedule, [0.999])
    second.load_state_dict(saved)
    assert second.step == 3
    assert second.schedule(second.step) == pytest.approx(3e-4)
    assert all(pg['weight_decay']==1e-4 for pg in second.optimizer.param_groups)


@pytest.mark.parametrize('family,policy', [('lsd','convex'),('psd','stopped'),('esd','stopped')])
def test_flowmap_m4_default_policy_is_lsd_only(family, policy, monkeypatch):
    argv = ['learn.py', '--cfg_path', 'configs.ace_a_nme', '--mode', 'pretrain_flowmap',
            '--device', 'cpu', '--loss_type', family]
    if family == 'psd':
        argv += ['--psd_type', 'midpoint']
    monkeypatch.setattr('sys.argv', argv)
    cfg = learn.setup_config_dict(learn.parse_command_line_arguments())
    assert cfg.training.stopgrad_type == policy
    assert cfg.optimization.learning_rate == 5e-4
    assert cfg.optimization.decay_steps == 10000
