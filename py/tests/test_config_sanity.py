"""Sanity checks for the peptide configs (ace_a_nme, aaa,
ace_aaa_nme, aaaaaa, chignolin).

``get_config`` builds the ConfigDict without touching OpenMM (only the later
``datasets.setup_target`` reads the TS training scale), so we can validate
structure and dimensional consistency without JAX or molecular data.
"""

import importlib
import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest


KEPT_CONFIGS = ["ace_a_nme", "aaa", "ace_aaa_nme", "aaaaaa", "chignolin"]


@pytest.mark.parametrize("name,legacy", [("aaa", "ala3_implicit"), ("aaaaaa", "ala6_implicit")])
def test_current_configs_preserve_internal_system_ids(name, legacy):
    from common.ts_systems import resolve
    current = importlib.import_module(f"configs.{name}")
    for index in range(len(current.experiments)):
        cfg = current.get_config(index, "/tmp/config-data", "/tmp/config-output")
        assert cfg.problem.target == legacy
        assert resolve(cfg).sequence == name.upper()


@pytest.mark.parametrize("name", ["aaa", "aaaaaa"])
@pytest.mark.parametrize("saved_config", [None, "matching", "mismatched"])
def test_ode_config_names_and_saved_target_validation(name, saved_config, monkeypatch, tmp_path):
    from common import ts_systems
    from launchers import eval_ode_int
    cfg = importlib.import_module(f"configs.{name}").get_config(0)
    argv = ["eval_ode_int", "--system", name, "--out_dir", str(tmp_path),
            "--ckpt_path", f"{cfg.problem.target}_velocity.pt", "--device", "cpu"]
    if saved_config:
        saved = cfg if saved_config == "matching" else importlib.import_module("configs.ace_a_nme").get_config(0)
        saved.network.divergence_mode = "pretrain_v"
        path = tmp_path / "config.json"
        path.write_text(saved.to_json())
        argv += ["--training_config", str(path)]
    monkeypatch.setattr("sys.argv", argv)

    class ConfigResolved(Exception):
        pass

    def check_system(system, *args):
        assert system.target == cfg.problem.target
        raise ConfigResolved

    monkeypatch.setattr(ts_systems, "ensure_data", check_system)
    if saved_config == "mismatched":
        with pytest.raises(SystemExit, match="--system must match --training_config"):
            eval_ode_int.main()
    else:
        with pytest.raises(ConfigResolved):
            eval_ode_int.main()


@pytest.mark.parametrize("name", KEPT_CONFIGS)
def test_get_config_structure_and_dims(name):
    mod = importlib.import_module(f"configs.{name}")
    cfg = mod.get_config(0)

    for section in ("training", "problem", "optimization", "network"):
        assert section in cfg, f"{name}: missing config.{section}"

    p = cfg.problem
    assert p.d == p.n_particles * p.n_dimension, (
        f"{name}: d={p.d} != n_particles*n_dimension={p.n_particles * p.n_dimension}"
    )
    assert isinstance(p.target, str) and p.target, f"{name}: empty problem.target"

    # Network must agree with the problem geometry where it declares it.
    net = cfg.network
    if "n_particles" in net:
        assert net.n_particles == p.n_particles, f"{name}: network/problem n_particles mismatch"
    if "n_dimension" in net:
        assert net.n_dimension == p.n_dimension, f"{name}: network/problem n_dimension mismatch"


@pytest.mark.parametrize("name", KEPT_CONFIGS)
def test_all_experiment_indices_build(name):
    """Every slurm_id experiment branch produces a valid loss_type."""
    mod = importlib.import_module(f"configs.{name}")
    n_exp = len(getattr(mod, "experiments", [None] * 4))
    # Keep this declarative config check independent of the training launcher.
    valid_stopgrad = {
        "lsd": ("stopped", "convex"),
        "psd": ("stopped", "convex"),
        "esd": ("stopped", "convex"),
    }
    for slurm_id in range(n_exp):
        cfg = mod.get_config(slurm_id)
        loss_type = cfg.training.loss_type
        assert loss_type in ("lsd", "psd", "esd"), (
            f"{name}[{slurm_id}]: unexpected loss_type {loss_type}"
        )
        assert cfg.training.stopgrad_type in valid_stopgrad[loss_type], (
            f"{name}[{slurm_id}]: stopgrad_type {cfg.training.stopgrad_type!r} is not "
            f"valid for loss_type {loss_type!r} (valid: {valid_stopgrad[loss_type]})"
        )
        psd_type = getattr(cfg.training, "psd_type", None)
        if loss_type == "psd":
            assert psd_type in ("uniform", "midpoint"), (
                f"{name}[{slurm_id}]: loss_type psd needs psd_type, got {psd_type!r}"
            )
        else:
            assert psd_type is None, (
                f"{name}[{slurm_id}]: psd_type {psd_type!r} is meaningless for {loss_type}"
            )


@pytest.mark.parametrize("mode,family", [
    (mode, family) for mode in ("pretrain_flowmap", "hutchinson", "dlogpdt")
    for family in ("esd_cond", "esd_cond_div_only")
    if mode != "pretrain_flowmap" or family != "esd_cond_div_only"
])
@pytest.mark.parametrize("weight", ["uniform", "linear", "square"])
def test_conditional_esd_cli_keeps_density_ema_and_requested_weight(mode, family, weight, monkeypatch):
    from launchers import learn
    monkeypatch.setattr("sys.argv", ["learn.py", "--cfg_path", "configs.ace_a_nme",
        "--mode", mode, "--loss_type", family, "--div_loss_weighting", weight,
        "--from_scratch", "--offdiag_teacher_source", "ema", "--no_wandb"])
    cfg = learn.setup_config_dict(learn.parse_command_line_arguments())
    assert cfg.training.stopgrad_type == "stopped"
    assert cfg.training.loss_type == family
    assert 'esd_velocity_source' not in cfg.training
    assert cfg.training.diag_teacher_source == "conditional"
    assert cfg.training.offdiag_teacher_source == "ema"
    assert cfg.training.density_ema_fac == .999
    assert cfg.network.predict_divergence == (mode != "pretrain_flowmap")
    if cfg.network.predict_divergence:
        assert cfg.network.div_loss_weighting_type == weight
    assert "_stopped" in cfg.logging.output_name and family in cfg.logging.output_name
    assert learn._select_teacher("conditional", None, {}) is None


@pytest.mark.parametrize('mode', ['pretrain_flowmap', 'pretrain_v'])
def test_div_only_cli_requires_density_head(mode, monkeypatch):
    from launchers import learn
    monkeypatch.setattr("sys.argv", ["learn.py", "--cfg_path", "configs.ace_a_nme", "--mode", mode,
        "--loss_type", "esd_cond_div_only"])
    with pytest.raises(ValueError, match='requires a density head'):
        learn.setup_config_dict(learn.parse_command_line_arguments())


def test_replaced_conditional_flag_is_rejected(monkeypatch):
    from launchers import learn
    monkeypatch.setattr("sys.argv", ["learn.py", "--loss_type", "esd",
        "--esd_velocity_source", "conditional"])
    with pytest.raises(SystemExit):
        learn.parse_command_line_arguments()


@pytest.mark.parametrize("name", KEPT_CONFIGS)
@pytest.mark.parametrize("option,lift", [(None, True), ("--no_com_augmentation", False)])
def test_com_default_and_opt_out_keep_data_model_and_prior_consistent(name, option, lift, monkeypatch):
    import torch
    from common import datasets
    from launchers import learn
    argv = ["learn.py", "--cfg_path", f"configs.{name}", "--from_scratch"]
    if option:
        argv.append(option)
    monkeypatch.setattr("sys.argv", argv)
    cfg = learn.setup_config_dict(learn.parse_command_line_arguments())
    assert cfg.problem.com_augmentation is lift
    assert cfg.network.com_aware is lift
    sample = datasets.setup_base(cfg, cfg.network.input_dims)(
        32, torch.Generator().manual_seed(1), torch.device("cpu"))
    assert bool(sample.mean(1).abs().max() > 0.01) is lift
    # Configs used directly also enable both the loader and the model.
    default = importlib.import_module(f"configs.{name}").get_config(0)
    assert default.problem.com_augmentation and default.network.com_aware


@pytest.mark.parametrize("mode", ["pretrain_v", "pretrain_flowmap", "dlogpdt", "hutchinson"])
@pytest.mark.parametrize("policy", ["stopped", "convex"])
def test_lsd_cli_preserves_requested_correction_gradients(mode, policy, monkeypatch):
    from launchers import learn
    monkeypatch.setattr("sys.argv", ["learn.py", "--cfg_path", "configs.ace_a_nme",
        "--mode", mode, "--loss_type", "lsd", "--stopgrad_type", policy,
        "--from_scratch", "--no_wandb"])
    cfg = learn.setup_config_dict(learn.parse_command_line_arguments())
    assert cfg.training.stopgrad_type == policy
    assert cfg.logging.output_name == cfg.logging.wandb_name
    assert cfg.logging.output_name.count(f"_{policy}") == 1
    other = "convex" if policy == "stopped" else "stopped"
    assert f"_{other}" not in cfg.logging.output_name


@pytest.mark.parametrize("name", KEPT_CONFIGS)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("mode,steps,decay", [
    ("pretrain_v", 400000, 50000),
    ("pretrain_flowmap", 100000, 10000),
    ("dlogpdt", 100000, 50000),
    ("hutchinson", 100000, 50000),
])
def test_new_training_optimizer_and_graph_defaults(name, device, override, mode, steps, decay, monkeypatch):
    from launchers import learn
    argv = ["learn.py", "--cfg_path", f"configs.{name}",
            "--mode", mode, "--device", device]
    if override:
        argv += ["--optimizer", "radam", "--decay_steps", "10000", "--no-cuda_graphs"]
    monkeypatch.setattr("sys.argv", argv)
    cfg = learn.setup_config_dict(learn.parse_command_line_arguments())
    ace_velocity = name == "ace_a_nme" and mode == "pretrain_v"
    assert cfg.optimization.optimizer == ("radam" if override or ace_velocity else "muon_adamw")
    assert cfg.optimization.decay_steps == (10000 if override or ace_velocity else decay)
    assert cfg.optimization.weight_decay == (0. if ace_velocity else 1e-4)
    assert cfg.optimization.total_steps == steps
    assert cfg.training.cuda_graphs is (device.startswith("cuda") and not override)
    assert cfg.network.predict_divergence is (mode in ("dlogpdt", "hutchinson"))


def test_ace_velocity_builds_historical_radam_and_allows_explicit_muon(monkeypatch):
    import math
    import torch
    from common import state_utils
    from launchers import learn

    argv = ["learn.py", "--cfg_path", "configs.ace_a_nme", "--mode", "pretrain_v",
            "--from_scratch", "--device", "cpu"]
    monkeypatch.setattr("sys.argv", argv)
    cfg = learn.setup_config_dict(learn.parse_command_line_arguments())
    optimizer, schedule = state_utils.setup_optimizer(cfg, torch.nn.Linear(2, 1))
    assert isinstance(optimizer, torch.optim.RAdam)
    assert optimizer.param_groups[0]["weight_decay"] == 0
    assert optimizer.param_groups[0]["betas"] == (.9, .999)
    for step in (0, 10000, 40000, 200000, 399999):
        assert schedule(step) == 1e-4 / math.sqrt(max(step / 10000, 1))

    monkeypatch.setattr("sys.argv", argv + ["--optimizer", "muon_adamw",
                        "--weight_decay", ".0007", "--decay_steps", "12345"])
    cfg = learn.setup_config_dict(learn.parse_command_line_arguments())
    assert cfg.optimization.optimizer == "muon_adamw"
    assert cfg.optimization.weight_decay == .0007
    assert cfg.optimization.decay_steps == 12345


@pytest.mark.parametrize("legacy_muon", [False, True])
def test_ace_velocity_resume_preserves_selected_recipe(monkeypatch, legacy_muon):
    import math
    from common import state_utils
    from launchers import learn

    argv = ["learn.py", "--cfg_path", "configs.ace_a_nme", "--mode", "pretrain_v",
            "--device", "cpu", "--network_load_path", "velocity.pt", "--resume_optimizer"]
    if legacy_muon:
        argv += ["--optimizer", "muon_adamw", "--weight_decay", "1e-4",
                 "--decay_steps", "50000"]
    monkeypatch.setattr("sys.argv", argv)
    cfg = learn.setup_config_dict(learn.parse_command_line_arguments())
    assert cfg.network.load_path == "velocity.pt"
    assert cfg.network.reset_optimizer is False
    assert cfg.optimization.optimizer == ("muon_adamw" if legacy_muon else "radam")
    assert cfg.optimization.weight_decay == (1e-4 if legacy_muon else 0.)
    assert cfg.optimization.schedule_type == "sqrt"
    decay_steps = 50000 if legacy_muon else 10000
    assert cfg.optimization.decay_steps == decay_steps
    schedule = state_utils.setup_schedule(cfg)
    for step in (200000, 399999):
        assert schedule(step) == pytest.approx(1e-4 / math.sqrt(step / decay_steps))
