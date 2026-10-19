"""PyTorch training loop for self-distillation of flow maps (molecule branch).

Mirrors the JAX ``launchers/learn.py``. Supported modes (see README):
``pretrain_v``, ``pretrain_flowmap``, ``hutchinson``, ``cond_trace``, ``dlogpdt``.

Example:
    python py/launchers/learn.py \
        --cfg_path configs.aaa \
        --slurm_id 0 \
        --ts_data_dir /path/to/ts-cache \
        --output_folder /path/to/outputs \
        --mode dlogpdt --vectorized_divergence
"""

import argparse
import importlib
import json
import math
import os
import sys
import time
import types

# Keep optional Flax checkpoint imports on CPU; PyTorch owns the GPU.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PY_DIR = os.path.join(_THIS_DIR, "..")
sys.path.insert(0, os.path.abspath(_PY_DIR))

import torch
from tqdm.auto import tqdm

# Pin matmul precision explicitly (audit N9, 2026-09-03). The upstream
# transferable-samplers entry points set exactly these three
# (transferable_samplers/train.py:23-28, "issues with invertibility"),
# but iF2D2 never imports those modules, so until now the values were torch's
# defaults by accident. Same values, now chosen.
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import common.datasets as datasets
import common.distributed as distributed
import common.interpolant as interpolant
import common.logging as logging
import common.loss_args as loss_args
import common.losses as losses
import common.state_utils as state_utils
import common.updates as updates

try:
    import wandb
except Exception:
    wandb = None


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Direct flow map learning (PyTorch).")
    parser.add_argument("--cfg_path", type=str)
    parser.add_argument("--slurm_id", type=int, default=0)
    parser.add_argument("--dataset_location", "--ts_data_dir", dest="dataset_location", type=str, default="",
                        help="Root of the official transferable-samplers data cache")
    parser.add_argument("--output_folder", type=str, default="")
    parser.add_argument(
        "--mode", type=str, default="pretrain_v",
        choices=["hutchinson", "cond_trace", "dlogpdt", "pretrain_flowmap", "pretrain_v"],
        help="training stage (default: pretrain_v)",
    )
    parser.add_argument(
        "--loss_type", type=str, default=None, choices=losses.LOSS_TYPES,
        help="override the config's self-distillation family (default: the config's "
             "experiments[slurm_id] row). esd_cond uses x1-z in both heads; "
             "esd_cond_div_only uses x1-z only for density and the selected "
             "velocity teacher for velocity. esd_imf uses the student boundary "
             "velocity as tangent and conditional labels (pretrain_flowmap only).",
    )
    parser.add_argument(
        "--psd_type", type=str, default=None, choices=["uniform", "midpoint"],
        help="required with --loss_type psd; picks the intermediate time u.",
    )
    parser.add_argument(
        "--stopgrad_type", type=str, default=None, choices=["stopped", "convex"],
        help="'stopped' detaches correction JVPs (default); 'convex' keeps both "
             "velocity and density time corrections differentiable for LSD. "
             "Other loss families retain their stopped update.",
    )
    parser.add_argument(
        "--offdiag_teacher_source", type=str, default=None, choices=["self", "ema", "external"],
        help="velocity teacher for LSD/ESD or short velocity maps for PSD; "
             "density consistency always uses its own EMA snapshot",
    )
    parser.add_argument(
        "--diag_teacher_source", choices=["external", "ema", "conditional"], default=None,
        help="diagonal velocity anchor; conditional uses interpolant labels and "
             "estimates divergence from the current learned marginal velocity",
    )
    parser.add_argument("--density_ema_fac", type=float, default=None,
                        help="configured EMA decay for density consistency (default: first EMA)")
    parser.add_argument("--cycle_weight", type=float, default=None,
                        help="LSD weight of the velocity and density round-trip (cycle) losses (default 0)")
    parser.add_argument("--offdiag_time_order", choices=["square", "forward"], default=None,
                        help="off-diagonal time pairs: independent square or ordered forward")
    parser.add_argument("--diag_tmax", type=float, default=None,
                        help="upper time bound for diagonal anchors; dlogpdt needs <1 (default .999)")
    parser.add_argument(
        "--div_loss_weighting_type", "--div_loss_weighting", type=str, default=None,
        choices=["uniform", "linear", "square"],
        help="density time weight (1, 1-time, or (1-time)^2); conditional ESD "
             "uses source time s and drops the stopped density target-square term",
    )
    parser.add_argument("--divergence_rescale", type=float, default=0.0)
    parser.add_argument("--vectorized_divergence", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--cuda_graphs", action=argparse.BooleanOptionalAction, default=None,
                        help="capture loss forward/backward on CUDA (including NCCL DDP); "
                             "enabled by default on CUDA; fixed shapes/loss required; optimizer/EMA stay eager")
    parser.add_argument("--use_importance_sample_for_diag_t", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="enable W&B logging (off by default)")
    parser.add_argument("--require_wandb", action="store_true",
                        help="enable W&B and abort at startup if an online run cannot be created")
    parser.add_argument("--log_freq", type=int, default=None,
                        help="scalar logging frequency; saves/evaluation keep their own frequencies")
    parser.add_argument("--total_steps", type=int, default=None, help="override stage budget (velocity: 400k; density/flow map: 100k)")
    parser.add_argument("--training_recipe", choices=["branched_muon"], default=None,
                        help="dlogpdt LSD/PSD/ESD: 4 shared + 2 blocks/head, Muon/AdamW, "
                             "sqrt decay after 50k without warmup, forward times, vectorized/linear density, "
                             "and CUDA graphs on CUDA; explicit flags override the recipe")
    parser.add_argument("--optimizer", choices=["radam", "muon_adamw"], default=None,
                        help="default Muon uses hidden attention/MLP matrices with auxiliary AdamW; radam opts out")
    parser.add_argument("--decay_steps", type=int, default=None,
                        help="decay scale for --schedule_type sqrt (cosine uses total_steps)")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--schedule_type", choices=["cosine", "sqrt", "constant"], default=None)
    parser.add_argument("--warmup_fraction", type=float, default=None,
                        help="fraction of an explicitly selected cosine stage spent in warmup (default 0)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="global training batch size, divided equally across torchrun ranks")
    parser.add_argument("--diag_fraction", type=float, default=None,
                        help="fraction of the global batch used for diagonal anchors (0 < f <= 1); "
                             "overrides config/recipe; pretrain_v requires 1")
    parser.add_argument("--network_load_path", type=str, default=None,
                        help="override config.network.load_path (e.g. a saved .pt)")
    parser.add_argument("--teacher_load_path", type=str, default=None,
                        help="override config.teacher.load_path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    from common.com_lift import add_com_cli_arguments
    add_com_cli_arguments(parser)
    parser.add_argument("--visual_freq", type=int, default=None, help="override eval/plot frequency")
    parser.add_argument("--save_freq", type=int, default=None, help="override checkpoint save frequency")
    parser.add_argument("--n_sampling_steps", type=int, nargs="+", default=None,
                        help="override learned jumps used for in-training validation")
    parser.add_argument("--importance_weight_clip", type=float, default=None,
                        help="fraction of largest log weights dropped in validation SNIS")
    parser.add_argument("--wandb_project", type=str, default=None, help="override wandb project")
    parser.add_argument("--use_weight", action="store_true",
                        help="enable learned EDM2-style logvar loss weighting")
    parser.add_argument("--logvar_channels", type=int, default=None,
                        help="channels for the learned logvar weighting (requires --use_weight)")
    parser.add_argument("--wandb_name_suffix", type=str, default=None,
                        help="append a suffix to wandb_name/output_name to distinguish runs")
    parser.add_argument("--resume_optimizer", action="store_true",
                        help="load optimizer state + step from the checkpoint (true continuation, "
                             "not a warm restart); default resets the optimizer for a new stage")
    parser.add_argument("--use_bfloat16", action="store_true",
                        help="truncate the network coordinate input to bfloat16 (audit N1). "
                             "Off by default since 2026-09-03; pass it to reproduce runs from "
                             "before that date. Adds a _bf16 tag to the run/checkpoint name.")
    parser.add_argument("--mixed_precision", choices=["off", "bf16_trunk"], default=None,
                        help="run the DiT trunk's matmuls (attention + MLP + modulation) in "
                             "bfloat16 while keeping the LayerNorm interior, the conditioning, "
                             "both output heads and everything outside the DiT in fp32. "
                             "Independent of --use_bfloat16, which only rounds the input. "
                             "Adds a _mp-bf16 tag to the run/checkpoint name.")
    return parser.parse_args()


# Stopped corrections remain the default. LSD convex explicitly enables
# parameter gradients through both time corrections (2026-09-10).
_STOPGRAD_DEFAULTS = {family: "stopped" for family in losses.LOSS_TYPES}
_VALID_STOPGRAD = {family: ("stopped", "convex") for family in _STOPGRAD_DEFAULTS}


def apply_loss_family_overrides(cfg, args):
    """Apply --loss_type / --psd_type / --stopgrad_type on top of the config's row.

    Every config's ``experiments`` list starts at ``("lsd", None, "stopped")``, so
    without these flags a psd/esd run could only be launched by editing a config
    file. The run name carries the family, so an override rewrites the method tag
    rather than reusing the lsd name (which would collide with existing checkpoints).
    """
    if args.loss_type is None and args.psd_type is None and args.stopgrad_type is None:
        return cfg

    old_loss = cfg.training.loss_type
    old_psd = getattr(cfg.training, "psd_type", None)
    old_stop = cfg.training.stopgrad_type

    loss_type = args.loss_type or old_loss
    if args.psd_type is not None:
        psd_type = args.psd_type
    elif loss_type == old_loss:
        psd_type = old_psd
    else:
        psd_type = None
    if args.stopgrad_type is not None:
        stopgrad = args.stopgrad_type
    elif loss_type == old_loss:
        stopgrad = old_stop
    else:
        stopgrad = _STOPGRAD_DEFAULTS[loss_type]

    if loss_type == "psd" and psd_type not in ("uniform", "midpoint"):
        raise ValueError("--loss_type psd requires --psd_type {uniform,midpoint}")
    if loss_type != "psd" and psd_type is not None:
        raise ValueError(f"--psd_type is only meaningful for psd (loss_type={loss_type})")
    if stopgrad not in _VALID_STOPGRAD[loss_type]:
        raise ValueError(
            f"stopgrad_type {stopgrad!r} is invalid for loss_type {loss_type!r} "
            f"(valid: {_VALID_STOPGRAD[loss_type]})"
        )

    # ignore_type(): psd_type is declared None in every config, so a plain assignment
    # of a string would trip ml_collections' type check.
    with cfg.training.ignore_type():
        cfg.training.loss_type = loss_type
        cfg.training.psd_type = psd_type
        cfg.training.stopgrad_type = stopgrad

    old_method = f"{old_loss}_{old_psd}" if old_psd else old_loss
    new_method = f"{loss_type}_{psd_type}" if psd_type else loss_type
    if new_method != old_method:
        if old_method in cfg.logging.wandb_name:
            cfg.logging.wandb_name = cfg.logging.wandb_name.replace(old_method, new_method, 1)
        else:
            cfg.logging.wandb_name += f"_{new_method}"
    cfg.logging.output_name = cfg.logging.wandb_name
    print(f"[loss] loss_type={loss_type} psd_type={psd_type} stopgrad_type={stopgrad}")
    return cfg


def setup_config_dict(args):
    cfg_module = importlib.import_module(args.cfg_path)
    cfg = cfg_module.get_config(args.slurm_id, args.dataset_location, args.output_folder)
    from common.training_recipes import PRETRAIN_DEFAULTS, AAAAAA_PRETRAIN_V_DEFAULTS
    cfg.optimization.update(PRETRAIN_DEFAULTS.get(args.mode, {}))
    if args.mode in ("pretrain_v", "pretrain_flowmap"):
        cfg.training.offdiag_teacher_source = "self"
    if args.mode == "pretrain_v":
        cfg.training.diag_teacher_source = "conditional"
    if args.mode == "dlogpdt":
        cfg.network.div_loss_weighting_type = "linear"
    if cfg.problem.target in ("ala6_implicit", "chignolin") and args.mode == "pretrain_v":
        cfg.optimization.update(AAAAAA_PRETRAIN_V_DEFAULTS)
        cfg.training.recipe = "A3"
    cfg.problem.com_augmentation = args.com_augmentation
    cfg.network.com_aware = args.com_augmentation
    # Recipe v1 adopts sweep winner M4 for LSD flowmap pretraining only.
    if (args.mode == "pretrain_flowmap"
            and (args.loss_type or cfg.training.loss_type) == "lsd"
            and args.stopgrad_type is None):
        cfg.training.stopgrad_type = "convex"
    cfg = apply_loss_family_overrides(cfg, args)
    recipe = getattr(args, "training_recipe", None)
    if recipe == "branched_muon":
        from common.training_recipes import apply_branched_muon
        apply_branched_muon(cfg, mode=args.mode, device=args.device)
    sampling_steps = getattr(args, "n_sampling_steps", None)
    if sampling_steps is not None:
        if any(step < 1 for step in sampling_steps):
            raise ValueError("--n_sampling_steps must be positive")
        cfg.logging.n_sampling_steps = sampling_steps
    importance_clip = getattr(args, "importance_weight_clip", None)
    if importance_clip is not None:
        if not math.isfinite(importance_clip) or not 0 <= importance_clip < 1:
            raise ValueError("--importance_weight_clip must be finite and in [0, 1)")
        cfg.logging.importance_weight_clip = importance_clip
    # Explicit flags override both configuration defaults and the optional recipe.
    if getattr(args, "optimizer", None) is not None:
        cfg.optimization.optimizer = args.optimizer
    if getattr(args, "decay_steps", None) is not None:
        if args.decay_steps <= 0:
            raise ValueError("--decay_steps must be positive")
        cfg.optimization.decay_steps = args.decay_steps
    for field in ("learning_rate", "weight_decay", "schedule_type", "warmup_fraction", "total_steps"):
        value = getattr(args, field, None)
        if value is not None:
            setattr(cfg.optimization, field, value)
    total_steps = cfg.optimization.total_steps
    if not isinstance(total_steps, int) or isinstance(total_steps, bool) or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    state_utils.setup_schedule(cfg)  # Validate the resolved stage and warmup before data loading.
    if not math.isfinite(cfg.optimization.weight_decay) or cfg.optimization.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    diag_fraction = getattr(args, "diag_fraction", None)
    if diag_fraction is not None:
        if not 0.0 < diag_fraction <= 1.0:
            raise ValueError("--diag_fraction must be finite and in (0, 1]")
        if args.mode == "pretrain_v" and diag_fraction != 1.0:
            raise ValueError("--mode pretrain_v requires --diag_fraction 1 (all diagonal)")
        cfg.optimization.diag_fraction = diag_fraction
    if getattr(args, "cuda_graphs", None) is not None:
        cfg.training.cuda_graphs = args.cuda_graphs
    elif getattr(cfg.training, "cuda_graphs", None) is None:
        cfg.training.cuda_graphs = str(args.device).split(":")[0] == "cuda"
    cfg.network.divergence_mode = args.mode
    losses._validate_stopgrad(cfg.training.stopgrad_type)
    cfg.logging.wandb_name += f"_{cfg.training.stopgrad_type}"
    for name in ("diag_teacher_source", "offdiag_teacher_source",
                 "density_ema_fac", "diag_tmax", "offdiag_time_order"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg.training, name, value)
    if cfg.training.loss_type == "esd_imf":
        if args.mode != "pretrain_flowmap":
            raise ValueError("esd_imf supports only velocity-only pretrain_flowmap")
        if getattr(args, "diag_teacher_source", None) not in (None, "conditional"):
            raise ValueError("esd_imf requires conditional diagonal supervision")
        if getattr(args, "offdiag_teacher_source", None) not in (None, "self"):
            raise ValueError("esd_imf uses the current student boundary tangent")
        if getattr(args, "teacher_load_path", None):
            raise ValueError("esd_imf does not use an external teacher checkpoint")
        cfg.training.diag_teacher_source = "conditional"
        cfg.training.offdiag_teacher_source = "self"
        cfg.teacher.load_path = ""
    velocity_source = getattr(cfg.training, "offdiag_teacher_source", "self")
    if velocity_source != "self":
        cfg.logging.wandb_name += f"_{velocity_source}teacher"
    cfg.logging.wandb_name += f"_{args.mode}"
    if args.from_scratch:
        cfg.network.load_path = ""
        cfg.teacher.load_path = ""
        cfg.logging.wandb_name += "_from_scratch"
        cfg.network.from_scratch = True
        cfg.training.diag_teacher_source = "conditional"
    else:
        cfg.network.from_scratch = False

    if args.mode in ("pretrain_flowmap", "pretrain_v"):
        if cfg.training.loss_type == "esd_cond_div_only":
            raise ValueError("esd_cond_div_only requires a density head; use a likelihood mode")
        if args.mode == "pretrain_v":
            cfg.optimization.diag_fraction = 1.0
        cfg.network.predict_divergence = False
        cfg.logging.output_name = cfg.logging.wandb_name
        cfg.training.time_distr = "uniform"
        print("Time distribution: uniform")
    else:

        cfg.network.vectorized_divergence = (
            args.vectorized_divergence if args.vectorized_divergence is not None
            else cfg.network.vectorized_divergence
        )
        cfg.network.div_loss_weighting_type = (
            args.div_loss_weighting_type if args.div_loss_weighting_type is not None
            else cfg.network.div_loss_weighting_type
        )
        if cfg.network.div_loss_weighting_type == "linear" and args.use_importance_sample_for_diag_t:
            cfg.training.time_distr = "linear"
            print(r"Time distribution: $1/(1-t)$")
        elif cfg.network.div_loss_weighting_type == "square" and args.use_importance_sample_for_diag_t:
            cfg.training.time_distr = "linear"
            print(r"Time distribution: $1/(1-t)$")
        else:
            cfg.training.time_distr = "uniform"
            print("Time distribution: uniform")

        if args.divergence_rescale <= 0.0:
            if cfg.network.vectorized_divergence:
                cfg.network.divergence_rescale = 1.0
            else:
                cfg.network.divergence_rescale = cfg.problem.d
        else:
            cfg.network.divergence_rescale = args.divergence_rescale

        if cfg.network.vectorized_divergence:
            cfg.logging.wandb_name += "_vectorized_divergence"
        else:
            cfg.logging.wandb_name += "_scalar_divergence"
        cfg.logging.wandb_name += f"_{cfg.network.divergence_rescale}_divergence_rescale"
        if args.mode != "hutchinson" or cfg.network.div_loss_weighting_type != "uniform":
            cfg.logging.wandb_name += f"_{cfg.network.div_loss_weighting_type}_div_loss_weighting"
        if recipe:
            cfg.logging.wandb_name += f"_{recipe}"
        cfg.logging.output_name = cfg.logging.wandb_name
    if args.batch_size is not None:
        # Read by the ts dataloader (setup_state) and by loss_args.get_loss_fn_args,
        # which splits it into diag/off-diag via cfg.optimization.diag_fraction.
        cfg.optimization.bs = args.batch_size
    if args.network_load_path is not None:
        cfg.network.load_path = args.network_load_path
    if args.teacher_load_path is not None:
        cfg.teacher.load_path = args.teacher_load_path
    if args.resume_optimizer:
        cfg.network.reset_optimizer = False  # load optimizer + step (true continuation)
    if args.visual_freq is not None:
        cfg.logging.visual_freq = args.visual_freq
    if args.save_freq is not None:
        cfg.logging.save_freq = args.save_freq
    if args.wandb_project is not None:
        cfg.logging.wandb_project = args.wandb_project
    # The `_ts` tag is now historical, but it is kept unconditionally so run and
    # checkpoint names stay comparable with (and restartable from) every ts-era run.
    cfg.logging.wandb_name += "_ts"
    cfg.logging.output_name = cfg.logging.wandb_name
    if args.use_bfloat16:
        cfg.network.use_bfloat16 = True
        cfg.logging.wandb_name += "_bf16"
        cfg.logging.output_name = cfg.logging.wandb_name
    if args.mixed_precision is not None:
        cfg.network.mixed_precision = args.mixed_precision
    if getattr(cfg.network, "mixed_precision", "off") != "off":
        cfg.logging.wandb_name += f"_mp-{cfg.network.mixed_precision.replace('_trunk', '')}"
        cfg.logging.output_name = cfg.logging.wandb_name
    if args.com_augmentation:
        cfg.logging.wandb_name += "_comaug"
        cfg.logging.output_name = cfg.logging.wandb_name
    if args.use_weight:
        cfg.network.use_weight = True
        if args.logvar_channels is not None:
            cfg.network.logvar_channels = args.logvar_channels
        if getattr(cfg.network, "logvar_channels", None) is None:
            raise ValueError("--use_weight requires --logvar_channels (or config logvar_channels)")
        cfg.logging.wandb_name += f"_logvar{cfg.network.logvar_channels}"
        cfg.logging.output_name = cfg.logging.wandb_name
    if args.cycle_weight is not None:
        cfg.training.cycle_weight = args.cycle_weight
        cfg.logging.wandb_name += f"_cycle{args.cycle_weight}"
        cfg.logging.output_name = cfg.logging.wandb_name
    if args.wandb_name_suffix:
        cfg.logging.wandb_name += args.wandb_name_suffix
        cfg.logging.output_name = cfg.logging.wandb_name
    if args.log_freq is not None:
        if args.log_freq < 1:
            raise ValueError("--log_freq must be positive")
        cfg.logging.log_freq = args.log_freq

    return cfg


def _select_teacher(source, external, self_teacher, ema_teacher=None):
    """Pick the teacher for one loss term.

    Audit N17: this used to fall through to the self-teacher for *any* unrecognised
    source, so a typo in ``diag_teacher_source`` silently converted external
    distillation into self-distillation. The JAX reference raises
    (jax_impl/common/loss_args.py:261-266); so do we. ``train_loop`` has already
    resolved unavailable external sources before entering this function, so
    reaching here with source == "external" and external is None is a real bug.
    """
    if source == "conditional":
        return None
    if source == "external":
        if external is None:
            raise RuntimeError(
                "teacher source is 'external' but no external teacher params were loaded"
            )
        return external
    if source == "ema":
        if ema_teacher is None:
            raise RuntimeError("teacher source is 'ema' but no EMA params were provided")
        return ema_teacher
    if source != "self":
        raise ValueError(
            f"unknown teacher source {source!r} (expected 'external', 'self' or 'ema')"
        )
    return self_teacher


def setup_state(cfg, device, seed):
    interp = interpolant.setup_interpolant(cfg)

    # The transferable-samplers loader is the ONLY training data path (see
    # DECISIONS.md): official splits, their StandardizeTransform (train std) and
    # their SO(3) transform, so training and evaluation share one convention.
    from common import ts_data
    ts_com = getattr(cfg.problem, "com_augmentation", False)
    ds, ts_std = ts_data.load_peptide(
        cfg, getattr(cfg.optimization, "local_bs", cfg.optimization.bs), device,
        com_augmentation=ts_com, seed=seed
    )
    cfg, ex_input = datasets.setup_target(cfg, train_std=ts_std)
    print(f"[ts] transferable-samplers dataloader "
          f"(unscale_factor={ts_std:.6g}, com_aug={ts_com})")

    state, net, schedule = state_utils.setup_training_state(cfg, device)
    teacher_params = state_utils.load_teacher_params(cfg, state, device)
    loss = losses.setup_loss(cfg, net, interp)

    # Every mode uses held-out TS evaluation; pretraining has proposal metrics only.
    if distributed.is_main() and getattr(cfg.logging, "visual_freq", 0):
        from common import ts_eval, ts_systems
        ts_val = ts_systems.load_split(ts_systems.resolve(cfg), "val",
                                       getattr(cfg.problem, "ts_data_dir", None) or None)
        # Audit N8: the evaluator top-slices the reference to cfg.logging.plot_bs, so a
        # stride computed for 10k frames collapsed to val[0:2*plot_bs:2]. Size the
        # stride by what is actually kept, matching ts_systems.build_eval_context:348.
        state.ts_eval_ctx = ts_eval.build_eval_context(
            cfg, ts_val, com_aware=ts_com, num_true=int(cfg.logging.plot_bs),
            num_tica=getattr(cfg.logging, "num_tica", None)
        )
        state.ts_com_aug = ts_com

    statics = types.SimpleNamespace(
        loss=loss,
        ds=ds,
        sample_rho0=datasets.setup_base(cfg, ex_input.shape),
        teacher_params=teacher_params,
    )
    return cfg, statics, state


def train_loop(cfg, statics, state, device, seed):
    logging.register_signal_handlers(cfg, state)
    log_freq = getattr(cfg.logging, "log_freq", 1)
    train_step = updates.setup_train_step(cfg)
    ema_update = updates.setup_ema_update(cfg)

    seed = seed + 100003 * distributed.rank()
    gen_cpu = torch.Generator()
    gen_cpu.manual_seed(seed + 7)
    gen_dev = torch.Generator(device=device)
    gen_dev.manual_seed(seed + 11)

    diag_source = getattr(cfg.training, "diag_teacher_source", None)
    offdiag_source = getattr(cfg.training, "offdiag_teacher_source", None)
    if diag_source is None:
        diag_source = "external" if statics.teacher_params is not None else "conditional"
    if offdiag_source is None:
        offdiag_source = "self" if diag_source == "conditional" else diag_source

    # Audit N17: resolve the configured sources once, loudly. A config that asks for
    # an external teacher AND names a load path, but has no loaded teacher, is a
    # failed load -- that must raise, not silently self-distill. A config that asks
    # for "external" with no load path at all (e.g. pretrain_v --from_scratch) is a
    # legitimate fallback to conditional diagonal training and a self velocity
    # teacher off-diagonal; the resolved choices are printed.
    resolved = {}
    for term, src in (("diag", diag_source), ("offdiag", offdiag_source)):
        allowed = ("external", "self", "ema", "conditional") if term == "diag" else ("external", "self", "ema")
        if src not in allowed:
            raise ValueError(
                f"{term}_teacher_source={src!r} is invalid; expected {allowed}"
            )
        if src == "external" and statics.teacher_params is None:
            load_path = getattr(getattr(cfg, "teacher", None), "load_path", "") or ""
            if load_path:
                raise RuntimeError(
                    f"{term}_teacher_source='external' but the teacher checkpoint at "
                    f"{load_path!r} produced no params"
                )
            src = "conditional" if term == "diag" else "self"
        resolved[term] = src
    diag_source, offdiag_source = resolved["diag"], resolved["offdiag"]
    print(f"[teacher] diag={diag_source} offdiag={offdiag_source} "
          f"density=EMA({getattr(cfg.training, 'density_ema_fac', state.ema_facs[0] if state.ema_facs else None)}) "
          f"loss_type={cfg.training.loss_type} "
          f"(external loaded: {statics.teacher_params is not None})")

    total_steps = cfg.optimization.total_steps
    pbar = tqdm(range(state.step, total_steps), initial=state.step, total=total_steps,
                disable=not distributed.is_main(), mininterval=10.0)
    for step in pbar:
        start = time.time()
        args = loss_args.get_loss_fn_args(cfg, statics, gen_cpu, device)

        self_teacher = state.self_teacher() if "self" in (diag_source, offdiag_source) else None
        ema_teacher = state.ema_teacher() if "ema" in (diag_source, offdiag_source) else None
        teacher_diag = _select_teacher(
            diag_source, statics.teacher_params, self_teacher, ema_teacher
        )
        teacher_offdiag = _select_teacher(
            offdiag_source, statics.teacher_params, self_teacher, ema_teacher
        )

        loss_value, metrics = train_step(
            state, statics.loss, teacher_diag, teacher_offdiag, args, gen_dev
        )
        state = ema_update(state)
        elapsed = time.time() - start

        completed = step + 1
        periodic = any(freq and completed % freq == 0 for freq in (
            getattr(cfg.logging, "save_freq", 0), getattr(cfg.logging, "visual_freq", 0)
        ))
        if log_freq <= 1 or completed % log_freq == 0 or periodic or completed == total_steps:
            loss_value, metrics = distributed.mean_metrics(loss_value, metrics)
            if distributed.is_main():
                metrics["train/optimizer_step"] = state.step
                logging.log_metrics(
                    cfg, state, device, loss_value, metrics, completed, elapsed
                )
                pbar.set_postfix(loss=float(loss_value), refresh=False)
            # Evaluation and checkpoint IO run on rank zero with the unwrapped net.
            distributed.barrier()

    complete = torch.tensor(state.step == total_steps, device=device, dtype=torch.int32)
    if distributed.active():
        torch.distributed.all_reduce(complete, op=torch.distributed.ReduceOp.MIN)
    if not bool(complete):
        raise RuntimeError(f"Incomplete training: optimizer step {state.step}, requested {total_steps}")
    if distributed.is_main():
        logging.save_state(state, cfg)
    distributed.barrier()


def _init_wandb(cfg, args):
    """Best-effort wandb init. Returns the run, or None if wandb is off/unavailable.

    Logging must never be able to kill a multi-day training run. A bare ``wandb.init``
    here raised ``ServiceStartTimeoutError`` and killed a 4-stage chain at step 0 on
    2026-09-03, the same failure that lost a completed 72 h integration on 2026-08-09.
    ``eval_ode_int.py`` grew this guard then; the training launcher never did.
    """
    if wandb is None or args.no_wandb:
        if getattr(args, "require_wandb", False):
            raise RuntimeError("--require_wandb needs wandb installed and enabled")
        return None
    if not (args.wandb or args.require_wandb):
        return None
    try:
        run = wandb.init(
            project=cfg.logging.wandb_project,
            entity=getattr(cfg.logging, "wandb_entity", None),
            name=cfg.logging.wandb_name,
            config=cfg.to_dict(),
            dir=cfg.logging.output_folder or ".",
        )
        if getattr(args, "require_wandb", False) and (run is None or run.settings.mode != "online"):
            raise RuntimeError("--require_wandb requires an online run")
        if getattr(args, "resume_optimizer", False):
            run.config.update(cfg.to_dict(), allow_val_change=True)
        print(f"[wandb] run {run.name} ({run.url})")
        return run
    except Exception as exc:  # noqa: BLE001 - logging is normally best effort
        if getattr(args, "require_wandb", False):
            raise RuntimeError("Required W&B initialization failed") from exc
        print(f"[wandb] WARNING: disabled for this run "
              f"({type(exc).__name__}: {exc}). Training continues; metrics go to "
              f"stdout and checkpoints are unaffected.")
        return None


def main():
    args = parse_command_line_arguments()
    cfg = setup_config_dict(args)
    os.makedirs(cfg.logging.output_folder or ".", exist_ok=True)

    use_graphs = getattr(cfg.training, "cuda_graphs", False)
    if use_graphs and torch.device(args.device).type != "cuda":
        raise ValueError("--cuda_graphs requires --device cuda")
    device = distributed.initialize(args.device, cuda_graphs=use_graphs)
    run = None
    try:
        distributed.configure_batch(cfg)
        seed = cfg.training.seed
        torch.manual_seed(seed)  # identical network initialization on every rank
        print(f"[distributed] rank={distributed.rank()}/{distributed.world_size()} "
              f"device={device} global_bs={cfg.optimization.bs} "
              f"local_bs={cfg.optimization.local_bs}", flush=True)
        cfg, statics, state = setup_state(cfg, device, seed)
        statics.loss = distributed.wrap_loss(state.net, statics.loss, device,
                                              cuda_graphs=use_graphs)
        print(f"[cuda_graphs] enabled={use_graphs}; optimizer and EMA remain eager", flush=True)
        # DDP broadcasts live params; explicitly synchronize the separate EMA state.
        if distributed.active():
            for params in state.ema_params.values():
                for value in params.values():
                    torch.distributed.broadcast(value, src=0)
        torch.manual_seed(seed + 100003 * distributed.rank())
        if distributed.is_main():
            with open(os.path.join(cfg.logging.output_folder or ".", "config.json"), "w") as f:
                json.dump(cfg.to_dict(), f, indent=2)
            run = _init_wandb(cfg, args)
        distributed.barrier()
        train_loop(cfg, statics, state, device, seed)
    finally:
        if run is not None:
            logging._wandb_safe(run.finish, exit_code=1 if sys.exc_info()[0] else 0)
        if distributed.active():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
