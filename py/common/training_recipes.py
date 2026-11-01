"""Defaults for new training and explicit density-stage recipes."""

# Missing optimizer fields in historical configs still mean RAdam in state_utils.
MUON_DEFAULTS = dict(
    optimizer="muon_adamw", learning_rate=1e-4, weight_decay=1e-4,
    schedule_type="sqrt", warmup_fraction=0., decay_steps=50000,
    b1=.9, b2=.999, eps=1e-8,
    muon_momentum=.95, muon_ns_steps=5,
    muon_lr_multiplier=1., adamw_lr_multiplier=1.,
)


# New-stage budgets and sqrt decay onset, without warmup.
# Applied before recipes and explicit CLI overrides; frozen run configs retain
# their recorded schedules.
PRETRAIN_DEFAULTS = {
    "pretrain_v": dict(total_steps=400000, decay_steps=50000),
    "pretrain_flowmap": dict(total_steps=100000, learning_rate=5e-4, decay_steps=10000),
}


# Ace's archived velocity teacher used RAdam; its density stages keep Muon.
ACE_A_NME_PRETRAIN_V_DEFAULTS = dict(
    optimizer="radam", learning_rate=1e-4, weight_decay=0.,
    schedule_type="sqrt", warmup_fraction=0., decay_steps=10000,
)


# A3's archived recipe is adopted for AAAAAA and Chignolin velocity pretraining.
AAAAAA_PRETRAIN_V_DEFAULTS = dict(
    total_steps=400000, learning_rate=5e-4, weight_decay=1e-4,
    optimizer="muon_adamw", schedule_type="cosine", warmup_fraction=.05, bs=1024,
)


def apply_branched_muon(cfg, *, mode, device):
    """Select branched density training with the current optimizer/schedule defaults.

    Call after selecting the loss family and before explicit CLI overrides.
    Parent/teacher paths remain caller-supplied; density checkpoints with a
    different branch layout are rejected by the existing checkpoint loader.
    """
    if mode != "dlogpdt" or cfg.training.loss_type not in ("lsd", "psd", "esd"):
        raise ValueError("branched_muon requires --mode dlogpdt and loss_type lsd, psd or esd")
    cfg.training.recipe = "branched_muon"
    cfg.training.cuda_graphs = str(device).split(":")[0] == "cuda"
    cfg.training.offdiag_time_order = "forward"
    cfg.training.diag_teacher_source = "external"
    cfg.training.offdiag_teacher_source = "ema" if cfg.training.loss_type == "psd" else "external"
    cfg.training.ema_facs = [.999]
    cfg.training.density_ema_fac = .999
    cfg.training.diag_tmax = .999
    cfg.network.dit_kwargs.update(num_layers=6, num_shared_layers=4, num_div_layers=2)
    cfg.network.predict_divergence = True
    cfg.network.vectorized_divergence = True
    cfg.network.div_loss_weighting_type = "linear"
    cfg.network.use_weight = False
    cfg.network.velocity_adaptive_power = 0.
    cfg.network.velocity_adaptive_eps = .01
    cfg.network.use_bfloat16 = False
    cfg.network.mixed_precision = "off"
    cfg.network.reset_optimizer = True
    cfg.network.reset_best_metrics = True
    cfg.teacher.ema_fac = .999
    cfg.optimization.update(MUON_DEFAULTS)
    cfg.optimization.update(clip=10., bs=1024, diag_fraction=.5, total_steps=400000)
    cfg.logging.update(log_freq=100, save_freq=10000, visual_freq=10000,
                       plot_bs=1000, n_sampling_steps=[4], importance_weight_clip=.002)
    return cfg
