"""PyTorch port of common/losses.py.

The JAX code is written per-sample and lifted with ``jax.vmap`` + a mean.
The consistency losses use the selected correction-gradient contract in
``.agents/skills/flowmap``; their gradients intentionally differ from the
vendored historical JAX losses. The loss terms operate on the full batch
directly (the DiT has no cross-sample coupling), returning a per-sample loss
vector and per-sample metric vectors. The diagonal/off-diagonal split and the
final mean are handled in :func:`setup_loss`.

Parameter selection mirrors ``X.apply(params, ...)`` vs
``X.apply(teacher_params, ...)``:

* the *student* is the live module (``net``), called directly so gradients flow;
* a *teacher* is supplied as a detached ``{name: tensor}`` dict and applied with
  ``torch.func.functional_call``.
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch.func import functional_call

from .se3_utils import remove_mean
from .flow_map import _bcast
from .loss_args import _get_diag_offdiag_bs


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sum_non_batch(x: torch.Tensor) -> torch.Tensor:
    """Sum over every axis except the leading batch axis."""
    if x.ndim <= 1:
        return x
    return x.sum(dim=tuple(range(1, x.ndim)))


def _apply(net, params: Optional[Dict[str, torch.Tensor]], method: str, *args, **kwargs):
    """Run ``net.<method>`` with the student (params=None) or a teacher dict."""
    if params is None:
        return net(*args, method=method, **kwargs)
    return functional_call(net, params, args=tuple(args), kwargs={**kwargs, "method": method})


def _stop(x):
    return x.detach() if torch.is_tensor(x) else x


def _split_div(value):
    """Split a ``(primal, div)`` return into ``(primal, div_or_None)``.

    ``flow_call``/``calc_b``/``partial_s`` return a bare tensor rather than a tuple
    when the network has no divergence head, so every caller that asks for
    ``return_div`` has to cope with both shapes (``lsd_term`` did; ``psd_term`` and
    ``esd_term`` used to unpack blindly and raised at step 0 under
    ``predict_divergence=False``).
    """
    if isinstance(value, tuple):
        return value[0], value[1]
    return value, None


def _split_X_phi_div(value):
    """Same for ``flow_call(return_X_and_phi=True, return_div=True)``."""
    if len(value) == 3:
        return value
    return value[0], value[1], None


def _require_div_head(net, term: str) -> None:
    """Raise if a divergence model silently produced no divergence output.

    The JAX reference only prints a warning (``jax.debug.print``). Training a
    likelihood model whose density head sees the diagonal term alone is the exact
    failure this guard exists to make impossible, so it raises instead.
    """
    if getattr(net.config, "predict_divergence", False):
        raise RuntimeError(
            f"{term}_term: network.predict_divergence is True but no divergence output "
            "was returned; the density head would be trained on the diagonal only"
        )


# --------------------------------------------------------------------------- #
# divergence estimators
# --------------------------------------------------------------------------- #
def _project_state(net, value):
    """Project onto the modeled mean-free subspace, or retain all CoM degrees."""
    return value if getattr(net.config, "com_aware", False) else remove_mean(value)


def _hutchinson_divergence_with_phi(
    net, params, t, x, *, eps=None, generator=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stopped trace target for the selected diagonal velocity.

    Gaussian probes have covariance I in full space, P in mean-free space.
    Vectorized outputs estimate the diagonal entries; summing gives the trace.
    No reverse-over-Jacobian optimization path is retained.
    """
    if eps is None:
        eps = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    if eps.shape != x.shape:
        raise ValueError("Hutchinson eps must have the same shape as x")
    eps = _project_state(net, eps.detach().to(x))
    x_in = x.detach().requires_grad_(True)
    with torch.enable_grad():
        phi = _apply(net, params, "calc_b", t, x_in, train=False)
        if phi.requires_grad:
            (vjp_eps,) = torch.autograd.grad(
                phi, x_in, grad_outputs=eps, create_graph=False, allow_unused=True,
            )
        else:
            vjp_eps = None
    if vjp_eps is None:  # a spatially constant field has exactly zero divergence
        vjp_eps = torch.zeros_like(x)
    div_hat = vjp_eps * eps
    if not net.config.vectorized_divergence:
        div_hat = _sum_non_batch(div_hat)
    return phi.detach(), div_hat.detach()


def _calc_divergence_with_phi(
    net, params, t, xt, x0, x1, *, eps=None, generator=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Stopped diagonal anchor, in physical positive-divergence units.

    dlogpdt assumes independent isotropic Gaussian base noise and a linear
    interpolant at THIS sampled state. For an approximate drift it is a
    continuity-based target, not an unbiased trace of that approximate drift.
    Off-diagonal consistency must use the EMA raw density head instead.
    """
    mode = net.config.divergence_mode
    if mode == "hutchinson":
        return _hutchinson_divergence_with_phi(
            net, params, t, xt, eps=eps, generator=generator,
        )
    with torch.no_grad():
        phi = _apply(net, params, "calc_b", t, xt.detach(), train=False)
        if mode in ("pretrain_flowmap", "pretrain_v"):
            return phi.detach(), None
        if mode not in ("dlogpdt", "cond_trace"):
            raise ValueError(f"Invalid divergence_mode: {mode}")
        from .cuda_graphs import check_tensor
        check_tensor(((t >= 0) & (t < 1) & torch.isfinite(t)).all(),
                     f"{mode} requires diagonal times 0 <= t < 1; set diag_tmax < 1")
        tb = _bcast(t, xt)
        check_tensor(torch.isclose(xt, (1 - tb) * x0 + tb * x1,
                                    rtol=1e-5, atol=1e-6).all(),
                     f"{mode} requires the original linear-interpolant state and labels")
        sigma = float(net.config.rescale)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Gaussian base rescale must be finite and positive")
        cond_vel = x1 - x0
        # diag(P) = 1 - 1/N for P = (I_N - 11^T/N) tensor I_dimension.
        # Its sum is 3N-3, not 3N. In full space diag(P)=1.
        projector_diag = 1.0 if getattr(net.config, "com_aware", False) else 1 - 1 / xt.shape[-2]
        if mode == "dlogpdt":
            noise = _project_state(net, x0)
            div_hat = ((phi - cond_vel) * noise / sigma**2 - projector_diag) / (1 - tb)
        else:
            # Retained experimental cond_trace estimator; not identified with
            # the dlogpdt identity for general data distributions.
            div_hat = (-(phi - cond_vel) * cond_vel * tb / sigma**2 - projector_diag) / (1 - tb)
        if not net.config.vectorized_divergence:
            div_hat = _sum_non_batch(div_hat)
        return phi.detach(), div_hat.detach()


def _get_div_loss_scaling(net) -> float:
    if net.config.vectorized_divergence:
        return float(math.prod(net.config.input_dims))
    return 1.0


def _calc_div_loss(net, div_pred, div_target, t, *, factorized=None) -> torch.Tensor:
    """Time-weight the density objective after optionally dropping target^2.

    The diagonal linear objective and conditional ESD density objectives use
    pred^2 - 2*pred*sg(target). This preserves prediction semigradients for fixed
    weights, but changes loss values AND learned log-variance gradients. Never
    silently add the constant back when a learned weight is enabled.
    """
    wtype = net.config.div_loss_weighting_type
    if wtype == "uniform":
        factor = torch.ones_like(t)
    elif wtype == "square":
        factor = (1.0 - t) ** 2
    elif wtype == "linear":
        factor = 1.0 - t
    else:
        raise ValueError(f"Invalid div_loss_weighting_type: {wtype}")
    if factorized is None:
        factorized = wtype == "linear"
    target = div_target.detach()
    error = div_pred.square() - 2.0 * div_pred * target if factorized else (div_pred - target).square()
    return _sum_non_batch(error) * factor / _get_div_loss_scaling(net)


# --------------------------------------------------------------------------- #
# loss terms (batched)
# --------------------------------------------------------------------------- #
def _velocity_objective(net, squared):
    """Optional MeanFlow metric on per-example coordinate-SUM residuals.

    The adaptive factor is stopped. Power zero is exactly the historical
    objective; this screen deliberately restricts adaptation to velocity-only
    pretraining without learned log-variance weights.
    """
    power = float(getattr(net.config, "velocity_adaptive_power", 0.0))
    eps = float(getattr(net.config, "velocity_adaptive_eps", 0.01))
    if not math.isfinite(power) or power < 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("velocity adaptive power must be finite >=0 and eps finite >0")
    if power == 0:
        return squared, None
    if (getattr(net.config, "predict_divergence", False)
            or getattr(net.config, "use_weight", False)
            or getattr(net, "use_weight", False)
            or net.config.divergence_mode not in ("pretrain_v", "pretrain_flowmap")):
        raise ValueError("adaptive velocity weighting requires velocity-only pretraining without use_weight")
    weight = (squared.detach() + eps).pow(-power)
    return weight * squared, weight


def diagonal_term(
    net,
    teacher_params,
    x0,
    x1,
    label,
    t,
    *,
    interp,
    eps=None,
    generator=None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Diagonal (interpolant) term; returns (loss_vec, metrics)."""
    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)

    bt_rslt = net.calc_b(t, It, train=True, return_div=True)

    if net.config.divergence_mode == "pretrain_v":
        div_hat = None
        bt_teacher = It_dot
    else:
        if net.config.divergence_mode in ("dlogpdt", "cond_trace") and getattr(interp, "kind", None) != "linear":
            raise ValueError("dlogpdt/cond_trace require the linear interpolant")
        if net.config.from_scratch or teacher_params is None:
            _, div_hat = _calc_divergence_with_phi(
                net, None, t, It, x0, x1, eps=eps, generator=generator
            )
            bt_teacher = It_dot
        else:
            bt_teacher, div_hat = _calc_divergence_with_phi(
                net, teacher_params, t, It, x0, x1, eps=eps, generator=generator
            )
            bt_teacher = _stop(bt_teacher)
        divergence_rescale = net.config.divergence_rescale
        div_hat = _stop(div_hat) / divergence_rescale if div_hat is not None else None

    if isinstance(bt_rslt, tuple):
        bt, div_tt = bt_rslt
    else:
        bt, div_tt = bt_rslt, None

    base_velocity_loss = _sum_non_batch((bt - bt_teacher) ** 2)
    weighted_velocity, adaptive_weight = _velocity_objective(net, base_velocity_loss)
    div_loss = torch.zeros_like(base_velocity_loss)
    if div_tt is not None:
        if div_hat is None:
            raise ValueError("A density head requires a diagonal divergence estimator")
        div_loss = _calc_div_loss(net, div_tt, div_hat, t)
        velocity_loss = weighted_velocity + div_loss
    else:
        _require_div_head(net, "diag")
        velocity_loss = weighted_velocity

    weight_tt = net.calc_weight(t, t)  # zero log variance when use_weight is False
    if torch.is_tensor(weight_tt):
        weight_tt = weight_tt.reshape(-1)
        loss_value = torch.exp(-weight_tt) * velocity_loss + weight_tt
    else:
        loss_value = math.exp(-weight_tt) * velocity_loss + weight_tt

    metrics = {
        "diag/velocity_loss": _stop(base_velocity_loss),
        "diag/div_loss": _stop(div_loss),
    }
    if adaptive_weight is not None:
        metrics["diag/adaptive_weight"] = adaptive_weight
        metrics["diag/weighted_velocity_loss"] = weighted_velocity.detach()
    if getattr(net, "use_weight", False):
        metrics["diag/learned_logvar"] = weight_tt.detach()
        metrics["diag/learned_multiplier"] = torch.exp(-weight_tt.detach())
        metrics["diag/learned_velocity_loss"] = torch.exp(-weight_tt.detach()) * weighted_velocity.detach()
    return loss_value, metrics


def _validate_stopgrad(stopgrad_type):
    # LSD convex differentiates both raw-head time corrections. Other families
    # retain their stopped update for either accepted spelling.
    if stopgrad_type not in ("stopped", "convex"):
        raise ValueError(
            f"stopgrad_type={stopgrad_type!r} violates the stopped-target contract; "
            "use 'stopped' or 'convex' (differentiable LSD corrections). "
            "Historical 'none'/'full' gradients are no longer supported."
        )


def _density_source(net, density_ema_params, s, t, x):
    if density_ema_params is None:
        raise ValueError("Density consistency requires explicit density_ema_params from TrainState.ema_teacher()")
    with torch.no_grad():
        _, d = _split_div(_apply(
            net, density_ema_params, "calc_phi", s, t, x.detach(),
            train=False, return_div=True, return_velocity=False,
        ))
    if d is None:
        _require_div_head(net, "density")
        raise RuntimeError("EMA density source returned no raw density head")
    return d.detach()


def _raw_jvp(net, s, t, x, velocity=None):
    """Current raw u,d and a destination partial or joint source-material JVP.

    calc_phi accepts (s,t,x) and handles the internal (s,t-s) coordinates. ESD
    holds t fixed: its internal gap tangent is therefore -1 automatically.
    The primal remains live; each loss applies its selected tangent policy.
    """
    if velocity is None:  # LSD
        tangents = (torch.zeros_like(s), torch.ones_like(t), torch.zeros_like(x))
    else:  # ESD: partial_s + velocity dot grad_x
        tangents = (torch.ones_like(s), torch.zeros_like(t), velocity.detach())
    return torch.func.jvp(
        lambda ss, tt, xx: net.calc_phi(ss, tt, xx, train=False, return_div=True),
        (s, t, x), tangents,
    )


def _consistency_loss(net, family, s, t, u, target_u, d=None, target_d=None,
                      *, conditional_density=False, cycle=None):
    v_loss = _sum_non_batch((u - target_u.detach()) ** 2)
    if d is None:
        _require_div_head(net, family)
        d_loss = torch.zeros_like(v_loss)
    elif conditional_density:
        # ESD is anchored at x_s, so its density time weight is evaluated at s.
        d_loss = _calc_div_loss(net, d, target_d, s, factorized=True)
    else:
        d_loss = _sum_non_batch((d - target_d.detach()) ** 2) / _get_div_loss_scaling(net)
    weighted_velocity, adaptive_weight = _velocity_objective(net, v_loss)
    base = weighted_velocity + d_loss
    if cycle is not None:
        base = base + cycle
    weight = net.calc_weight(s, t)
    if torch.is_tensor(weight):
        weight = weight.reshape(-1)
        value = torch.exp(-weight) * base + weight
    else:
        value = math.exp(-weight) * base + weight
    metrics = {f"{family}/{family}_v_loss": v_loss.detach(),
               f"{family}/{family}_div_loss": d_loss.detach()}
    if adaptive_weight is not None:
        metrics[f"{family}/adaptive_weight"] = adaptive_weight
        metrics[f"{family}/weighted_velocity_loss"] = weighted_velocity.detach()
    if getattr(net, "use_weight", False):
        metrics[f"{family}/learned_logvar"] = weight.detach()
        metrics[f"{family}/learned_multiplier"] = torch.exp(-weight.detach())
        metrics[f"{family}/learned_velocity_loss"] = torch.exp(-weight.detach()) * weighted_velocity.detach()
    return value, metrics


def _cycle_loss(net, s, t, x, u, d, cycle_weight):
    """Live round-trip residuals u(x,s,t) - u(X,t,s) and d(x,s,t) - d(X,t,s)."""
    back_u, back_d = _split_div(net.calc_phi(
        t, s, x + _bcast(t - s, u) * u, train=False, return_div=True,
    ))
    v_loss = _sum_non_batch((u - back_u) ** 2)
    d_loss = (torch.zeros_like(v_loss) if d is None
              else _sum_non_batch((d - back_d) ** 2) / _get_div_loss_scaling(net))
    metrics = {"lsd/cycle_v_loss": v_loss.detach(), "lsd/cycle_div_loss": d_loss.detach()}
    return cycle_weight * (v_loss + d_loss), metrics


def lsd_term(
    net, teacher_params, x0, x1, label, s, t, *, interp,
    stopgrad_type="stopped", density_ema_params=None, eps=None, generator=None,
    cycle_weight=0.0,
):
    """u + delta*partial_t u = sg(v_t(X_theta)); likewise for raw d.

    ``convex`` differentiates both correction JVPs; ``stopped`` detaches them.
    Velocity uses the explicitly selected teacher. Density uses its own EMA
    diagonal source at the detached CURRENT student endpoint. Divergence
    estimators and source conditional labels are never used at that endpoint.
    ``cycle_weight`` adds the round-trip losses of :func:`_cycle_loss` inside
    the learned weight.
    """
    _validate_stopgrad(stopgrad_type)
    x = interp.calc_It(s, x0, x1)
    primal, tangent = _raw_jvp(net, s, t, x)
    u, d = _split_div(primal)
    dt_u, dt_d = _split_div(tangent)
    cycle, cycle_metrics = _cycle_loss(net, s, t, x, u, d, cycle_weight) if cycle_weight else (None, {})
    with torch.no_grad():
        endpoint = x + _bcast(t - s, u) * u.detach()
        b = _apply(net, teacher_params, "calc_b", t, endpoint.detach(), train=False)
    # Put live corrections on the prediction side: _consistency_loss always
    # detaches its targets. Teacher calls and the endpoint remain frozen.
    if stopgrad_type == "convex":
        u = u + _bcast(t - s, u) * dt_u
        target_u = b
    else:
        target_u = b - _bcast(t - s, u) * dt_u.detach()
    target_d = None
    if d is not None:
        g = _density_source(net, density_ema_params, t, t, endpoint)
        if stopgrad_type == "convex":
            d = d + _bcast(t - s, d) * dt_d
            target_d = g
        else:
            target_d = g - _bcast(t - s, d) * dt_d.detach()
    value, metrics = _consistency_loss(net, "lsd", s, t, u, target_u, d, target_d, cycle=cycle)
    return value, {**metrics, **cycle_metrics}


def psd_term(
    net, teacher_params, x0, x1, label, s, t, u, h, *, interp,
    psd_type, stopgrad_type="stopped", density_ema_params=None,
):
    """Regress long-step raw heads onto stopped two-step composition targets.

    With u=h*s+(1-h)*t the raw-head weights are (1-h,h). Density uses
    two EMA calls and their EMA intermediate map, independently of velocity.
    """
    _validate_stopgrad(stopgrad_type)
    if psd_type not in ("uniform", "midpoint"):
        raise ValueError(f"Invalid psd_type: {psd_type}")
    if u is None or (psd_type == "uniform" and h is None):
        raise ValueError("PSD requires sampled intermediate time u and interval weights")
    weight = 0.5 if psd_type == "midpoint" else h
    # sample_times constructs u from the interval weights. Avoid a synchronizing
    # value check here so this loss can run inside a CUDA graph.
    x = interp.calc_It(s, x0, x1)
    phi, d = _split_div(net.calc_phi(s, t, x, train=False, return_div=True))

    def short_steps(params, want_div):
        first = _apply(net, params, "flow_call", s, u, x, train=False,
                       return_X_and_phi=True, return_div=want_div)
        mid, phi1, d1 = _split_X_phi_div(first)
        if want_div:
            # Only the first EMA velocity is needed to construct its midpoint.
            # The second short map contributes raw density, so skip its private
            # velocity branch (and the discarded second endpoint).
            d2 = _density_source(net, params, u, t, mid)
            return phi1, None, d1, d2
        phi2 = _apply(net, params, "calc_phi", u, t, mid.detach(), train=False)
        return phi1, phi2, None, None

    with torch.no_grad():
        phi1, phi2, _, _ = short_steps(teacher_params, False)
        w = _bcast(weight, phi)
        target_u = (1 - w) * phi1 + w * phi2
        target_d = None
        if d is not None:
            if density_ema_params is None:
                raise ValueError("Density consistency requires explicit density_ema_params from TrainState.ema_teacher()")
            _, _, d1, d2 = short_steps(density_ema_params, True)
            if d1 is None or d2 is None:
                _require_div_head(net, "psd")
            wd = _bcast(weight, d)
            target_d = (1 - wd) * d1 + wd * d2
    return _consistency_loss(net, "psd", s, t, phi, target_u, d, target_d)


ESD_LOSS_TYPES = ("esd", "esd_cond", "esd_cond_div_only")
DENSITY_LOSS_TYPES = ("lsd", "psd", *ESD_LOSS_TYPES)
LOSS_TYPES = (*DENSITY_LOSS_TYPES, "esd_imf")


def esd_term(
    net, teacher_params, x0, x1, label, s, t, *, interp,
    stopgrad_type="stopped", density_ema_params=None, loss_type="esd",
):
    """u = sg(a_u + delta*(partial_s u + J_x u a_u)).

    esd: teacher advection in both heads. esd_cond: x1-z in both heads.
    esd_cond_div_only: teacher for u, x1-z for d. All density sources remain
    independent EMA values: d_target = sg(d_ema(x,s,s) + delta*M_{a_d}d).
    Conditional density objectives drop target^2, then apply the configured
    weight at source time s. The ordinary u,d predictions remain live;
    every spatial/source/destination-time derivative correction is stopped,
    including the internal gap-coordinate derivative (gap=t-s). The public
    material JVP holds destination t fixed and already accounts for that chain
    rule. This contract also holds when the accepted flag is named "convex".
    """
    _validate_stopgrad(stopgrad_type)
    if loss_type not in ESD_LOSS_TYPES:
        raise ValueError(f"Invalid ESD loss_type: {loss_type}")
    conditional_density = loss_type != "esd"
    if conditional_density and getattr(interp, "kind", None) != "linear":
        raise ValueError("Conditional MeanFlow ESD requires the linear interpolant")
    if loss_type == "esd_cond_div_only" and not net.config.predict_divergence:
        raise ValueError("esd_cond_div_only requires a density head")
    x = interp.calc_It(s, x0, x1)
    with torch.no_grad():
        eta = interp.calc_It_dot(s, x0, x1) if conditional_density else None
        b = eta if loss_type == "esd_cond" else _apply(
            net, teacher_params, "calc_b", s, x.detach(), train=False,
        )
    primal, tangent = _raw_jvp(net, s, t, x, b.detach())
    phi, d = _split_div(primal)
    material_u, material_d = _split_div(tangent)
    target_u = (b.detach() + _bcast(t - s, phi) * material_u.detach()).detach()
    target_d = None
    if d is not None:
        if loss_type == "esd_cond_div_only":
            # A distinct density direction is essential: the velocity JVP uses
            # the teacher. no_grad preserves forward AD while avoiding a second
            # reverse graph; only the first JVP primal is a student prediction.
            with torch.no_grad():
                _, density_tangent = _raw_jvp(net, s, t, x, eta.detach())
                _, material_d = _split_div(density_tangent)
        g = _density_source(net, density_ema_params, s, s, x)
        target_d = (g + _bcast(t - s, d) * material_d.detach()).detach()
    return _consistency_loss(net, loss_type, s, t, phi, target_u, d, target_d,
                             conditional_density=conditional_density)


def esd_imf_term(
    net, teacher_params, x0, x1, label, s, t, *, interp,
    stopgrad_type="stopped", density_ema_params=None,
):
    """Boundary-velocity iMF in forward time, restricted to velocity flow maps.

    a = sg[u(x_s,s,s)], eta = x1-x0,
    V = u(x_s,s,t) - (t-s)*sg[partial_s u + J_x u*a].
    Regress V against eta. The equivalent stopped target below keeps the
    ordinary u prediction live. External teacher and density snapshots are
    intentionally unused; the boundary velocity belongs to the current student.
    """
    _validate_stopgrad(stopgrad_type)
    if net.config.predict_divergence or net.config.divergence_mode != "pretrain_flowmap":
        raise ValueError("esd_imf supports only velocity-only pretrain_flowmap")
    if getattr(interp, "kind", None) != "linear":
        raise ValueError("esd_imf requires the linear interpolant")
    x = interp.calc_It(s, x0, x1)
    with torch.no_grad():
        eta = interp.calc_It_dot(s, x0, x1).detach()
        direction = net.calc_b(s, x.detach(), train=False).detach()
    primal, tangent = _raw_jvp(net, s, t, x, direction)
    phi, _ = _split_div(primal)
    material, _ = _split_div(tangent)
    correction = _bcast(t-s, phi) * material.detach()
    target = (eta + correction).detach()
    value, metrics = _consistency_loss(net, "esd_imf", s, t, phi, target)
    metrics["esd_imf/tangent_label_mse"] = _sum_non_batch((direction-eta).square()).detach()
    metrics["esd_imf/correction_mse"] = _sum_non_batch(correction.square()).detach()
    return value, metrics


# --------------------------------------------------------------------------- #
# loss assembly (diagonal / off-diagonal split)
# --------------------------------------------------------------------------- #


def setup_loss(cfg, net, interp):
    """Assemble diagonal anchors and the selected consistency update."""
    _velocity_objective(net, torch.zeros(1))  # validate opt-in metric settings
    loss_type = cfg.training.loss_type
    stopgrad_type = cfg.training.stopgrad_type
    _validate_stopgrad(stopgrad_type)
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    if getattr(cfg.training, "esd_velocity_source", "teacher") != "teacher":
        raise ValueError("esd_velocity_source was replaced by loss_type='esd_cond'")
    if loss_type in ("esd_cond", "esd_cond_div_only") and getattr(interp, "kind", None) != "linear":
        raise ValueError("Conditional MeanFlow ESD requires the linear interpolant")
    if loss_type == "esd_cond_div_only" and not net.config.predict_divergence:
        raise ValueError("esd_cond_div_only requires a density head")
    if loss_type == "esd_imf":
        if net.config.predict_divergence or net.config.divergence_mode != "pretrain_flowmap":
            raise ValueError("esd_imf supports only velocity-only pretrain_flowmap")
        if getattr(interp, "kind", None) != "linear":
            raise ValueError("esd_imf requires the linear interpolant")
        if getattr(cfg.training, "diag_teacher_source", "conditional") != "conditional":
            raise ValueError("esd_imf requires conditional diagonal supervision")
    psd_type = getattr(cfg.training, "psd_type", None)
    cycle_weight = getattr(cfg.training, "cycle_weight", 0.0)
    if not math.isfinite(cycle_weight) or cycle_weight < 0:
        raise ValueError("cycle_weight must be finite and nonnegative")
    if cycle_weight and loss_type != "lsd":
        raise ValueError("cycle_weight requires loss_type='lsd'")
    if getattr(net.config, "predict_divergence", False):
        if not math.isfinite(net.config.divergence_rescale) or net.config.divergence_rescale <= 0:
            raise ValueError("Density training requires a finite positive divergence_rescale")
        if net.config.divergence_mode in ("dlogpdt", "cond_trace"):
            if getattr(interp, "kind", None) != "linear":
                raise ValueError("dlogpdt/cond_trace require the linear interpolant")
            base = getattr(getattr(cfg, "problem", None), "base", "mean-free-gaussian")
            if base not in ("gaussian", "mean-free-gaussian"):
                raise ValueError("dlogpdt/cond_trace require an independent Gaussian base")
            if base == "gaussian" and not getattr(net.config, "com_aware", False):
                raise ValueError("A full Gaussian base requires com_aware=True for consistent density coordinates")
    print(f"Setting up loss: {loss_type}")
    print(f"Stopgrad type: {stopgrad_type}")

    def offdiag_term(teacher, density_ema, x0, x1, label, s, t, u, h):
        if loss_type == "esd_imf":
            return esd_imf_term(net, teacher, x0, x1, label, s, t,
                                interp=interp, stopgrad_type=stopgrad_type)
        if loss_type == "psd":
            return psd_term(
                net, teacher, x0, x1, label, s, t, u, h,
                interp=interp, psd_type=psd_type, stopgrad_type=stopgrad_type,
                density_ema_params=density_ema,
            )
        elif loss_type == "lsd":
            return lsd_term(
                net, teacher, x0, x1, label, s, t,
                interp=interp, stopgrad_type=stopgrad_type,
                density_ema_params=density_ema, cycle_weight=cycle_weight,
            )
        elif loss_type in ESD_LOSS_TYPES:
            return esd_term(
                net, teacher, x0, x1, label, s, t,
                interp=interp, stopgrad_type=stopgrad_type,
                density_ema_params=density_ema, loss_type=loss_type,
            )
        raise ValueError(f"Unknown loss_type: {loss_type}")

    def loss(
        teacher_params_diag,
        teacher_params_offdiag,
        x0,
        x1,
        label,
        s,
        t,
        u,
        h,
        generator=None,
        diag_eps=None,
        density_ema_params=None,
    ):
        total_bs = x0.shape[0]
        diag_bs, offdiag_bs = _get_diag_offdiag_bs(cfg, total_bs)
        if net.config.predict_divergence and diag_bs <= 0:
            raise ValueError("Density training requires a diagonal divergence anchor")

        total_loss = x0.new_zeros(())
        total_metrics: Dict[str, torch.Tensor] = {}

        def _accumulate(metrics, new):
            for k, v in new.items():
                total = _sum_non_batch(v).sum() if v.ndim >= 1 else v
                metrics[k] = metrics.get(k, 0.0) + total

        if diag_bs > 0:
            label_diag = None if label is None else label[:diag_bs]
            diag_loss_vec, diag_metrics = diagonal_term(
                net, None if loss_type == "esd_imf" else teacher_params_diag, x0[:diag_bs], x1[:diag_bs], label_diag,
                t[:diag_bs], interp=interp, eps=diag_eps, generator=generator,
            )
            total_loss = total_loss + diag_loss_vec.sum()
            _accumulate(total_metrics, diag_metrics)

        if offdiag_bs > 0:
            label_off = None if label is None else label[diag_bs:]
            u_off = None if u is None else u[diag_bs:]
            h_off = None if h is None else h[diag_bs:]
            off_loss_vec, off_metrics = offdiag_term(
                teacher_params_offdiag, density_ema_params, x0[diag_bs:], x1[diag_bs:], label_off,
                s[diag_bs:], t[diag_bs:], u_off, h_off,
            )
            total_loss = total_loss + off_loss_vec.sum()
            _accumulate(total_metrics, off_metrics)

        total_loss = total_loss / total_bs
        # Every loss type reports its per-term diagnostics. The old code dropped the
        # metrics dict unless loss_type == "lsd", which left psd/esd runs with no way
        # to tell a diverging velocity term from a diverging divergence term.
        total_metrics = {k: (_stop(v) / total_bs) for k, v in total_metrics.items()}
        return total_loss, total_metrics

    return loss
