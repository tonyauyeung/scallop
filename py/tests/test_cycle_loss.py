"""LSD cycle losses: live velocity and density round-trip residuals."""
from types import SimpleNamespace

import pytest
import torch

from common import interpolant, losses
from _equiv_utils import build_problem_config
from test_stopped_flowmap import DEVICE, PolynomialMap, batch, heads, snapshots


@pytest.mark.parametrize('density,vectorized', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('stopgrad_type', ['stopped', 'convex'])
@pytest.mark.parametrize('weighted', [False, True])
def test_cycle_adds_weighted_round_trip_values_and_gradients(density, vectorized, stopgrad_type, weighted):
    net = PolynomialMap(vectorized, density, weighted).to(DEVICE)
    teacher, ema = snapshots(net)
    x0, x1, s, t, _ = batch()
    interp = interpolant.setup_interpolant(build_problem_config())
    x = interp.calc_It(s, x0, x1)
    params = dict(net.named_parameters())
    u, d = heads(params, s, t, x, vectorized)
    back_u, back_d = heads(params, t, s, x + (t - s)[:, None, None] * u, vectorized)
    cycle_v = (u - back_u).square().sum((1, 2))
    cycle_d = (d - back_d).square().reshape(4, -1).sum(1) / (6 if vectorized else 1)
    expected = (cycle_v + cycle_d if density else cycle_v) * torch.exp(-torch.as_tensor(net.calc_weight(s, t)))

    kw = dict(interp=interp, stopgrad_type=stopgrad_type, density_ema_params=ema)
    base, _ = losses.lsd_term(net, teacher, x0, x1, None, s, t, **kw)
    value, metrics = losses.lsd_term(net, teacher, x0, x1, None, s, t, cycle_weight=.7, **kw)
    torch.testing.assert_close(value - base, .7 * expected)
    torch.testing.assert_close(metrics['lsd/cycle_v_loss'], cycle_v.detach())
    torch.testing.assert_close(metrics['lsd/cycle_div_loss'],
                               cycle_d.detach() if density else torch.zeros_like(cycle_v))
    wanted = torch.autograd.grad((base + .7 * expected).sum(), (net.p, net.q), allow_unused=True)
    got = torch.autograd.grad(value.sum(), (net.p, net.q), allow_unused=True)
    torch.testing.assert_close(got, wanted)


@pytest.mark.parametrize('loss_type,cycle_weight', [('psd', 1.), ('lsd', -1.), ('lsd', float('nan'))])
def test_setup_rejects_invalid_cycle_weight(loss_type, cycle_weight):
    cfg = SimpleNamespace(training=SimpleNamespace(
        loss_type=loss_type, stopgrad_type='stopped', psd_type='uniform', cycle_weight=cycle_weight,
    ))
    with pytest.raises(ValueError, match='cycle_weight'):
        losses.setup_loss(cfg, PolynomialMap(), interpolant.setup_interpolant(build_problem_config()))
