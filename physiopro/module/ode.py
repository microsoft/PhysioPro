# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from inspect import signature
import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_bias.weight.data.fill_(0.0)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(-1, 1))


class ConcatLinearNorm(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_bias.weight.data.fill_(0.0)
        self.norm = nn.LayerNorm(dim_out, eps=1e-6)

    def forward(self, t, x):
        return self.norm(self._layer(x) + self._hyper_bias(t.view(-1, 1)))


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(-1, 1))) \
               + self._hyper_bias(t.view(-1, 1))


class DiffEqWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, t, y):
        if len(signature(self.module.forward).parameters) == 1:
            return self.module(y)
        if len(signature(self.module.forward).parameters) == 2:
            return self.module(t, y)
        raise ValueError("Differential equation needs to either take (t, y) or (y,) as input.")

    def __repr__(self):
        return self.module.__repr__()


def diffeq_wrapper(layer):
    return DiffEqWrapper(layer)


class SequentialDiffEq(nn.Module):
    """A container for a sequential chain of layers. Supports both regular and diffeq layers.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList([diffeq_wrapper(layer) for layer in layers])

    def forward(self, t, x):
        for layer in self.layers:
            x = layer(t, x)
        return x


class TimeDependentSwish(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.beta = nn.Sequential(
            nn.Linear(1, min(64, dim * 4)),
            nn.Softplus(),
            nn.Linear(min(64, dim * 4), dim),
            nn.Softplus(),
        )

    def forward(self, t, x):
        beta = self.beta(t.reshape(-1, 1))
        return x * torch.sigmoid_(x * beta)


class WrapRegularization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, reg, *x):
        ctx.save_for_backward(reg)
        return x

    @staticmethod
    def backward(ctx, *grad_x):
        reg, = ctx.saved_variables
        return (torch.ones_like(reg), *grad_x)


class TimeVariableODE(nn.Module):
    start_time = -1.0
    end_time = 1.0

    def __init__(self, func, atol=1e-6, rtol=1e-6, method="dopri5", energy_regularization=0.01, regularize=False,
                 ortho=False):
        super().__init__()
        self.func = func
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.energy_regularization = energy_regularization
        self.regularize = regularize
        self.ortho = ortho
        self.nfe = 0
        self.gauss_legendre = {
            1: torch.tensor([-1, 0, 1]),
            2: torch.tensor([-1, -0.57735, 0.57735, 1]),
            3: torch.tensor([-1, -0.77459, 0, 0.77459, 1]),
            4: torch.tensor([-1, -0.86113, -0.33998, 0.33998, 0.86113, 1]),
            5: torch.tensor([-1, -0.90618, -0.53846, 0, 0.53846, 0.90618, 1])
        }

    def integrate(self, t0, t1, x0, nlinspace=1, method=None, gauss=False, save=None, atol=None):
        """
        t0: start time of shape [n]
        t1: end time of shape [n]
        x0: initial state of shape [n, d]
        """
        if save is not None:
            save_timestamp = save
        elif gauss:
            save_timestamp = self.gauss_legendre[nlinspace].to(t0)
        else:
            save_timestamp = torch.linspace(self.start_time, self.end_time, nlinspace + 1).to(t0)
        method = method or self.method
        atol = atol or self.atol

        solution = odeint(
            self,
            (t0, t1, torch.zeros(1).to(x0[0]), x0),
            save_timestamp,
            rtol=self.rtol,
            atol=self.atol,
            method=method,
            options={"step_size": atol}
        )
        _, _, energy, xs = solution
        if gauss:
            xs = xs[1: -1, ...]

        if self.regularize:
            reg = energy * self.energy_regularization
            return WrapRegularization.apply(reg, xs)[0]
        return xs

    def forward(self, s, state):
        """Solves the same dynamics but uses a dummy variable that always integrates [0, 1]."""
        self.nfe += 1
        t0, t1, _, x = state

        ratio = (t1 - t0) / (self.end_time - self.start_time)
        t = (s - self.start_time) * ratio + t0

        with torch.enable_grad():
            dx = self.func(t, x)

            if self.ortho:
                dx = dx - (dx * x).sum(dim=-1, keepdim=True) / (x * x).sum(dim=-1, keepdim=True) * x
            dx = dx * ratio.reshape(-1, *([1] * (dx.ndim - 1)))

            d_energy = torch.sum(dx * dx) / x.numel()

        if not self.training:
            dx = dx.detach()

        return tuple([torch.zeros_like(t0), torch.zeros_like(t1), d_energy, dx])

    def extra_repr(self):
        return f"method={self.method}, atol={self.atol}, rtol={self.rtol}, energy={self.energy_regularization}"


ACTFNS = {
    "softplus": (lambda dim: nn.Softplus()),
    "tanh": (lambda dim: nn.Tanh()),
    "swish": TimeDependentSwish,
    "relu": (lambda dim: nn.ReLU()),
    'leakyrelu': (lambda dim: nn.LeakyReLU()),
    'sigmoid': (lambda dim: nn.Sigmoid()),
}

LAYERTYPES = {
    "concatsquash": ConcatSquashLinear,
    "concat": ConcatLinear_v2,
    "concatlinear": ConcatLinear_v2,
    "concatnorm": ConcatLinearNorm,
}


def build_fc_odefunc(dim=2, hidden_dims=[64, 64, 64], out_dim=None, nonzero_dim=None, actfn="softplus",
                     layer_type="concat",
                     zero_init=True, actfirst=False):
    assert layer_type in LAYERTYPES, f"layer_type must be one of {LAYERTYPES.keys()} but was given {layer_type}"
    layer_fn = LAYERTYPES[layer_type]
    if layer_type == "concatlinear":
        hidden_dims = None

    nonzero_dim = dim if nonzero_dim is None else nonzero_dim
    out_dim = out_dim or hidden_dims[-1]
    if hidden_dims:
        dims = [dim] + list(hidden_dims)
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(layer_fn(d_in, d_out))
            layers.append(ACTFNS[actfn](d_out))
        layers.append(layer_fn(hidden_dims[-1], out_dim))
        layers.append(ACTFNS[actfn](out_dim))
    else:
        layers = [layer_fn(dim, out_dim), ACTFNS[actfn](out_dim)]

    if actfirst and len(layers) > 1:
        layers = layers[1:]

    if nonzero_dim < dim:
        # zero out weights for auxiliary inputs.
        layers[0]._layer.weight.data[:, nonzero_dim:].fill_(0)

    if zero_init:
        for m in layers[-2].modules():
            if isinstance(m, nn.Linear):
                m.weight.data.fill_(0)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    return SequentialDiffEq(*layers)
