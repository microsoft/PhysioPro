# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torchcde
import torch.nn as nn
import torch
from ..module.ode import build_fc_odefunc, TimeVariableODE
from ..module.interpolate import linear_interpolation_coeffs, hermite_cubic_coefficients_with_backward_differences

PAD = 0


class InterpLinear(nn.Module):
    def __init__(self, d_model, d_out=None, args_interp=None, norm=True):
        super().__init__()
        d_out = d_out or d_model
        if args_interp.linear_type != 'inside':
            self.lin_outside = nn.Linear(d_model, d_out)
            nn.init.xavier_uniform_(self.lin_outside.weight)
        self.norm = norm

        self.gauss_weight = {
            1: torch.tensor([2]),
            2: torch.tensor([1, 1]),
            3: torch.tensor([0.55555, 0.88888, 0.55555]),
            4: torch.tensor([0.34785, 0.65214, 0.65214, 0.34785]),
            5: torch.tensor([0.23692, 0.47863, 0.56888, 0.47863, 0.23692])
        }
        self.gauss_legendre = {
            1: torch.tensor([0]),
            2: torch.tensor([-0.57735, 0.57735]),
            3: torch.tensor([-0.77459, 0, 0.77459]),
            4: torch.tensor([-0.86113, -0.33998, 0.33998, 0.86113]),
            5: torch.tensor([-0.90618, -0.53846, 0, 0.53846, 0.90618])
        }

        self.nlinspace = args_interp.nlinspace
        self.approximate_method = args_interp.approximate_method
        self.interpolation = args_interp.interpolate
        self.linear_type = args_interp.linear_type
        if self.approximate_method == 'bilinear':
            self.nlinspace = 1
        self.atol = args_interp.itol
        self.d_model = d_model

    def pre_integrals(self, x, t):
        """
        x: hidden state of shape: [B, T, D]
        t: event time of shape: [B, T]
        """
        # fill PAD in t with the last observed time
        pad_num = torch.sum(t[:, 1:] == PAD, dim=-1)

        T = t.clone()
        for i in range(t.size(0)):
            if pad_num[i] == 0:
                continue
            T[i][-pad_num[i]:] = T[i][-pad_num[i] - 1]

        tt, sort_index = torch.unique(T, sorted=True, return_inverse=True)

        xx = torch.full((x.size(0), len(tt), x.size(-1)), float('nan')).to(x)

        r = torch.arange(x.size(0)).reshape(-1, 1).repeat(1, x.size(1)).reshape(-1)

        # fill in non-nan values
        xx[r.numpy(), sort_index.reshape(-1).cpu().numpy(), :] = x.reshape(x.size(0) * x.size(1), -1)  # [B, TT, D]

        # interpolation
        if self.interpolation == 'linear':
            coeffs = linear_interpolation_coeffs(xx, t=tt)
        elif self.interpolation == 'cubic':
            coeffs = hermite_cubic_coefficients_with_backward_differences(xx, t=tt)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")
        return coeffs, tt

    def forward(self, x, t, approximate_method=None):
        """
            x: hidden state vector for q or k with shape [B, T, D]
            t: event time with shape [B, T]

            Return: [B, T, T, K, D]
        """
        T, Q = t.shape[1], t.shape[1]

        if self.linear_type == 'before':
            x = self.lin_outside(x)

        coeffs, intervals = self.pre_integrals(x, t)
        if self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs, t=intervals)
        elif self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs, t=intervals)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        t0 = t.unsqueeze(-2).repeat(1, Q, 1)  # [B, Q, T]
        t1 = t.unsqueeze(-1).repeat(1, 1, T)  # [B, Q, T]

        approximate_method = approximate_method or self.approximate_method
        if approximate_method == 'bilinear':
            interp_t = torch.cat((t0.unsqueeze(-1), t1.unsqueeze(-1)), dim=-1)
        elif approximate_method == 'gauss':
            interp_rate = self.gauss_legendre[self.nlinspace].to(t0.device)
            interp_t = t0.unsqueeze(-1) + \
                       (t1.unsqueeze(-1) - t0.unsqueeze(-1)) * (interp_rate + 1) / 2
        else:
            interp_t = t1.unsqueeze(-1)

        discret_t = (interp_t / self.atol).long()  # [B, T, T, K]

        linspace_t = torch.linspace(0, t.max().item() + 5 * self.atol, int(t.max().item() / self.atol + 6))
        interp_f = X.evaluate(linspace_t)   # [B, L, D]

        discret_t = discret_t.reshape(t0.shape[0], -1).detach().cpu()  # [B, T * T * K]
        idx = torch.arange(t0.shape[0]).unsqueeze(-1).repeat(1, discret_t.shape[1])  # [B, T * T * K]

        x = interp_f[idx.reshape(-1), discret_t.reshape(-1), ...]
        x = x.reshape(t0.shape[0], t0.shape[1], t0.shape[2], -1, self.d_model)

        if self.linear_type == 'after':
            x = self.lin_outside(x)

        if self.approximate_method == 'bilinear':
            x = x * 0.5
        else:
            x = x * self.gauss_weight[self.nlinspace].to(x).reshape(1, 1, 1, -1, 1) * 0.5

        if not self.norm:
            x = x * (t1 - t0).unsqueeze(-1).unsqueeze(-1)
        return x

    def interpolate(self, x, t, qt, mask=None, approximate_method=None):
        """
            x: hidden state vector for q or k with shape [B, T, D]
            t: event time with shape [B, T]
            qt: query time with shape [B, Q]
            mask: mask for unknown events [B, Q, T]

            Return: [B, Q, T, K, D]
        """

        T, Q = t.shape[1], qt.shape[1]

        # current_times = torch.ones(t.shape[0], Q).to(t.device) * 100000

        if self.linear_type == 'before':
            x = self.lin_outside(x)

        coeffs, intervals = self.pre_integrals(x, t)
        if self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs, t=intervals)
        elif self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs, t=intervals)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        t0 = t.unsqueeze(-2).repeat(1, Q, 1)  # [B, Q, T]
        t1 = qt.unsqueeze(-1).repeat(1, 1, T)  # [B, Q, T]

        approximate_method = approximate_method or self.approximate_method
        if approximate_method == 'bilinear':
            interp_t = torch.cat((t0.unsqueeze(-1), t1.unsqueeze(-1)), dim=-1)
        elif approximate_method == 'gauss':
            interp_rate = self.gauss_legendre[self.nlinspace].to(t0.device)
            interp_t = t0.unsqueeze(-1) + \
                       (t1.unsqueeze(-1) - t0.unsqueeze(-1)) * (interp_rate + 1) / 2
        else:
            interp_t = t1.unsqueeze(-1)

        # current_times = current_times.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, interp_t.shape[2], interp_t.shape[3])
        # interp_t = torch.min(interp_t, current_times)

        discret_t = (interp_t / self.atol).long()  # [B, Q, T, K]

        linspace_t = torch.linspace(0, qt.max().item() + 5 * self.atol, int(qt.max().item() / self.atol + 6))
        interp_f = X.evaluate(linspace_t)   # [B, L, D]

        discret_t = discret_t.reshape(t0.shape[0], -1).detach().cpu()  # [B, Q * T * K]
        idx = torch.arange(t0.shape[0]).unsqueeze(-1).repeat(1, discret_t.shape[1])  # [B, Q * T * K]

        x = interp_f[idx.reshape(-1), discret_t.reshape(-1), ...]
        x = x.reshape(t0.shape[0], t0.shape[1], t0.shape[2], -1, self.d_model)

        if self.linear_type == 'after':
            x = self.lin_outside(x)

        if self.approximate_method == 'bilinear':
            x = x * 0.5
        else:
            x = x * self.gauss_weight[self.nlinspace].to(x).reshape(1, 1, 1, -1, 1) * 0.5

        if not self.norm:
            x = x * (t1 - t0).reshape(x.size(0), x.size(-2), x.size(-2)).unsqueeze(-1).unsqueeze(-1)

        return x


class ODELinear(nn.Module):
    def __init__(self, d_model, d_out=None, args_ode=None, norm=True):
        super().__init__()
        d_out = d_out or d_model
        self.norm = norm
        if args_ode.linear_type == 'inside':
            self.ode_func = build_fc_odefunc(d_model, out_dim=d_model, actfn=args_ode.actfn, layer_type=args_ode.layer_type,
                                             zero_init=args_ode.zero_init, hidden_dims=[d_model])
        else:
            if args_ode.linear_type == 'before':
                self.ode_func = build_fc_odefunc(d_out, out_dim=d_out, actfn=args_ode.actfn,
                                                 layer_type=args_ode.layer_type,
                                                 zero_init=args_ode.zero_init, hidden_dims=None)
                self.lin_outside = nn.Linear(d_model, d_out)
                nn.init.xavier_uniform_(self.lin_outside.weight)
            else:
                self.ode_func = build_fc_odefunc(d_model, out_dim=d_model, actfn=args_ode.actfn,
                                                 layer_type=args_ode.layer_type,
                                                 zero_init=args_ode.zero_init, hidden_dims=None)
                self.lin_outside = nn.Linear(d_model, d_out)
                nn.init.xavier_uniform_(self.lin_outside.weight)

        self.linear_type = args_ode.linear_type
        self.ode = TimeVariableODE(self.ode_func, atol=args_ode.atol, rtol=args_ode.rtol,
                                   method=args_ode.method, regularize=args_ode.regularize)
        self.approximate_method = args_ode.approximate_method
        self.gauss_weight = {
            1: torch.tensor([2]),
            2: torch.tensor([1, 1]),
            3: torch.tensor([0.55555, 0.88888, 0.55555]),
            4: torch.tensor([0.34785, 0.65214, 0.65214, 0.34785]),
            5: torch.tensor([0.23692, 0.47863, 0.56888, 0.47863, 0.23692])
        }
        self.d_model = d_model
        self.nlinspace = args_ode.nlinspace
        if self.approximate_method == 'bilinear' or self.approximate_method == 'last':
            self.nlinspace = 1

    def forward(self, x, t, approximate_method=None):
        """
        x: hidden state vector for q or k with shape [B, T, D]
        t: event time with shape [B, T]

        Return: [B, T, T, K, D]
        """

        T, Q = t.shape[1], t.shape[1]
        BS = t.shape[0]

        x0 = x.unsqueeze(-3).repeat(1, Q, 1, 1)  # [B, Q, T, D]
        t0 = t.unsqueeze(-2).repeat(1, Q, 1)  # [B, Q, T]
        t1 = t.unsqueeze(-1).repeat(1, 1, T)  # [B, Q, T]

        x0 = x0.reshape(-1, self.d_model)
        t0 = t0.reshape(-1)
        t1 = t1.reshape(-1)

        if self.linear_type == 'before':
            x0 = self.lin_outside(x0)

        y = self.ode.integrate(t0, t1, x0, nlinspace=self.nlinspace, gauss=self.approximate_method == 'gauss')

        if self.linear_type == 'after':
            y = self.lin_outside(y)

        y = y.reshape(y.size(0), BS, Q, T, -1)  # [K, B, Q, T, D]
        y = y.permute(1, 2, 3, 0, 4)

        approximate_method = approximate_method or self.approximate_method

        if approximate_method == 'bilinear':
            y = y * 0.5
        elif approximate_method == 'gauss':
            y = y * self.gauss_weight[self.nlinspace].to(x).reshape(1, 1, 1, -1, 1) * 0.5
        else:
            y = y[..., -1, :]
        if not self.norm:
            y = y * (t1 - t0).reshape(x.size(0), x.size(-2), x.size(-2)).unsqueeze(-1).unsqueeze(-1)
        return y

    def interpolate(self, x, t, qt, approximate_method=None):
        """
            x: hidden state vector for q or k with shape [B, T, D]
            t: event time with shape [B, T]
            qt: query time with shape [B, Q]
            Return: [B, Q, T, K, D]
        """

        T, Q = t.shape[1], qt.shape[1]

        x0 = x.unsqueeze(-3).repeat(1, Q, 1, 1)  # [B, Q, T, D]
        t0 = t.unsqueeze(-2).repeat(1, Q, 1)  # [B, Q, T]
        t1 = qt.unsqueeze(-1).repeat(1, 1, T)  # [B, Q, T]

        x0 = x0.reshape(-1, self.d_model)
        t0 = t0.reshape(-1)
        t1 = t1.reshape(-1)

        if self.linear_type == 'before':
            x0 = self.lin_outside(x0)

        y = self.ode.integrate(t0, t1, x0, nlinspace=self.nlinspace, gauss=self.approximate_method == 'gauss')

        if self.linear_type == 'after':
            y = self.lin_outside(y)

        y = y.reshape(y.size(0), -1, Q, T, self.d_model)  # [K, B, Q, T, D]
        y = y.permute(1, 2, 3, 0, 4)

        approximate_method = approximate_method or self.approximate_method

        if approximate_method == 'bilinear':
            y = y * 0.5
        elif approximate_method == 'gauss':
            y = y * self.gauss_weight[self.nlinspace].to(x).reshape(1, 1, 1, -1, 1) * 0.5
        else:
            y = y[..., -1:, :]

        if not self.norm:
            y = y * (t1 - t0).reshape(x.size(0), x.size(-2), x.size(-2)).unsqueeze(-1).unsqueeze(-1)
        return y
