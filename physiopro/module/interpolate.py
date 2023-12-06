# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


def _setup_hermite_cubic_coeffs_w_backward_differences(times, coeffs, derivs, device=None):
    """Compute backward hermite from linear coeffs."""
    x_prev = coeffs[..., :-1, :]
    x_next = coeffs[..., 1:, :]
    # Let x_0 - x_{-1} = x_1 - x_0
    derivs_prev = torch.cat((derivs[..., [0], :], derivs[..., :-1, :]), axis=-2)
    derivs_next = derivs
    x_diff = x_next - x_prev
    t_diff = (times[1:] - times[:-1]).unsqueeze(-1)
    # Coeffs
    a = x_prev
    b = derivs_prev
    two_c = 2 * (3 * (x_diff / t_diff - b) - derivs_next + derivs_prev) / t_diff
    three_d = (1 / t_diff ** 2) * (derivs_next - b) - (two_c) / t_diff
    coeffs = torch.cat([a, b, two_c, three_d], dim=-1).to(device)
    return coeffs


def hermite_cubic_coefficients_with_backward_differences(x, t=None):
    """Computes the coefficients for hermite cubic splines with backward differences.

    Arguments:
        As `torchcde.linear_interpolation_coeffs`.

    Returns:
        A tensor, which should in turn be passed to `torchcde.CubicSpline`.
    """
    # Linear coeffs
    coeffs = linear_interpolation_coeffs(x, t=t)

    if t is None:
        t = torch.linspace(0, coeffs.size(-2) - 1, coeffs.size(-2), dtype=coeffs.dtype, device=coeffs.device)

    # Linear derivs
    derivs = (coeffs[..., 1:, :] - coeffs[..., :-1, :]) / (t[1:] - t[:-1]).unsqueeze(-1)

    # Use the above to compute hermite coeffs
    hermite_coeffs = _setup_hermite_cubic_coeffs_w_backward_differences(t, coeffs, derivs, device=coeffs.device)

    return hermite_coeffs


def linear_interpolation_coeffs(x, t):
    coeffs = []
    for i in range(x.size(0)):
        index = torch.where(~torch.isnan(x[i, :, 0]))[0]

        if len(index) == 1:
            coeff = x[i][index[0]].unsqueeze(0)
        elif len(index) == 0:
            coeff = torch.zeros_like(x[i])
        else:
            coeff = x[i].clone()
            if coeff[0].isnan().any():
                coeff[:index[0], :] = x[i][index[0]].unsqueeze(0)
            if coeff[-1].isnan().any():
                coeff[index[-1] + 1:, :] = x[i][index[-1]].unsqueeze(0)
            for (pre, post) in zip(index[:-1], index[1:]):
                rate = []
                for j in range(pre + 1, post):
                    rate.append(((t[j] - t[pre]) / (t[post] - t[pre])))
                rate = torch.tensor(rate).to(coeff)
                coeff[pre + 1: post] = x[i][pre].unsqueeze(0) \
                                       + rate.unsqueeze(-1) * (x[i][post] - x[i][pre]).unsqueeze(0)
        coeffs.append(coeff.unsqueeze(0))
    return torch.cat(coeffs, dim=0)
