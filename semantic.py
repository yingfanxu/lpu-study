from utils import *
from common import *

def Broadcast(x, slice_m, slice_n, br_m, br_n):
    M1 = ceil_div(slice_m, M0)
    N1 = ceil_div(slice_n, N0)
    if(br_m and br_n):
        assert x.ndim == 0
        x_m1n1m0n0 = x.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(M1, N1, M0, N0)
    elif(br_m and not br_n):
        assert x.ndim == 1 and x.shape[0] == slice_n
        x_mn = x.unsqueeze(0).expand(slice_m, slice_n)
        x_m1m0n1n0 = torch.zeros(
            (M1 * M0, N1 * N0),
            dtype=x.dtype,
            device=x.device,
        )
        x_m1m0n1n0[:slice_m, :slice_n] = x_mn
        x_m1n1m0n0 = x_m1m0n1n0.reshape(M1, M0, N1, N0).permute(0, 2, 1, 3)
    elif(not br_m and br_n):
        assert x.ndim == 1 and x.shape[0] == slice_m
        x_mn = x.unsqueeze(1).expand(slice_m, slice_n)
        x_m1m0n1n0 = torch.zeros(
            (M1 * M0, N1 * N0),
            dtype=x.dtype,
            device=x.device,
        )
        x_m1m0n1n0[:slice_m, :slice_n] = x_mn
        x_m1n1m0n0 = x_m1m0n1n0.reshape(M1, M0, N1, N0).permute(0, 2, 1, 3)
    else:
        assert False, "x data must broadcast in M or N dimension"
    return x_m1n1m0n0

def Binary(x1, x2, add_en, sub_en, max_en, min_en, mul_en, div_en):
    enabled_ops = sum([
        add_en,
        sub_en,
        max_en,
        min_en,
        mul_en,
        div_en,
    ])
    assert enabled_ops == 1, "exactly one binary op must be enabled"

    if add_en:
        return x1 + x2
    if sub_en:
        return x1 - x2
    if max_en:
        return torch.max(x1, x2)
    if min_en:
        return torch.min(x1, x2)
    if mul_en:
        return x1 * x2
    if div_en:
        return x1 / x2

def Unary(x, neg_en, clamp_en, clamp_min, clamp_max, exp_en, sqrt_en, pow_en, recp_en):
    if neg_en:
        x = -x
    if clamp_en:
        x = torch.clamp(x, clamp_min, clamp_max)
    if exp_en:
        x = torch.exp(x)
    if sqrt_en:
        x = torch.sqrt(x)
    if pow_en:
        x = torch.pow(x, 2)
    if recp_en:
        x = 1.0 / x
    return x

def Reduce(
    x,
    slice_m,
    slice_n,
    reduce_m_en,
    reduce_n_en,
    reduce_mode,
):
    assert x.ndim == 4

    M1, N1, M0, N0 = x.shape

    x = x.permute(0, 2, 1, 3)
    x = x.reshape(M1 * M0, N1 * N0)
    x = x[:slice_m, :slice_n]
    if reduce_mode == 0:
        if reduce_n_en:
            x = torch.max(x, dim=1).values
        if reduce_m_en:
            x = torch.max(x, dim=0).values
    elif reduce_mode == 1:
        if reduce_n_en:
            x = torch.min(x, dim=1).values
        if reduce_m_en:
            x = torch.min(x, dim=0).values
    elif reduce_mode == 2:
        if reduce_n_en:
            x = torch.sum(x, dim=1)
        if reduce_m_en:
            x = torch.sum(x, dim=0)
    elif reduce_mode == 3:
        if reduce_n_en:
            x = torch.mean(x, dim=1)
        if reduce_m_en:
            x = torch.mean(x, dim=0)
    return x

def Attention(x, context, mask=None):
    attn_weights = torch.matmul(x, context.transpose(-2, -1))
    if mask is not None:
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
    attn_weights = torch.softmax(attn_weights, dim=-1)
    output = torch.matmul(attn_weights, context)
    return output
