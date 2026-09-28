# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import glob
import json
import math
import os
import types
import warnings
from typing import Any, Dict, Optional, Union,Tuple

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version, logging
from torch import nn
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from ..dist import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    usp_attn_forward, xFuserLongContextAttention)
from ..utils import cfg_skip
from .attention_utils import attention
from .cache_utils import TeaCache
from .wan_camera_adapter import SimpleAdapter
from diffusers.models.attention_processor import Attention
import torch.nn.functional as F


@torch.compiler.disable()
def build_freqs_i(grid_size, freqs, f_off=0, h_off=0, w_off=0):
    """Build 3D RoPE for a specific latent (with TPB offsets)"""
    f, h, w = grid_size
    c = freqs.size(1)
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    freq_f = freqs_split[0][f_off : f_off + f].view(f, 1, 1, -1).expand(f, h, w, -1)
    freq_h = freqs_split[1][h_off : h_off + h].view(1, h, 1, -1).expand(f, h, w, -1)
    freq_w = freqs_split[2][w_off : w_off + w].view(1, 1, w, -1).expand(f, h, w, -1)

    freqs_i = torch.cat([freq_f, freq_h, freq_w], dim=-1).reshape(f * h * w, 1, -1)
    return freqs_i

@amp.autocast(enabled=False)
def rope_apply_precomputed(x, freqs):
    """Minimal global RoPE application function"""
    x_complex = torch.view_as_complex(x.to(torch.float32).reshape(*x.shape[:3], -1, 2))
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(3)
    return x_rotated.type_as(x)

def separated_patch_attention_routing(q, k, v, L_sv, L_si, L_ti, L_tv, global_freqs, dtype=torch.bfloat16):
    freqs_v1, (freqs_sv_first_v2, freqs_si_v2, freqs_ti_v2) = global_freqs

    # Compute the number of tokens in one frame
    L_sv_first_frame = freqs_sv_first_v2.size(0)
    # === View 1 (Video View) ===
    q_v1 = rope_apply_precomputed(q, freqs_v1)
    k_v1 = rope_apply_precomputed(k, freqs_v1)
    
    # A & B: sv and si attend only to themselves
    q_sv, k_sv, v_sv = q_v1[:, :L_sv, :, :], k_v1[:, :L_sv, :, :], v[:, :L_sv, :, :]
    out_sv = attention(q_sv, k_sv, v_sv, dtype=dtype)
    del q_sv, k_sv, v_sv
    
    q_si, k_si, v_si = q_v1[:, L_sv:L_sv+L_si, :, :], k_v1[:, L_sv:L_sv+L_si, :, :], v[:, L_sv:L_sv+L_si, :, :]
    out_si = attention(q_si, k_si, v_si, dtype=dtype)
    del q_si, k_si, v_si
    
    # D: tv attends to everything
    q_tv = q_v1[:, L_sv+L_si+L_ti:, :, :]
    out_tv = attention(q_tv, k_v1, v, dtype=dtype)
    del q_tv, q_v1, k_v1 
    
    # === View 2 (Image View) ===
    # rope_apply_precomputed calls .to(float32) internally, which creates a new contiguous copy, so no prior .contiguous() is needed
    # First frame of sv
    k_sv_first_raw = k[:, :L_sv_first_frame, :, :]
    v_sv_first     = v[:, :L_sv_first_frame, :, :]

    # si (condition only, needs only k, v)
    k_si_raw = k[:, L_sv : L_sv+L_si, :, :]
    v_si     = v[:, L_sv : L_sv+L_si, :, :]

    # ti (the querying side, needs full q, k, v)
    q_ti_raw = q[:, L_sv+L_si : L_sv+L_si+L_ti, :, :]
    k_ti_raw = k[:, L_sv+L_si : L_sv+L_si+L_ti, :, :]
    v_ti     = v[:, L_sv+L_si : L_sv+L_si+L_ti, :, :]

    # 2. Rotate each with its own frequencies under the v2 view
    k_sv_first_v2 = rope_apply_precomputed(k_sv_first_raw, freqs_sv_first_v2.unsqueeze(0))
    k_si_v2       = rope_apply_precomputed(k_si_raw, freqs_si_v2.unsqueeze(0))
    
    q_ti_v2 = rope_apply_precomputed(q_ti_raw, freqs_ti_v2.unsqueeze(0))
    k_ti_v2 = rope_apply_precomputed(k_ti_raw, freqs_ti_v2.unsqueeze(0))

    # Free the raw slices immediately
    del k_sv_first_raw, k_si_raw, q_ti_raw, k_ti_raw
    
    # 3. Assemble the full Image View Key and Value (all attended-to tokens)
    k_img_view = torch.cat([k_ti_v2, k_sv_first_v2, k_si_v2], dim=1)
    v_img_view = torch.cat([v_ti, v_sv_first, v_si], dim=1)
    
    # 4. Only ti is the active querier; compute its own attention
    out_ti = attention(q_ti_v2, k_img_view, v_img_view, dtype=dtype)
    
    # Garbage collection
    del q_ti_v2, k_img_view, v_img_view, q, k, v

    # Reassemble the four parts in their original order (sv, si, ti, tv)
    out = torch.cat([out_sv, out_si, out_ti, out_tv], dim=1)
    return out

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.to(torch.float32)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# modified from https://github.com/thu-ml/RIFLEx/blob/main/riflex_utils.py
@amp.autocast(enabled=False)
def get_1d_rotary_pos_embed_riflex(
    pos: Union[np.ndarray, int],
    dim: int,
    theta: float = 10000.0,
    use_real=False,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
    L_test_scale: Optional[int] = None,
):
    """
    RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
        L_test (`int`, *optional*, defaults to None): the number of frames for inference
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    freqs = 1.0 / torch.pow(theta,
        torch.arange(0, dim, 2).to(torch.float64).div(dim))

    # === Riflex modification start ===
    # Reduce the intrinsic frequency to stay within a single period after extrapolation (see Eq. (8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply by 0.9 to keep the extrapolated length below 90% of a single period.
    if k is not None:
        freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===
    if L_test_scale is not None:
        freqs[k-1] = freqs[k-1] / L_test_scale

    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


@amp.autocast(enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


def rope_apply_qk(q, k, grid_sizes, freqs):
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)
    return q, k


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, slice_lens, global_freqs, dtype=torch.bfloat16, t=0):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
        k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
        v = self.v(x.to(dtype)).view(b, s, n, d)

        # Do not apply global RoPE here; pass the raw q, k, v to the routing function
        L_sv, L_si, L_ti, L_tv = slice_lens
        x = separated_patch_attention_routing(q, k, v, L_sv, L_si, L_ti, L_tv, global_freqs, dtype=dtype)

        x = x.to(dtype).flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context_tar, context_ref, slice_lens, dtype=torch.bfloat16, t=0):
        b, s, n, d = x.size(0), x.size(1), self.num_heads, self.head_dim
        L_sv, L_si, L_ti, L_tv = slice_lens

        # Compute Q only for target tokens (sv/si need no Q; skipping saves a L_src/L_total fraction of the linear compute)
        # Note: slicing then view yields a contiguous tensor; memory is freed immediately after del q_tgt
        x_tgt = x[:, L_sv+L_si:].to(dtype)
        q_tgt = self.norm_q(self.q(x_tgt)).view(b, -1, n, d)
        del x_tgt
        q_ti = q_tgt[:, :L_ti].contiguous()
        q_tv = q_tgt[:, L_ti:].contiguous()
        del q_tgt

        # Prepare Key and Value
        k_ref = self.norm_k(self.k(context_ref.to(dtype))).view(b, -1, n, d)
        v_ref = self.v(context_ref.to(dtype)).view(b, -1, n, d)

        k_tar = self.norm_k(self.k(context_tar.to(dtype))).view(b, -1, n, d)
        v_tar = self.v(context_tar.to(dtype)).view(b, -1, n, d)

        # Run separate CrossAttention
        out_ti = attention(q_ti, k_ref, v_ref, dtype=dtype).flatten(2)
        del q_ti, k_ref, v_ref

        out_tv = attention(q_tv, k_tar, v_tar, dtype=dtype).flatten(2)
        del q_tv, k_tar, v_tar

        # Apply the output projection once to the concatenated ti/tv outputs
        out_tgt = torch.cat([out_ti, out_tv], dim=1)
        del out_ti, out_tv
        out_tgt = self.o(out_tgt)

        # Put the result back at the target positions; zeros for sv/si mean no residual update there (F.pad avoids allocating a separate zeros tensor)
        out = F.pad(out_tgt, (0, 0, L_sv + L_si, 0))
        return out



class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        # compute attention
        x = attention(q, k, v, k_lens=context_lens)
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'cross_attn': WanCrossAttention,
}

class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        slice_lens,
        global_freqs,
        context_tar,
        context_ref,
        dtype=torch.bfloat16,
        t=0,
    ):
        L_sv, L_si, L_ti, L_tv = slice_lens
        L_cond = L_sv + L_si

        if isinstance(e, tuple):
            cond_e0, tgt_e0 = e
            cm = (self.modulation + cond_e0).chunk(6, dim=1)
            tm = (self.modulation + tgt_e0).chunk(6, dim=1)

            temp_x = torch.cat([
                self.norm1(x[:, :L_cond]) * (1 + cm[1]) + cm[0],
                self.norm1(x[:, L_cond:]) * (1 + tm[1]) + tm[0],
            ], dim=1).to(dtype)

            y = self.self_attn(temp_x, slice_lens, global_freqs, dtype, t=t)
            del temp_x
            x = torch.cat([
                x[:, :L_cond] + y[:, :L_cond] * cm[2],
                x[:, L_cond:] + y[:, L_cond:] * tm[2],
            ], dim=1)
            del y

            y_cross = self.cross_attn(self.norm3(x), context_tar, context_ref, slice_lens, dtype, t=t)
            x = x + y_cross
            del y_cross

            temp_x = torch.cat([
                self.norm2(x[:, :L_cond]) * (1 + cm[4]) + cm[3],
                self.norm2(x[:, L_cond:]) * (1 + tm[4]) + tm[3],
            ], dim=1).to(dtype)
            y = self.ffn(temp_x)
            del temp_x
            x = torch.cat([
                x[:, :L_cond] + y[:, :L_cond] * cm[5],
                x[:, L_cond:] + y[:, L_cond:] * tm[5],
            ], dim=1)
            del y
            return x

        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [e.squeeze(2) for e in e]
        else:
            e = (self.modulation + e).chunk(6, dim=1)

        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)

        y = self.self_attn(temp_x, slice_lens, global_freqs, dtype, t=t)
        x = x + y * e[2]

        def cross_attn_ffn(x, context_tar, context_ref, slice_lens, e):
            y_cross = self.cross_attn(self.norm3(x), context_tar, context_ref, slice_lens, dtype, t=t)
            x = x + y_cross

            temp_x = self.norm2(x) * (1 + e[4]) + e[3]
            temp_x = temp_x.to(dtype)

            y = self.ffn(temp_x)
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context_tar, context_ref, slice_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, hidden_states, temb):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        if temb.dim() > 2:
            temb = (self.modulation.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            temb = [temb.squeeze(2) for temb in temb]
        else:
            temb = (self.modulation + temb.unsqueeze(1)).chunk(2, dim=1)
        
        hidden_states = (self.head(self.norm(hidden_states) * (1 + temb[1]) + temb[0]))
        return hidden_states


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class QwenRMSNorm(nn.Module):
    """Qwen2RMSNorm for Qwen embedding projection"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class QwenProjectIn(nn.Module):
    """
    Two-layer MLP to project Qwen embedding directly to transformer dim.
    Outputs dim directly, skipping text_embedding (text_embedding is designed for T5).
    """
    def __init__(self, qwen_hidden_size, dim, hidden_mult=4):
        super().__init__()
        hidden_dim = qwen_hidden_size * hidden_mult
        self.ln = QwenRMSNorm(qwen_hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(qwen_hidden_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),  # Outputs dim directly, not text_dim
        )

    def forward(self, x):
        return self.mlp(self.ln(x))



class WanTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        cross_attn_type=None,
        # Qwen encoder related parameters
        use_qwen_encoder=True,
        qwen_hidden_size=4096,  # Qwen3-VL-8B hidden_size (also matches T5 text_dim)
        use_t5=True,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        # assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attention_head_dim = int(dim/self.num_heads)
        # self.num_heads
        # 12
        # dim
        # 1536

        # self.rope = WanRotaryPosEmbed(attention_head_dim=self.attention_head_dim, patch_size=self.patch_size,max_seq_len=1024 )

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        

        
        self.use_t5 = use_t5
        
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        # Qwen encoder projection layer: qwen_hidden_size -> dim (outputs the transformer inner dim directly, skipping text_embedding)
        self.use_qwen_encoder = use_qwen_encoder
        if use_qwen_encoder:
            self.qwen_project_in = QwenProjectIn(
                qwen_hidden_size=qwen_hidden_size,
                dim=dim,  # Outputs dim directly, not text_dim
                hidden_mult=4
            )
        else:
            self.qwen_project_in = None

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        if cross_attn_type is None:
            cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d
        self.dim = dim
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1
        )
        

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
        
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:], downscale_factor=downscale_factor_control_adapter)
        else:
            self.control_adapter = None

        if add_ref_conv:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None

        self.teacache = None
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None
        self.gradient_checkpointing = False
        self.all_gather = None
        self.sp_world_size = 1
        self.sp_world_rank = 0
        self.init_weights()

    def _set_gradient_checkpointing(self, *args, **kwargs):
        if "value" in kwargs:
            self.gradient_checkpointing = kwargs["value"]
            if hasattr(self, "motioner") and hasattr(self.motioner, "gradient_checkpointing"):
                self.motioner.gradient_checkpointing = kwargs["value"]
        elif "enable" in kwargs:
            self.gradient_checkpointing = kwargs["enable"]
            if hasattr(self, "motioner") and hasattr(self.motioner, "gradient_checkpointing"):
                self.motioner.gradient_checkpointing = kwargs["enable"]
        else:
            raise ValueError("Invalid set gradient checkpointing")

    def enable_teacache(
        self,
        coefficients,
        num_steps: int,
        rel_l1_thresh: float,
        num_skip_start_steps: int = 0,
        offload: bool = True,
    ):
        self.teacache = TeaCache(
            coefficients, num_steps, rel_l1_thresh=rel_l1_thresh, num_skip_start_steps=num_skip_start_steps, offload=offload
        )

    def share_teacache(
        self,
        transformer = None,
    ):
        self.teacache = transformer.teacache

    def disable_teacache(self):
        self.teacache = None

    def enable_cfg_skip(self, cfg_skip_ratio, num_steps):
        if cfg_skip_ratio != 0:
            self.cfg_skip_ratio = cfg_skip_ratio
            self.current_steps = 0
            self.num_inference_steps = num_steps
        else:
            self.cfg_skip_ratio = None
            self.current_steps = 0
            self.num_inference_steps = None

    def share_cfg_skip(
        self,
        transformer = None,
    ):
        self.cfg_skip_ratio = transformer.cfg_skip_ratio
        self.current_steps = transformer.current_steps
        self.num_inference_steps = transformer.num_inference_steps

    def disable_cfg_skip(self):
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None

    def enable_riflex(
        self,
        k = 6,
        L_test = 66,
        L_test_scale = 4.886,
    ):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                get_1d_rotary_pos_embed_riflex(2048, self.d - 4 * (self.d // 6), use_real=False, k=k, L_test=L_test, L_test_scale=L_test_scale),
                rope_params(2048, 2 * (self.d // 6)),
                rope_params(2048, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def disable_riflex(self):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                rope_params(1024, self.d - 4 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def enable_multi_gpus_inference(self,):
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather

        # For normal model.
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn)

        # For vace model.
        if hasattr(self, 'vace_blocks'):
            for block in self.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)

    def forward(
        self,
        tar_latents,
        tar_img_latents,
        src_vid_latents,
        ref_img_latents,
        t,
        context,
        context_qwen,
        ref_context_qwen,
        cond_flag=True,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shap e [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            cond_flag (`bool`, *optional*, defaults to True):
                Flag to indicate whether to forward the condition input

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # Wan2.2 don't need a clip.
        # if self.model_type == 'i2v':
        #     assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        dtype = tar_latents.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        # if y is not None:
        #     x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # ==========================================
        # 1. Separate Patchify and correct Grid retrieval
        # ==========================================
        # First pass through Embedding to get [1, Dim, F_p, H_p, W_p]
        tokens_tv_3d = [self.patch_embedding(u.unsqueeze(0)) for u in tar_latents]
        tokens_ti_3d = [self.patch_embedding(u.unsqueeze(0)) for u in tar_img_latents]
        tokens_sv_3d = [self.patch_embedding(u.unsqueeze(0)) for u in src_vid_latents]
        tokens_si_3d = [self.patch_embedding(u.unsqueeze(0)) for u in ref_img_latents]

        # Fix: shape[2:] must be taken from the 3D tokens to get the correct (F_p, H_p, W_p)
        grid_tv = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in tokens_tv_3d])
        grid_ti = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in tokens_ti_3d])
        grid_sv = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in tokens_sv_3d])
        grid_si = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in tokens_si_3d])

        # Flatten [1, Dim, F_p, H_p, W_p] -> [1, Seq_Len, Dim]
        tokens_tv = [u.flatten(2).transpose(1, 2) for u in tokens_tv_3d]
        tokens_ti = [u.flatten(2).transpose(1, 2) for u in tokens_ti_3d]
        tokens_sv = [u.flatten(2).transpose(1, 2) for u in tokens_sv_3d]
        tokens_si = [u.flatten(2).transpose(1, 2) for u in tokens_si_3d]
        del tokens_tv_3d, tokens_ti_3d, tokens_sv_3d, tokens_si_3d

        # ==========================================
        # 2. Generate dual-coordinate RoPE (Dual-View TPB)
        # ==========================================
        F_val = grid_tv[0][0].item() 
        W_val = grid_tv[0][2].item() 

        # --- View 1: Video denoising --- 
        # [tv,sv,ti][tv,sv][tv,sv].....[si]
        freqs_sv_v1 = build_freqs_i(grid_sv[0], self.freqs, f_off=0,     h_off=0, w_off=W_val)
        freqs_si_v1 = build_freqs_i(grid_si[0], self.freqs, f_off=F_val, h_off=0, w_off=0)
        freqs_ti_v1 = build_freqs_i(grid_ti[0], self.freqs, f_off=0,     h_off=0, w_off=2*W_val)
        freqs_tv_v1 = build_freqs_i(grid_tv[0], self.freqs, f_off=0,     h_off=0, w_off=0)

        global_freqs_v1 = torch.cat([freqs_sv_v1, freqs_si_v1, freqs_ti_v1, freqs_tv_v1], dim=0).unsqueeze(0).to(device)

        # --- View 2: Image denoising --- 
        # [ti,sv][si]: the reference image (si, before adaptation) is a separate frame placed last; of sv only the first frame is kept
        # freqs_sv_v2 = build_freqs_i(grid_sv[0], self.freqs, f_off=0, h_off=0, w_off=W_val)

        grid_sv_first_frame = torch.tensor([1, grid_sv[0][1], grid_sv[0][2]], dtype=torch.long)
        freqs_sv_first_v2 = build_freqs_i(grid_sv_first_frame, self.freqs, f_off=0, h_off=0, w_off=W_val)

        # freqs_sv_v2 = torch.cat([freqs_sv_v2, freqs_sv_first_frame_v2], dim=0)

        freqs_si_v2 = build_freqs_i(grid_si[0], self.freqs, f_off=1, h_off=0, w_off=0)   
        freqs_ti_v2 = build_freqs_i(grid_ti[0], self.freqs, f_off=0, h_off=0, w_off=0)
          
        global_freqs = (global_freqs_v1, (freqs_sv_first_v2.to(device), freqs_si_v2.to(device), freqs_ti_v2.to(device)))
        # Optimization: no need to allocate tv memory again; reuse the v1 pointer directly
        # global_freqs_v2 = torch.cat([freqs_sv_v2, freqs_si_v2, freqs_ti_v2, freqs_tv_v1], dim=0).unsqueeze(0).to(device)
        
        # Aggressive memory reclaim: delete the intermediate fragment frequency tensors immediately
        del freqs_sv_v1, freqs_si_v1, freqs_ti_v1, freqs_tv_v1
        del freqs_sv_first_v2, freqs_ti_v2



        # ==========================================
        # 3. Concatenate Tokens in the same order
        # ==========================================
        x_combined = []
        slice_lens_list = []
        for i in range(len(tokens_tv)):
            L_sv = tokens_sv[i].size(1)
            L_si = tokens_si[i].size(1)
            L_ti = tokens_ti[i].size(1)
            L_tv = tokens_tv[i].size(1)
            slice_lens_list.append((L_sv, L_si, L_ti, L_tv))
            
            # Strict prefix order
            x_i = torch.cat([tokens_sv[i], tokens_si[i], tokens_ti[i], tokens_tv[i]], dim=1)
            x_combined.append(x_i)

        x = torch.cat(x_combined, dim=0) # [B, Total_Seq_Len, Dim]
        slice_lens = slice_lens_list[0]  # Assumes batch_size=1

        # time embeddings
        total_seq_len = L_sv + L_si + L_ti + L_tv
        bt = t.size(0)
        with amp.autocast(dtype=torch.float32):
            if t.dim() != 1:
                # 1. Regardless of the input, Source (sv, si) must be strictly clean at t=0
                cond_t = torch.zeros((bt, L_sv + L_si), dtype=t.dtype, device=t.device)
                
                # 2. Safely handle Target timesteps (crash-proof: truncate excess, pad shortage)
                target_len = L_ti + L_tv
                if t.size(1) < target_len:
                    pad_size = target_len - t.size(1)
                    padding = t[:, -1].unsqueeze(1).repeat(1, pad_size)
                    t_target = torch.cat([t, padding], dim=1)
                elif t.size(1) > target_len:
                    t_target = t[:, :target_len] # Fix: truncate any excess
                else:
                    t_target = t
                
                # 3. Assemble our dedicated 2D timesteps
                t_2d = torch.cat([cond_t, t_target], dim=1) # Length is exactly total_seq_len
                
                # 4. Compute the 4D AdaLN tensor (never raises an unflatten error)
                ft = t_2d.flatten()
                e_full = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, ft).unflatten(0, (bt, total_seq_len)).float())
                e0_combined = self.time_projection(e_full).unflatten(2, (6, self.dim))
                
                # Extract the global 1D time embedding for the final Head (taking the first Target timestep is reasonable)
                e = e_full[:, L_sv + L_si, :]
            else:
                # Fast path for 1D timesteps (training)
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim)) 

                cond_t = torch.zeros_like(t)
                cond_e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, cond_t).float())
                cond_e0 = self.time_projection(cond_e).unflatten(1, (6, self.dim)) 

                # Pass a tuple instead of an expanded 4D tensor, saving ~1.9GB of GPU memory
                # cond_e0 is used for sv/si (condition tokens), e0 for ti/tv (target tokens)
                e0_combined = (cond_e0, e0)

            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            # e0 = e0.to(dtype)
            # e = e.to(dtype)

        if self.use_qwen_encoder:
            context_qwen = self.qwen_project_in(context_qwen) 
            ref_context_qwen = self.qwen_project_in(ref_context_qwen)

        # context
        if self.use_t5:
            # print("Warning::::USING T5")
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))
            context = torch.cat([context_qwen, context], dim=1)
        else:
            context = context_qwen
        ref_context = ref_context_qwen

        del tokens_tv, tokens_ti, tokens_sv, tokens_si
        del x_combined

        # ==========================================
        # 6. TeaCache and Transformer Block loop
        # ==========================================


        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs): return module(*inputs)
                    return custom_forward
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    e0_combined,
                    slice_lens,
                    global_freqs,
                    context,
                    ref_context,
                    dtype,
                    t,
                    **ckpt_kwargs,
                )
            else:
                x = block(
                    x,
                    e=e0_combined,
                    slice_lens=slice_lens,
                    global_freqs=global_freqs,
                    context_tar=context,
                    context_ref=ref_context,
                    dtype=dtype,
                    t=t,
                )


        # ==========================================
        # 7. Head prediction and output unpacking
        # ==========================================
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            x = self.head(x, e)  # the head is not checkpointed
        else:
            x = self.head(x, e)

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        # Core: split the predicted tensor into the two parts to be generated
        x_ti_out = x[:, L_sv+L_si : L_sv+L_si+L_ti, :]
        x_tv_out = x[:, L_sv+L_si+L_ti :, :]

        # Unpatchify each back to video/image dimensions
        tar_img_out = self.unpatchify([u for u in x_ti_out], grid_ti)
        tar_vid_out = self.unpatchify([u for u in x_tv_out], grid_tv)

        tar_img_out = torch.stack(tar_img_out)
        tar_vid_out = torch.stack(tar_vid_out)

        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
                
        # Return the target video and target image
        return tar_vid_out, tar_img_out

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # # init embeddings

        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        
        # init qwen_project_in if exists
        if self.qwen_project_in is not None:
            for m in self.qwen_project_in.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # init output layer
        nn.init.zeros_(self.head.head.weight)


    @staticmethod
    def _load_state_dict(pretrained_model_path, model_file, model_file_safetensors):
        """Load state_dict; supports .bin and .safetensors formats"""
        if os.path.exists(model_file):
            return torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file
            return load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            print(model_files_safetensors)
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                state_dict.update(_state_dict)
            return state_dict

    @staticmethod
    def _prepare_patch_embeddings(state_dict):
        """Placeholder hook for state-dict preprocessing; currently returns the state_dict unchanged."""
        patch_emb_weight = state_dict.get('patch_embedding.weight')
        patch_emb_bias = state_dict.get('patch_embedding.bias')
        

        
        return state_dict

    @staticmethod
    def _filter_state_dict(state_dict, model_state_dict):
        """Filter state_dict, keeping only keys that match the model"""
        filtered = {}
        for key, value in state_dict.items():
            if key in model_state_dict and model_state_dict[key].size() == value.size():
                filtered[key] = value
            else:
                print(f"################# '{key}' Mismatch (skipped) #################")
        return filtered

    @staticmethod
    def _initialize_missing_parameters(missing_keys, model_state_dict, torch_dtype=None):
        """Initialize missing parameters"""
        initialized_dict = {}
        
        with torch.no_grad():
            for key in missing_keys:
                param_shape = model_state_dict[key].shape
                param_dtype = torch_dtype if torch_dtype is not None else model_state_dict[key].dtype
                if 'weight' in key:
                    if any(norm_type in key for norm_type in ['norm', 'ln_', 'layer_norm', 'group_norm', 'batch_norm']):
                        initialized_dict[key] = torch.ones(param_shape, dtype=param_dtype)
                    elif 'embedding' in key or 'embed' in key:
                        initialized_dict[key] = torch.randn(param_shape, dtype=param_dtype) * 0.02
                    elif 'head' in key or 'output' in key or 'proj_out' in key:
                        initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                    elif len(param_shape) >= 2:
                        initialized_dict[key] = torch.empty(param_shape, dtype=param_dtype)
                        nn.init.xavier_uniform_(initialized_dict[key])
                    else:
                        initialized_dict[key] = torch.randn(param_shape, dtype=param_dtype) * 0.02
                elif 'bias' in key:
                    initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                elif 'running_mean' in key:
                    initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                elif 'running_var' in key:
                    initialized_dict[key] = torch.ones(param_shape, dtype=param_dtype)
                elif 'num_batches_tracked' in key:
                    initialized_dict[key] = torch.zeros(param_shape, dtype=torch.long)
                else:
                    initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                
        return initialized_dict

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16, Debug = False
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        # Load state_dict (only once)
        # state_dict = cls._prepare_patch_embeddings(state_dict)
        state_dict = cls._load_state_dict(pretrained_model_path, model_file, model_file_safetensors)
        
        # state_dict = cls._prepare_patch_embeddings(state_dict)


        if low_cpu_mem_usage:
            try:
                import re
                from diffusers import __version__ as diffusers_version
                if diffusers_version >= "0.33.0":
                    from diffusers.models.model_loading_utils import load_model_dict_into_meta
                else:
                    from diffusers.models.modeling_utils import load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available
                if is_accelerate_available():
                    import accelerate
                
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                filtered_state_dict = cls._filter_state_dict(state_dict, model.state_dict())
                
                model_keys = set(model.state_dict().keys())
                loaded_keys = set(filtered_state_dict.keys())
                missing_keys = model_keys - loaded_keys

                if missing_keys:
                    print(f"################# Missing keys will be initialized: {sorted(missing_keys)} ############### ")
                    initialized_params = cls._initialize_missing_parameters(
                        missing_keys, model.state_dict(), torch_dtype
                    )
                    filtered_state_dict.update(initialized_params)

                if diffusers_version >= "0.33.0":
                    load_model_dict_into_meta(
                        model,
                        filtered_state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )
                else:
                    model._convert_deprecated_attention_blocks(filtered_state_dict)
                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        filtered_state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        print(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )
                
                # Check whether QwenProjectIn weights were loaded successfully
                qwen_proj_keys = [k for k in filtered_state_dict.keys() if 'qwen_project_in' in k]
                if qwen_proj_keys:
                    print("load QwenProjectIn successfully")
                
                return model
            except Exception as e:
                print(
                    f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead."
                )
        
        # Normal mode (or fallback after low_cpu_mem_usage fails)
        model = cls.from_config(config, **transformer_additional_kwargs)
        filtered_state_dict = cls._filter_state_dict(state_dict, model.state_dict())

        m, u = model.load_state_dict(filtered_state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        # print(m)

        # Check whether QwenProjectIn weights were loaded successfully
        qwen_proj_keys = [k for k in filtered_state_dict.keys() if 'qwen_project_in' in k]
        if qwen_proj_keys and Debug:
            print("[from_pretrained] QwenProjectIn weights loaded from checkpoint")
        elif Debug:
            print("[from_pretrained] WARNING: QwenProjectIn weights not found in checkpoint")


        model = model.to(torch_dtype)
        return model

