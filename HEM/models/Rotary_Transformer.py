# coding=utf-8
# Copyright 2022 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections.abc
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    logging,
)
from transformers.models.swin.configuration_swin import SwinConfig

logger = logging.get_logger(__name__)

SWIN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/swin-tiny-patch4-window7-224",
    # See all Swin models at https://huggingface.co/models?filter=swin
]

# ------------------------------ Mona 模塊定義開始 ------------------------------

class MonaOp(nn.Module):
    """
    Mona 核心操作: 使用 1D 卷積處理序列數據。
    """
    def __init__(self, in_features):
        super().__init__()
        # <<< 核心修改：將 Conv2d 改為 Conv1d >>>
        self.conv1 = nn.Conv1d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)
        self.conv2 = nn.Conv1d(in_features, in_features, kernel_size=5, padding=2, groups=in_features)
        self.conv3 = nn.Conv1d(in_features, in_features, kernel_size=7, padding=3, groups=in_features)
        self.projector = nn.Conv1d(in_features, in_features, kernel_size=1)

    def forward(self, x):
        # 輸入 x 的形狀預期為 (B, C, N)，其中 C 是特徵維度，N 是序列長度
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
        identity = x
        x = self.projector(x)
        return identity + x

class Mona(nn.Module):
    """
    完整的 Mona 適配器模塊 (1D 版本)。
    """
    def __init__(self, in_dim, inner_dim=64, dropout_p=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))
        self.project1 = nn.Linear(in_dim, inner_dim)
        self.project2 = nn.Linear(inner_dim, in_dim)
        self.adapter_conv = MonaOp(inner_dim) # 調用 1D 版本的 MonaOp
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # 輸入 x 的形狀是 (B, N, C)
        identity = x
        
        x = self.norm(x) * self.gamma + x * self.gammax
        
        project1 = self.project1(x)
        
        # <<< 核心修改：不再需要 H, W，直接交換維度以適配 Conv1d >>>
        # (B, N, C) -> (B, C, N)
        project1_permuted = project1.permute(0, 2, 1)
        
        # 進行 1D 卷積
        project1_conv = self.adapter_conv(project1_permuted)
        
        # 換回原始維度順序
        # (B, C, N) -> (B, N, C)
        project1_restored = project1_conv.permute(0, 2, 1)

        nonlinear = F.gelu(project1_restored)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2
# ------------------------------ Mona 模塊定義結束 ------------------------------

@dataclass
class SwinEncoderOutput(ModelOutput):
    """
    Swin encoder's outputs, with potential hidden states and attentions.
    """
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SwinModelOutput(ModelOutput):
    """
    Swin model's outputs that also contains a pooling of the last hidden states.
    """
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SwinMaskedImageModelingOutput(ModelOutput):
    """
    Swin masked image model outputs.
    """
    loss: Optional[torch.FloatTensor] = None
    reconstruction: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

    @property
    def logits(self):
        warnings.warn(
            "logits attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the reconstruction attribute to retrieve the final output instead.",
            FutureWarning,
        )
        return self.reconstruction


@dataclass
class SwinImageClassifierOutput(ModelOutput):
    """
    Swin outputs for image classification.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    windows = input_feature.view(-1, window_size, input_feature.shape[2])
    return windows


def window_reverse(windows, size):
    """
    Merges windows to produce higher resolution features.
    """
    windows = windows.view(-1, size, windows.shape[2])
    return windows

class SwinPatchMerging(nn.Module):
    """
    Patch Merging Layer.
    """
    def __init__(self, input_resolution: int, dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def maybe_pad(self, input_feature, size):
        should_pad = size % 2 == 1
        if should_pad:
            pad_values = (0, 0, 0, size % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)
        return input_feature

    def forward(self, input_feature: torch.Tensor, input_size: int) -> torch.Tensor:
        batch_size, dim, num_channels = input_feature.shape
        input_feature = self.maybe_pad(input_feature, input_size)
        input_feature_0 = input_feature[:, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, :]
        input_feature = torch.cat([input_feature_0, input_feature_1], -1)
        input_feature = input_feature.view(batch_size, -1, 2 * num_channels)
        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)
        return input_feature


def drop_path(input, drop_prob=0.0, training=False, scale_by_keep=True):
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()
    output = input.div(keep_prob) * random_tensor
    return output


class SwinDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)

class EfficientRoPE(nn.Module):
    def __init__(self, base=10000):
        super().__init__()
        self.base = base
        
    def forward(self, x):
        _, seq_len, _, head_dim = x.shape
        
        # 生成位置
        position = torch.arange(seq_len, dtype=torch.float, device=x.device)
        
        # 计算频率
        dim_range = torch.arange(0, head_dim, 2, dtype=torch.float, device=x.device)
        inv_freq = 1.0 / (self.base ** (dim_range / head_dim))
        
        # 计算正弦和余弦
        sinusoid = torch.einsum('i,j->ij', position, inv_freq)
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)
        
        # 重塑输入
        x1 = x[..., 0::2]  # 偶数索引
        x2 = x[..., 1::2]  # 奇数索引
        
        # 应用旋转
        rotated_x1 = x1 * cos.unsqueeze(0).unsqueeze(2) - x2 * sin.unsqueeze(0).unsqueeze(2)
        rotated_x2 = x1 * sin.unsqueeze(0).unsqueeze(2) + x2 * cos.unsqueeze(0).unsqueeze(2)
        
        # 交错合并结果
        output = torch.stack([rotated_x1, rotated_x2], dim=-1)
        output = output.flatten(-2, -1)
        
        return output

class Attention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, cross):
        super().__init__()
        # --- 原始 __init__ 內容 (保持不變) ---
        if dim % num_heads != 0:
            raise ValueError(f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})")
        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = window_size
        self.cross = cross
        self.relative_position_bias_table = nn.Parameter(torch.zeros(2 * self.window_size - 1, num_heads))
        self.relative_position_index = torch.arange(window_size).flip(0).repeat(window_size, 1)
        for i in range(self.relative_position_index.shape[0]):
            self.relative_position_index[i] -= i
        self.relative_position_index += window_size - 1
        self.relative_position_index = self.relative_position_index.flip(0)
        self.rope = EfficientRoPE()
        # self.register_buffer("relative_position_index", relative_position_index)
        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = False,
                query: Optional[torch.Tensor] = None,
                # --- 新增的 code 輸入 ---
                code: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        
        # 你的原始程式碼中，dim 是序列長度，num_channels 是特徵維度
        batch_size, seq_len, num_channels = hidden_states.shape
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)


        # --- 以下是原始的計算流程 ---
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        query_layer = self.rope(query_layer)
        key_layer = self.rope(key_layer)

        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)

        attention_scores = torch.einsum('bnqd,bnkd->bnqk', query_layer, key_layer)/ math.sqrt(self.attention_head_size)

        # --- 位置偏置計算：在原始計算基礎上增加 ---
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        final_bias = relative_position_bias.unsqueeze(0)

        # --- 以下是原始的計算流程 (保持不變) ---
        if attention_mask is not None:
            mask_shape = attention_mask.shape[0]
            # 你的原始碼中 dim 是序列長度
            attention_scores = attention_scores.view(batch_size // mask_shape, mask_shape, self.num_attention_heads, seq_len, seq_len)
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, seq_len, seq_len)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1) + final_bias

        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.einsum('bnqk,bnkd->bnqd', attention_probs.to(value_layer.dtype), value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class SwinSelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SwinAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, cross):
        super().__init__()
        self.self = Attention(config, dim, num_heads, window_size, cross)
        self.output = SwinSelfOutput(config, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None, head_mask: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = False, query: torch.Tensor = None,  code: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions, query, code = code)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class SwinIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SwinOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

# #################################################################################
# ## >>>>>>>>>> MODIFICATION START: SwinLayer has been modified <<<<<<<<<<< ##
# #################################################################################
class SwinLayer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0, cross=False):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.shift_size = shift_size
        self.cross = cross
        self.dim = dim
        self.window_size = config.window_size
        self.input_resolution = input_resolution
        
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attention = SwinAttention(config, dim, num_heads, window_size=self.window_size, cross=cross)
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.intermediate = SwinIntermediate(config, dim)
        self.output = SwinOutput(config, dim)
        
        # <<< Mona Integration >>>
        # 實例化兩個 Mona 模塊
        self.mona1 = Mona(in_dim=dim)
        self.mona2 = Mona(in_dim=dim)
        # <<< End Mona Integration >>>

    def set_shift_and_window_size(self, input_resolution):
        if input_resolution <= self.window_size:
            self.shift_size = 0
            self.window_size = input_resolution

    def get_attn_mask(self, context_size, dtype, device):
        if self.shift_size > 0:
            img_mask = torch.zeros((1, context_size, 1), dtype=dtype).to(device)
            context_size_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for context_size_slice in context_size_slices:
                img_mask[:, context_size_slice, :] = count
                count += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states, input_size):
        pad = (self.window_size - input_size % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def proc_hidden_stats(self, hidden_states: torch.Tensor, input_size: int, batch_size: int, channels: int):
        hidden_states = hidden_states.contiguous()
        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = hidden_states.view(batch_size, input_size, channels)
        hidden_states, pad_values = self.maybe_pad(hidden_states, input_size)
        _, size_pad, _ = hidden_states.shape
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size,), dims=(1,))
        else:
            shifted_hidden_states = hidden_states
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size, channels)
        return hidden_states_windows, size_pad, pad_values

    def proc_hidden_stats_code(self, hidden_states: torch.Tensor, input_size: int, batch_size: int, channels: int):
        hidden_states = hidden_states.contiguous()
        hidden_states = hidden_states.view(batch_size, input_size, channels)
        hidden_states, pad_values = self.maybe_pad(hidden_states, input_size)
        _, size_pad, _ = hidden_states.shape
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size,), dims=(1,))
        else:
            shifted_hidden_states = hidden_states
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size, channels)
        return hidden_states_windows, size_pad, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_size: int,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        query: torch.Tensor = None,
        code: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        
        shortcut = hidden_states

        batch_size, _, channels = hidden_states.size()
        hidden_states_windows, pad_size, pad_values = self.proc_hidden_stats(hidden_states, input_size, batch_size, channels)

        # ▼▼▼ 新增 code 的窗口化邏輯 ▼▼▼
        code_windows = None
        if code is not None:
            # code 的形狀是 (B, N)，需要 unsqueeze 成 (B, N, 1) 來使用 proc_hidden_stats
            # 注意：這裡我們借用 proc_hidden_stats 的 padding 和 roll 邏輯，但只關心其窗口劃分結果
            code_proc_result = self.proc_hidden_stats_code(code.unsqueeze(-1), input_size, batch_size, 1)[0]
            code_windows = code_proc_result.squeeze(-1) # 恢復形狀為 (num_windows, window_size)
        # ▲▲▲ 新增邏輯結束 ▲▲▲

        if self.cross:
            query_windows = self.proc_hidden_stats(query, input_size, batch_size, channels)[0]

        attn_mask = self.get_attn_mask(pad_size, dtype=hidden_states.dtype, device=hidden_states.device)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        attention_outputs = self.attention(
            hidden_states_windows,
            attn_mask,
            head_mask,
            output_attentions=output_attentions,
            query=query_windows if self.cross else None,
            code=code_windows, 
        )

        attention_output = attention_outputs[0]
        attention_windows = attention_output.view(-1, self.window_size, channels)
        shifted_windows = window_reverse(attention_windows, pad_size)
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size,), dims=(1,))
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0
        if was_padded:
            attention_windows = attention_windows[:, :input_size, :].contiguous()

        attention_windows = attention_windows.view(batch_size, input_size, channels)

        hidden_states = shortcut + attention_windows
        
        # <<< Mona Integration >>>
        # 在自注意力之後插入第一個 Mona 模塊
        hidden_states = self.mona1(hidden_states)
        # <<< End Mona Integration >>>
        
        # FFN (Feed-Forward Network) 部分
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output)
        
        # FFN 的殘差連接
        layer_output = hidden_states + layer_output

        # <<< Mona Integration >>>
        # 在 FFN 之後插入第二個 Mona 模塊
        layer_output = self.mona2(layer_output)
        # <<< End Mona Integration >>>
        
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs
# ###############################################################################
# ## >>>>>>>>>> MODIFICATION END: SwinLayer has been modified <<<<<<<<<<< ##
# ###############################################################################

class SwinStage(nn.Module):
    def __init__(self, config, dim, input_resolution, depth, num_heads, downsample, cross):
        super().__init__()
        self.config = config
        self.dim = dim
        self.cross = cross
        self.blocks = nn.ModuleList(
            [
                SwinLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                    cross=cross,
                )
                for i in range(depth)
            ]
        )
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None
        self.pointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_size: int,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        query: torch.Tensor = None,
        code: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, input_size, layer_head_mask, output_attentions, query,code = code)
            hidden_states = layer_outputs[0]
        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            size_downsampled = (input_size + 1) // 2
            output_dimensions = (input_size, size_downsampled)
            hidden_states = self.downsample(hidden_states_before_downsampling, input_size)

            # ▼▼▼ 新增 code 的降採樣邏輯 ▼▼▼
            if code is not None:
                # SwinPatchMerging 的核心是间隔採樣，我們對 code 做同樣操作
                # (B, N) -> (B, N/2)
                code = code[:, 0::2] 
            # ▲▲▲ 新增邏輯結束 ▲▲▲

            if self.cross:
                query = self.downsample(query, input_size)
        else:
            output_dimensions = (input_size, input_size)
        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions, query, code)
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class SwinEncoder(nn.Module):
    def __init__(self, config, context_size, cross):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        self.layers = nn.ModuleList(
            [
                SwinStage(
                    config=config,
                    dim=config.embed_dim,
                    input_resolution=context_size // 2**i_layer,
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    downsample=SwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                    cross=cross,
                )
                for i_layer in range(self.num_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: int,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        query: torch.Tensor = None,
        code: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SwinEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            reshaped_hidden_state = hidden_states.view(batch_size, input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 2, 1)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)
        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module), hidden_states, input_dimensions, layer_head_mask
                )
            else:
                layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions, query, code)
            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]
            query = layer_outputs[3]
            code = layer_outputs[4]

            input_dimensions = output_dimensions[-1]
            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                reshaped_hidden_state = hidden_states_before_downsampling.view(
                    batch_size, output_dimensions[0], hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 2, 1)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                reshaped_hidden_state = hidden_states.view(batch_size, input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 2, 1)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            if output_attentions:
                all_self_attentions += layer_outputs[3:]
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return SwinEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )