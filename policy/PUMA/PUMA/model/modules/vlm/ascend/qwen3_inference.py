"""Ascend-only Qwen3-VL inference request planning and adapter installation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from itertools import accumulate
from math import isqrt
from typing import Callable, Optional

import torch
from transformers import __version__ as _TRANSFORMERS_VERSION
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3VLModel,
    Qwen3VLTextModel,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionAttention,
    Qwen3VLVisionModel,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)

from PUMA.model.modules.vlm.ascend.patch_embed import linearize_qwen_vision_patch_embed


_QWEN3_TEXT_ROPE_OPTIMIZED_TRANSFORMERS_VERSION = "4.57.0"

@dataclass(frozen=True)
class _Qwen3VisionPlan:
    grid_signature: tuple[torch.dtype, tuple[tuple[int, int, int], ...]]
    num_position_embeddings: int
    bound_grid_thw: torch.Tensor
    patch_split_lengths: tuple[int, ...]
    merged_split_lengths: tuple[int, ...]
    cu_seqlens: torch.Tensor
    pos_embed_indices: torch.Tensor
    pos_embed_weights: torch.Tensor
    rotary_pos_ids: torch.Tensor
    uniform_length: Optional[int]
    group_lengths: tuple[int, ...]
    group_spans: tuple[tuple[tuple[int, int], ...], ...]

    def with_cu_seqlens(self, cu_seqlens):
        return replace(self, cu_seqlens=cu_seqlens)


@dataclass(frozen=True)
class _Qwen3InferencePlan:
    image: Optional[_Qwen3VisionPlan]
    video: Optional[_Qwen3VisionPlan]
    omit_attention_mask: bool = False
    bound_input_ids: Optional[torch.Tensor] = None
    image_token_indices: Optional[torch.Tensor] = None
    video_token_indices: Optional[torch.Tensor] = None
    visual_token_indices: Optional[torch.Tensor] = None


_QWEN3_NPU_POSITION_TOKEN_COUNT = ContextVar("qwen3_npu_position_token_count", default=None)
_QWEN3_INFERENCE_PLAN = ContextVar("qwen3_inference_plan", default=None)
_QWEN3_VISION_PLAN = ContextVar("qwen3_vision_plan", default=None)
_QWEN3_GRID_DTYPES = frozenset(
    (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
)


def _validate_qwen3_cpu_grid_thw(grid_thw, name="grid_thw"):
    if grid_thw.device.type != "cpu":
        raise ValueError(f"Qwen3 {name} control metadata must be on CPU")
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError(f"Qwen3 {name} must have shape [items, 3]")
    if grid_thw.dtype not in _QWEN3_GRID_DTYPES:
        raise ValueError(f"Qwen3 {name} must have an integral dtype")


def _qwen3_num_grid_per_side(num_position_embeddings):
    if type(num_position_embeddings) is not int or num_position_embeddings <= 0:
        raise ValueError("Qwen3 num_position_embeddings must be a positive integer perfect square")
    num_grid_per_side = isqrt(num_position_embeddings)
    if num_grid_per_side * num_grid_per_side != num_position_embeddings:
        raise ValueError("Qwen3 num_position_embeddings must be a positive integer perfect square")
    return num_grid_per_side


def _build_qwen3_vision_plan(grid_thw, spatial_merge_size, num_position_embeddings):
    _validate_qwen3_cpu_grid_thw(grid_thw)
    merge_size = int(spatial_merge_size)
    if merge_size <= 0:
        raise ValueError("Qwen3 spatial_merge_size must be positive")
    num_grid_per_side = _qwen3_num_grid_per_side(num_position_embeddings)

    rows = tuple(tuple(int(value) for value in row) for row in grid_thw.tolist())
    patch_split_lengths = []
    merged_split_lengths = []
    spans = []
    index_chunks = [[], [], [], []]
    weight_chunks = [[], [], [], []]
    rotary_pos_ids = []
    offset = 0
    merge_unit = merge_size * merge_size
    for temporal, height, width in rows:
        if temporal <= 0 or height <= 0 or width <= 0:
            raise ValueError("Qwen3 grid dimensions must be positive")
        if height % merge_size or width % merge_size:
            raise ValueError("Qwen3 spatial grid dimensions must be divisible by spatial_merge_size")
        frame_length = height * width
        for _ in range(temporal):
            patch_split_lengths.append(frame_length)
            spans.append((offset, offset + frame_length))
            offset += frame_length
        merged_split_lengths.append(temporal * frame_length // merge_unit)

        height_positions = torch.linspace(0, num_grid_per_side - 1, height)
        width_positions = torch.linspace(0, num_grid_per_side - 1, width)
        height_floor = height_positions.int()
        width_floor = width_positions.int()
        height_ceil = (height_floor + 1).clip(max=num_grid_per_side - 1)
        width_ceil = (width_floor + 1).clip(max=num_grid_per_side - 1)
        height_delta = height_positions - height_floor
        width_delta = width_positions - width_floor
        height_floor_base = height_floor * num_grid_per_side
        height_ceil_base = height_ceil * num_grid_per_side
        indices = (
            (height_floor_base[:, None] + width_floor[None, :]).flatten(),
            (height_floor_base[:, None] + width_ceil[None, :]).flatten(),
            (height_ceil_base[:, None] + width_floor[None, :]).flatten(),
            (height_ceil_base[:, None] + width_ceil[None, :]).flatten(),
        )
        weights = (
            ((1 - height_delta)[:, None] * (1 - width_delta)[None, :]).flatten(),
            ((1 - height_delta)[:, None] * width_delta[None, :]).flatten(),
            (height_delta[:, None] * (1 - width_delta)[None, :]).flatten(),
            (height_delta[:, None] * width_delta[None, :]).flatten(),
        )

        # Match the processor's t, block-h, block-w, inner-h, inner-w patch order.
        merge_order = (
            torch.arange(frame_length)
            .reshape(height // merge_size, merge_size, width // merge_size, merge_size)
            .permute(0, 2, 1, 3)
            .flatten()
        )
        for index in range(4):
            index_chunks[index].append(indices[index].index_select(0, merge_order).repeat(temporal))
            weight_chunks[index].append(weights[index].index_select(0, merge_order).repeat(temporal))

        row_ids = torch.arange(height)[:, None].expand(height, width).flatten()
        column_ids = torch.arange(width)[None, :].expand(height, width).flatten()
        item_pos_ids = torch.stack((row_ids, column_ids), dim=1).index_select(0, merge_order)
        rotary_pos_ids.append(item_pos_ids.repeat(temporal, 1))

    grouped_spans = {}
    for length, span in zip(patch_split_lengths, spans):
        grouped_spans.setdefault(length, []).append(span)
    group_lengths = tuple(grouped_spans)
    group_spans = tuple(tuple(grouped_spans[length]) for length in group_lengths)
    uniform_length = group_lengths[0] if len(group_lengths) == 1 else None
    cu_seqlens = torch.tensor(
        (0, *accumulate(patch_split_lengths)),
        dtype=torch.int32,
    )
    pos_embed_indices = torch.stack(
        tuple(torch.cat(chunks).to(dtype=torch.long) for chunks in index_chunks)
    )
    pos_embed_weights = torch.stack(tuple(torch.cat(chunks) for chunks in weight_chunks))
    return _Qwen3VisionPlan(
        grid_signature=(grid_thw.dtype, rows),
        num_position_embeddings=num_position_embeddings,
        bound_grid_thw=grid_thw,
        patch_split_lengths=tuple(patch_split_lengths),
        merged_split_lengths=tuple(merged_split_lengths),
        cu_seqlens=cu_seqlens,
        pos_embed_indices=pos_embed_indices,
        pos_embed_weights=pos_embed_weights,
        rotary_pos_ids=torch.cat(rotary_pos_ids),
        uniform_length=uniform_length,
        group_lengths=group_lengths,
        group_spans=group_spans,
    )


def _build_qwen3_inference_plan(
    *,
    image_grid_thw=None,
    video_grid_thw=None,
    spatial_merge_size,
    num_position_embeddings,
    omit_attention_mask=False,
):
    image_plan = (
        _build_qwen3_vision_plan(image_grid_thw, spatial_merge_size, num_position_embeddings)
        if image_grid_thw is not None
        else None
    )
    video_plan = (
        _build_qwen3_vision_plan(video_grid_thw, spatial_merge_size, num_position_embeddings)
        if video_grid_thw is not None
        else None
    )
    return _Qwen3InferencePlan(
        image=image_plan,
        video=video_plan,
        omit_attention_mask=bool(omit_attention_mask),
    )


def _build_qwen3_placeholder_indices(
    input_ids,
    *,
    image_token_id,
    video_token_id,
    image_plan,
    video_plan,
):
    if input_ids.device.type != "cpu" or input_ids.ndim != 2:
        raise ValueError("Qwen3 placeholder control input_ids must be a CPU [batch, sequence] tensor")

    image_mask = input_ids.eq(int(image_token_id))
    video_mask = input_ids.eq(int(video_token_id))
    image_indices = image_mask.flatten().nonzero(as_tuple=False).flatten()
    video_indices = video_mask.flatten().nonzero(as_tuple=False).flatten()
    visual_indices = (image_mask | video_mask).flatten().nonzero(as_tuple=False).flatten()

    for label, indices, plan in (
        ("image", image_indices, image_plan),
        ("video", video_indices, video_plan),
    ):
        expected = 0 if plan is None else sum(plan.merged_split_lengths)
        if indices.shape[0] != expected:
            raise ValueError(
                f"Qwen3 {label} placeholder count {indices.shape[0]} does not match "
                f"planned visual feature count {expected}"
            )
    return image_indices, video_indices, visual_indices


def _qwen3_npu_deepstack_process(hidden_states, visual_embeds, visual_token_indices):
    if hidden_states.ndim != 3 or visual_embeds.ndim != 2:
        raise RuntimeError("Qwen3 planned deepstack tensors have invalid ranks")
    if hidden_states.shape[-1] != visual_embeds.shape[-1]:
        raise RuntimeError("Qwen3 planned deepstack hidden dimensions do not match")
    if visual_token_indices.ndim != 1 or visual_token_indices.dtype != torch.long:
        raise RuntimeError("Qwen3 planned deepstack indices must be a one-dimensional long tensor")
    if visual_token_indices.shape[0] != visual_embeds.shape[0]:
        raise RuntimeError("Qwen3 planned deepstack token count does not match visual embeddings")
    if not (
        hidden_states.device == visual_embeds.device == visual_token_indices.device
    ):
        raise RuntimeError("Qwen3 planned deepstack tensors must share one device")

    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    selected = flat_hidden.index_select(0, visual_token_indices) + visual_embeds
    return torch.index_copy(flat_hidden, 0, visual_token_indices, selected).reshape_as(hidden_states)


def _bind_qwen3_vision_plan_grid(plan, grid_thw):
    if plan is None:
        return None
    grid_dtype, grid_rows = plan.grid_signature
    if grid_thw.dtype != grid_dtype or tuple(grid_thw.shape) != (len(grid_rows), 3):
        raise ValueError("Qwen3 transferred grid metadata does not match plan grid signature")
    return replace(
        plan,
        bound_grid_thw=grid_thw,
    )


@contextmanager
def _activate_qwen3_inference_plan(plan):
    token = _QWEN3_INFERENCE_PLAN.set(plan)
    try:
        yield plan
    finally:
        _QWEN3_INFERENCE_PLAN.reset(token)


def _get_active_qwen3_inference_plan():
    return _QWEN3_INFERENCE_PLAN.get()


def _build_qwen3_mrope_source_dims(frequency_count, mrope_section):
    if type(frequency_count) is not int or frequency_count <= 0:
        raise ValueError("Qwen3 text rotary frequency count must be a positive integer")
    if len(mrope_section) != 3 or any(type(value) is not int or value < 0 for value in mrope_section):
        raise ValueError("Qwen3 mrope_section must contain three non-negative integers")

    source_dims = torch.zeros(frequency_count, dtype=torch.long)
    for source_dim, offset in enumerate((1, 2), start=1):
        source_dims[offset : mrope_section[source_dim] * 3 : 3] = source_dim
    return source_dims


def _qwen3_npu_text_rotary_frequencies(position_ids, inv_freq, source_dims):
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError("Qwen3 text position_ids must have shape [3, batch, sequence]")
    if source_dims.ndim != 1 or source_dims.shape[0] != inv_freq.shape[0]:
        raise ValueError("Qwen3 text rotary source plan does not match inv_freq")

    positions_by_frequency = position_ids.permute(1, 2, 0).index_select(-1, source_dims)
    return positions_by_frequency.float() * inv_freq.float()[None, None, :]


@contextmanager
def _activate_qwen3_vision_plan(plan):
    token = _QWEN3_VISION_PLAN.set(plan)
    try:
        yield plan
    finally:
        _QWEN3_VISION_PLAN.reset(token)


def _qwen3_attention_interface(attn_implementation):
    if attn_implementation == "eager":
        return eager_attention_forward
    return ALL_ATTENTION_FUNCTIONS[attn_implementation]


def _run_qwen3_grouped_vision_attention(
    module,
    query_states,
    key_states,
    value_states,
    plan,
    attention_interface: Callable,
    **kwargs,
):
    outputs_by_start = {}
    dropout = 0.0 if not module.training else module.attention_dropout
    num_heads = query_states.shape[1]
    head_dim = query_states.shape[2]

    for length, spans in zip(plan.group_lengths, plan.group_spans):
        batch_size = len(spans)
        if plan.uniform_length is not None:
            query = query_states.reshape(batch_size, length, num_heads, head_dim)
            key = key_states.reshape(batch_size, length, num_heads, head_dim)
            value = value_states.reshape(batch_size, length, num_heads, head_dim)
        else:
            query = torch.cat([query_states[start:end] for start, end in spans], dim=0).reshape(
                batch_size, length, num_heads, head_dim
            )
            key = torch.cat([key_states[start:end] for start, end in spans], dim=0).reshape(
                batch_size, length, num_heads, head_dim
            )
            value = torch.cat([value_states[start:end] for start, end in spans], dim=0).reshape(
                batch_size, length, num_heads, head_dim
            )

        attention_output = attention_interface(
            module,
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
            attention_mask=None,
            scaling=module.scaling,
            dropout=dropout,
            is_causal=False,
            **kwargs,
        )[0]
        for index, (start, _) in enumerate(spans):
            outputs_by_start[start] = attention_output[index]

    return torch.cat([outputs_by_start[start] for start in sorted(outputs_by_start)], dim=0)


def _qwen3_merged_token_coordinates(grid_thw, token_count, merge_size):
    token_indices = torch.arange(token_count, dtype=grid_thw.dtype, device=grid_thw.device)
    tokens_per_grid = grid_thw.prod(dim=1)
    grid_ends = tokens_per_grid.cumsum(dim=0)
    grid_starts = grid_ends - tokens_per_grid

    # The vision processor stores patches in t, block-h, block-w, inner-h, inner-w order.
    grid_indices = (token_indices[:, None] >= grid_ends[:-1][None, :]).sum(dim=1)
    selected_grid = grid_thw.index_select(0, grid_indices)
    local_indices = token_indices - grid_starts.index_select(0, grid_indices)

    heights = selected_grid[:, 1]
    widths = selected_grid[:, 2]
    spatial_indices = local_indices.remainder(heights * widths)
    inner_width = spatial_indices.remainder(merge_size)
    remaining = torch.div(spatial_indices, merge_size, rounding_mode="floor")
    inner_height = remaining.remainder(merge_size)
    remaining = torch.div(remaining, merge_size, rounding_mode="floor")
    block_widths = torch.div(widths, merge_size, rounding_mode="floor")
    block_width = remaining.remainder(block_widths)
    block_height = torch.div(remaining, block_widths, rounding_mode="floor")

    rows = block_height * merge_size + inner_height
    columns = block_width * merge_size + inner_width
    return rows, columns, heights, widths


def _qwen3_interpolated_pos_embed(vision, grid_thw, token_count):
    rows, columns, heights, widths = _qwen3_merged_token_coordinates(
        grid_thw,
        token_count,
        vision.spatial_merge_size,
    )
    max_index = vision.num_grid_per_side - 1
    rows_float = rows.to(dtype=torch.float32)
    columns_float = columns.to(dtype=torch.float32)
    height_denominator = (heights - 1).clamp_min(1).to(dtype=torch.float32)
    width_denominator = (widths - 1).clamp_min(1).to(dtype=torch.float32)
    interpolated_rows = rows_float * max_index / height_denominator
    interpolated_columns = columns_float * max_index / width_denominator

    row_floor = interpolated_rows.to(dtype=torch.long)
    column_floor = interpolated_columns.to(dtype=torch.long)
    row_ceil = (row_floor + 1).clamp_max(max_index)
    column_ceil = (column_floor + 1).clamp_max(max_index)
    row_delta = interpolated_rows - row_floor.to(dtype=interpolated_rows.dtype)
    column_delta = interpolated_columns - column_floor.to(dtype=interpolated_columns.dtype)

    row_floor_base = row_floor * vision.num_grid_per_side
    row_ceil_base = row_ceil * vision.num_grid_per_side
    indices = torch.stack(
        (
            row_floor_base + column_floor,
            row_floor_base + column_ceil,
            row_ceil_base + column_floor,
            row_ceil_base + column_ceil,
        )
    )
    weights = torch.stack(
        (
            (1 - row_delta) * (1 - column_delta),
            (1 - row_delta) * column_delta,
            row_delta * (1 - column_delta),
            row_delta * column_delta,
        )
    ).to(dtype=vision.pos_embed.weight.dtype)
    embeddings = vision.pos_embed(indices) * weights[:, :, None]
    return embeddings[0] + embeddings[1] + embeddings[2] + embeddings[3]


def _qwen3_rotary_pos_embed(vision, grid_thw, token_count):
    rows, columns, _, _ = _qwen3_merged_token_coordinates(
        grid_thw,
        token_count,
        vision.spatial_merge_size,
    )
    inv_freq = vision.rotary_pos_emb.inv_freq
    row_frequencies = rows[:, None].to(dtype=inv_freq.dtype) * inv_freq[None, :]
    column_frequencies = columns[:, None].to(dtype=inv_freq.dtype) * inv_freq[None, :]
    return torch.stack((row_frequencies, column_frequencies), dim=1).flatten(1)


def _qwen3_plan_interpolated_pos_embed(vision, plan):
    weights = plan.pos_embed_weights.to(dtype=vision.pos_embed.weight.dtype)
    embeddings = vision.pos_embed(plan.pos_embed_indices)
    return (
        embeddings[0] * weights[0, :, None]
        + embeddings[1] * weights[1, :, None]
        + embeddings[2] * weights[2, :, None]
        + embeddings[3] * weights[3, :, None]
    )


def _qwen3_plan_rotary_pos_embed(vision, plan):
    inv_freq = vision.rotary_pos_emb.inv_freq
    return (
        plan.rotary_pos_ids[:, :, None].to(dtype=inv_freq.dtype)
        * inv_freq[None, None, :]
    ).flatten(1)


class _PUMAQwen3NPUVisionAttention(Qwen3VLVisionAttention):
    def forward(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs,
    ):
        if hidden_states.device.type != "npu":
            return super().forward(
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        plan = _QWEN3_VISION_PLAN.get()
        if plan is None:
            return super().forward(
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        attn_implementation = getattr(self.config, "_attn_implementation", "eager")
        if attn_implementation not in ("eager", "sdpa"):
            raise RuntimeError("Qwen3 NPU request-plan attention supports only eager and sdpa")

        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states,
            key_states,
            cos,
            sin,
        )
        attention_output = _run_qwen3_grouped_vision_attention(
            self,
            query_states,
            key_states,
            value_states,
            plan,
            _qwen3_attention_interface(attn_implementation),
            **kwargs,
        )
        return self.proj(attention_output.reshape(seq_length, -1).contiguous())


class _PUMAQwen3NPUTextRotaryEmbedding(Qwen3VLTextRotaryEmbedding):
    @torch.no_grad()
    def forward(self, x, position_ids):
        if x.device.type != "npu" or self.training or _QWEN3_INFERENCE_PLAN.get() is None:
            return super().forward(x, position_ids)
        if (
            _TRANSFORMERS_VERSION != _QWEN3_TEXT_ROPE_OPTIMIZED_TRANSFORMERS_VERSION
            or self.rope_type != "default"
        ):
            return super().forward(x, position_ids)

        source_dims = self._puma_mrope_source_dims
        if source_dims.device != position_ids.device or self.inv_freq.device != position_ids.device:
            raise RuntimeError("Qwen3 NPU text rotary plan tensors are not resident with position_ids")
        with torch.autocast(device_type="npu", enabled=False):
            freqs = _qwen3_npu_text_rotary_frequencies(
                position_ids,
                self.inv_freq,
                source_dims,
            )
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class _PUMAQwen3NPUPositionVisionModel(Qwen3VLVisionModel):
    def _validate_request_plan(self, plan, grid_thw):
        if self.pos_embed.num_embeddings != plan.num_position_embeddings:
            raise RuntimeError("Qwen3 vision request plan position table does not match visual")
        if grid_thw is not plan.bound_grid_thw:
            raise RuntimeError("Qwen3 vision request plan grid signature does not match grid_thw")

    def _position_token_count(self, token_count):
        if token_count is not None:
            return token_count
        active_token_count = _QWEN3_NPU_POSITION_TOKEN_COUNT.get()
        if active_token_count is None:
            raise RuntimeError("NPU position encoding must run inside Qwen3 vision forward")
        return active_token_count

    def fast_pos_embed_interpolate(self, grid_thw, token_count=None):
        if grid_thw.device.type != "npu":
            return super().fast_pos_embed_interpolate(grid_thw)
        plan = _QWEN3_VISION_PLAN.get()
        if plan is not None:
            self._validate_request_plan(plan, grid_thw)
            expected_token_count = self._position_token_count(token_count)
            if plan.pos_embed_indices.shape[1] != expected_token_count:
                raise RuntimeError("Qwen3 vision request plan does not match position token shape")
            return _qwen3_plan_interpolated_pos_embed(self, plan)
        return _qwen3_interpolated_pos_embed(
            self,
            grid_thw,
            self._position_token_count(token_count),
        )

    def rot_pos_emb(self, grid_thw, token_count=None):
        if grid_thw.device.type != "npu":
            return super().rot_pos_emb(grid_thw)
        plan = _QWEN3_VISION_PLAN.get()
        if plan is not None:
            self._validate_request_plan(plan, grid_thw)
            expected_token_count = self._position_token_count(token_count)
            if plan.rotary_pos_ids.shape[0] != expected_token_count:
                raise RuntimeError("Qwen3 vision request plan does not match rotary token shape")
            return _qwen3_plan_rotary_pos_embed(self, plan)
        return _qwen3_rotary_pos_embed(
            self,
            grid_thw,
            self._position_token_count(token_count),
        )

    def forward(self, hidden_states, grid_thw, **kwargs):
        if grid_thw.device.type != "npu":
            return super().forward(hidden_states, grid_thw, **kwargs)
        context_token = _QWEN3_NPU_POSITION_TOKEN_COUNT.set(hidden_states.shape[0])
        try:
            plan = _QWEN3_VISION_PLAN.get()
            if plan is None and len(self.blocks):
                raise RuntimeError("NPU vision forward requires a CPU-built request plan")
            if plan is not None:
                self._validate_request_plan(plan, grid_thw)
            hidden_states = self.patch_embed(hidden_states)
            if plan is not None and sum(plan.patch_split_lengths) != hidden_states.shape[0]:
                raise RuntimeError("Qwen3 vision request plan does not match patch token shape")

            hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw)
            rotary_pos_emb = self.rot_pos_emb(grid_thw)
            seq_len, _ = hidden_states.size()
            hidden_states = hidden_states.reshape(seq_len, -1)
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())
            cu_seqlens = (
                plan.cu_seqlens
                if plan is not None
                else torch.empty((0,), dtype=torch.int32, device=hidden_states.device)
            )

            deepstack_feature_lists = []
            for layer_num, block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                if layer_num in self.deepstack_visual_indexes:
                    merger_index = self.deepstack_visual_indexes.index(layer_num)
                    deepstack_feature_lists.append(
                        self.deepstack_merger_list[merger_index](hidden_states)
                    )
            hidden_states = self.merger(hidden_states)
            return hidden_states, deepstack_feature_lists
        finally:
            _QWEN3_NPU_POSITION_TOKEN_COUNT.reset(context_token)


class _PUMAQwen3NPURequestPlanVisionModel(_PUMAQwen3NPUPositionVisionModel):
    def fast_pos_embed_interpolate(self, grid_thw, token_count=None):
        if grid_thw.device.type == "npu" and _QWEN3_VISION_PLAN.get() is None:
            return Qwen3VLVisionModel.fast_pos_embed_interpolate(self, grid_thw)
        return super().fast_pos_embed_interpolate(grid_thw, token_count=token_count)

    def rot_pos_emb(self, grid_thw, token_count=None):
        if grid_thw.device.type == "npu" and _QWEN3_VISION_PLAN.get() is None:
            return Qwen3VLVisionModel.rot_pos_emb(self, grid_thw)
        return super().rot_pos_emb(grid_thw, token_count=token_count)

    def forward(self, hidden_states, grid_thw, **kwargs):
        if grid_thw.device.type == "npu" and _QWEN3_VISION_PLAN.get() is None:
            return Qwen3VLVisionModel.forward(self, hidden_states, grid_thw, **kwargs)
        return super().forward(hidden_states, grid_thw, **kwargs)


class _PUMAQwen3NPUTextModel(Qwen3VLTextModel):
    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        request_plan = _QWEN3_INFERENCE_PLAN.get()
        if hidden_states.device.type != "npu" or request_plan is None:
            return super()._deepstack_process(hidden_states, visual_pos_masks, visual_embeds)
        if request_plan.visual_token_indices is None:
            raise RuntimeError("Qwen3 NPU deepstack requires planned visual token indices")
        if request_plan.bound_input_ids is None:
            raise RuntimeError("Qwen3 NPU deepstack request plan is not bound to input_ids")
        if hidden_states.shape[:2] != request_plan.bound_input_ids.shape:
            raise RuntimeError("Qwen3 NPU deepstack hidden shape does not match planned input_ids")
        if visual_pos_masks.shape != request_plan.bound_input_ids.shape:
            raise RuntimeError("Qwen3 NPU deepstack mask shape does not match planned input_ids")
        return _qwen3_npu_deepstack_process(
            hidden_states,
            visual_embeds.to(hidden_states.device, hidden_states.dtype),
            request_plan.visual_token_indices,
        )


class _PUMAQwen3NPUModel(Qwen3VLModel):
    def get_placeholder_mask(
        self,
        input_ids,
        inputs_embeds,
        image_features=None,
        video_features=None,
    ):
        request_plan = _QWEN3_INFERENCE_PLAN.get()
        if inputs_embeds.device.type != "npu" or request_plan is None:
            return super().get_placeholder_mask(
                input_ids,
                inputs_embeds,
                image_features=image_features,
                video_features=video_features,
            )
        if input_ids is None or input_ids is not request_plan.bound_input_ids:
            raise RuntimeError("Qwen3 NPU placeholder plan is not bound to exact input_ids")

        image_mask = input_ids.eq(self.config.image_token_id)
        video_mask = input_ids.eq(self.config.video_token_id)
        for label, features, indices in (
            ("image", image_features, request_plan.image_token_indices),
            ("video", video_features, request_plan.video_token_indices),
        ):
            if features is None:
                continue
            if indices is None:
                raise RuntimeError(f"Qwen3 NPU {label} features have no placeholder plan")
            expected_numel = indices.shape[0] * inputs_embeds.shape[-1]
            if features.numel() != expected_numel:
                raise ValueError(
                    f"{label.title()} features and {label} tokens do not match: "
                    f"tokens: {indices.shape[0]}, features {features.shape[0]}"
                )
        return (
            image_mask.unsqueeze(-1).expand_as(inputs_embeds),
            video_mask.unsqueeze(-1).expand_as(inputs_embeds),
        )

    def get_image_features(self, pixel_values, image_grid_thw=None):
        if pixel_values.device.type != "npu":
            return super().get_image_features(pixel_values, image_grid_thw)
        request_plan = _QWEN3_INFERENCE_PLAN.get()
        if request_plan is None:
            return super().get_image_features(pixel_values, image_grid_thw)
        return self._puma_npu_get_features_with_grid(
            pixel_values,
            image_grid_thw,
            request_plan.image,
        )

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        if pixel_values_videos.device.type != "npu":
            return super().get_video_features(pixel_values_videos, video_grid_thw)
        request_plan = _QWEN3_INFERENCE_PLAN.get()
        if request_plan is None:
            return super().get_video_features(pixel_values_videos, video_grid_thw)
        return self._puma_npu_get_features_with_grid(
            pixel_values_videos,
            video_grid_thw,
            request_plan.video,
        )

    def _puma_npu_get_features_with_grid(self, pixel_values, grid_thw, plan):
        if plan is None:
            raise RuntimeError("NPU visual features require a matching CPU-built request plan")
        pixel_values = pixel_values.type(self.visual.dtype)
        with _activate_qwen3_vision_plan(plan):
            embeds, deepstack_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        return torch.split(embeds, plan.merged_split_lengths, dim=0), deepstack_embeds


def install_qwen3_npu_position_adapter(model):
    """Install device-side Qwen3-VL position encoding without changing checkpoint keys."""

    try:
        visual = model.model.visual
    except AttributeError:
        return False
    if isinstance(visual, _PUMAQwen3NPUPositionVisionModel):
        return False
    if type(visual) is not Qwen3VLVisionModel:
        return False
    visual.__class__ = _PUMAQwen3NPUPositionVisionModel
    return True


def install_qwen3_npu_inference_adapter(model):
    """Install request-plan NPU overrides without adding modules or checkpoint keys."""

    outer = getattr(model, "model", None)
    visual = getattr(outer, "visual", None)
    if visual is None:
        return False

    blocks = getattr(visual, "blocks", None)
    if blocks is None:
        return False
    attentions = tuple(getattr(block, "attn", None) for block in blocks)
    language_model = getattr(outer, "language_model", None)
    text_rotary = getattr(language_model, "rotary_emb", None)
    original_language_topology = language_model is None or type(language_model) is Qwen3VLTextModel
    installed_language_topology = (
        language_model is None or type(language_model) is _PUMAQwen3NPUTextModel
    )
    original_text_topology = text_rotary is None or type(text_rotary) is Qwen3VLTextRotaryEmbedding
    installed_text_topology = (
        text_rotary is None
        or (
            type(text_rotary) is _PUMAQwen3NPUTextRotaryEmbedding
            and "_puma_mrope_source_dims" in text_rotary._buffers
        )
    )
    original_topology = (
        type(outer) is Qwen3VLModel
        and type(visual) is Qwen3VLVisionModel
        and all(type(attention) is Qwen3VLVisionAttention for attention in attentions)
        and original_language_topology
        and original_text_topology
    )
    installed_topology = (
        type(outer) is _PUMAQwen3NPUModel
        and type(visual) is _PUMAQwen3NPURequestPlanVisionModel
        and all(type(attention) is _PUMAQwen3NPUVisionAttention for attention in attentions)
        and installed_language_topology
        and installed_text_topology
    )
    if installed_topology:
        return True
    if not original_topology:
        return False

    source_dims = None
    if text_rotary is not None:
        if "_puma_mrope_source_dims" in text_rotary._buffers:
            return False
        source_dims = _build_qwen3_mrope_source_dims(
            text_rotary.inv_freq.numel(),
            text_rotary.mrope_section,
        ).to(device=text_rotary.inv_freq.device)

    class_changes = []
    buffer_registered = False
    try:
        if text_rotary is not None:
            text_rotary.register_buffer(
                "_puma_mrope_source_dims",
                source_dims,
                persistent=False,
            )
            buffer_registered = True

        replacements = [
            (outer, _PUMAQwen3NPUModel),
            (visual, _PUMAQwen3NPURequestPlanVisionModel),
            *((attention, _PUMAQwen3NPUVisionAttention) for attention in attentions),
        ]
        if language_model is not None:
            replacements.append((language_model, _PUMAQwen3NPUTextModel))
        if text_rotary is not None:
            replacements.append((text_rotary, _PUMAQwen3NPUTextRotaryEmbedding))
        for module, replacement in replacements:
            original_class = module.__class__
            module.__class__ = replacement
            class_changes.append((module, original_class))
    except BaseException:
        for module, original_class in reversed(class_changes):
            module.__class__ = original_class
        if buffer_registered:
            text_rotary._buffers.pop("_puma_mrope_source_dims", None)
            text_rotary._non_persistent_buffers_set.discard("_puma_mrope_source_dims")
        raise
    return True


def configure_qwen3_ascend_inference(model, *, linearize_patch_embed):
    """Install the required Ascend inference adapters or fail before serving."""
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        raise RuntimeError("Ascend inference requires an available torch_npu backend")
    if not install_qwen3_npu_inference_adapter(model):
        raise RuntimeError("Qwen3-VL Ascend inference adapter could not be installed")
    if linearize_patch_embed and not linearize_qwen_vision_patch_embed(model):
        raise RuntimeError("Qwen3-VL Conv3d patch embedding could not be linearized")


_PUMA_QWEN3_PLAN_KEY = "_puma_qwen3_inference_plan"
QWEN3_ASCEND_INFERENCE_PLAN_KEY = _PUMA_QWEN3_PLAN_KEY
Qwen3AscendInferencePlan = _Qwen3InferencePlan
_PUMA_QWEN3_IMAGE_CU_KEY = "_puma_qwen3_image_cu_seqlens"
_PUMA_QWEN3_VIDEO_CU_KEY = "_puma_qwen3_video_cu_seqlens"
_PUMA_QWEN3_IMAGE_TOKEN_INDICES_KEY = "_puma_qwen3_image_token_indices"
_PUMA_QWEN3_VIDEO_TOKEN_INDICES_KEY = "_puma_qwen3_video_token_indices"
_PUMA_QWEN3_VISUAL_TOKEN_INDICES_KEY = "_puma_qwen3_visual_token_indices"
_PUMA_QWEN3_IMAGE_PLAN_TENSOR_KEYS = {
    "cu_seqlens": _PUMA_QWEN3_IMAGE_CU_KEY,
    "pos_embed_indices": "_puma_qwen3_image_pos_embed_indices",
    "pos_embed_weights": "_puma_qwen3_image_pos_embed_weights",
    "rotary_pos_ids": "_puma_qwen3_image_rotary_pos_ids",
}
_PUMA_QWEN3_VIDEO_PLAN_TENSOR_KEYS = {
    "cu_seqlens": _PUMA_QWEN3_VIDEO_CU_KEY,
    "pos_embed_indices": "_puma_qwen3_video_pos_embed_indices",
    "pos_embed_weights": "_puma_qwen3_video_pos_embed_weights",
    "rotary_pos_ids": "_puma_qwen3_video_rotary_pos_ids",
}


def _stage_qwen3_plan_tensors(batch_inputs, plan, tensor_keys):
    if plan is None:
        return
    for field, key in tensor_keys.items():
        batch_inputs[key] = getattr(plan, field)


def _take_qwen3_plan_tensors(batch_inputs, plan, tensor_keys, grid_thw):
    if plan is None:
        return None
    return _bind_qwen3_vision_plan_grid(
        replace(
            plan,
            **{field: batch_inputs.pop(key) for field, key in tensor_keys.items()},
        ),
        grid_thw,
    )


def _mask_omission_is_verified(attention_mask):
    torch_version = torch.__version__.split("+", 1)[0]
    transformers_version = _TRANSFORMERS_VERSION.split("+", 1)[0]
    return (
        torch_version == "2.5.1"
        and transformers_version == "4.57.0"
        and attention_mask is not None
        and attention_mask.device.type == "cpu"
        and attention_mask.ndim == 2
        and attention_mask.shape[0] == 1
        and bool(attention_mask.eq(1).all())
    )


def _prepare_qwen3_npu_inference_inputs(batch_inputs, qwen_model, device):
    input_ids = batch_inputs.get("input_ids")
    image_grid_thw = batch_inputs.get("image_grid_thw")
    video_grid_thw = batch_inputs.get("video_grid_thw")
    attention_mask = batch_inputs.get("attention_mask")
    for name, tensor in (
        ("input_ids", input_ids),
        ("attention_mask", attention_mask),
    ):
        if tensor is not None and tensor.device.type != "cpu":
            raise ValueError(f"Qwen3 {name} control input must still be on CPU")
    for name, grid_thw in (
        ("image_grid_thw", image_grid_thw),
        ("video_grid_thw", video_grid_thw),
    ):
        if grid_thw is not None:
            _validate_qwen3_cpu_grid_thw(grid_thw, name)

    vision_config = qwen_model.config.vision_config
    try:
        num_position_embeddings = vision_config.num_position_embeddings
    except AttributeError as error:
        raise ValueError("Qwen3 vision config must define num_position_embeddings") from error
    _qwen3_num_grid_per_side(num_position_embeddings)
    try:
        visual_num_embeddings = qwen_model.visual.pos_embed.num_embeddings
    except AttributeError as error:
        raise ValueError("Qwen3 visual must expose pos_embed.num_embeddings") from error
    if visual_num_embeddings != num_position_embeddings:
        raise ValueError(
            "Qwen3 visual pos_embed.num_embeddings must match config num_position_embeddings"
        )

    position_ids = None
    target_device = torch.device(device)
    if input_ids is not None:
        position_ids, _ = qwen_model.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask=attention_mask,
        )
        batch_inputs["position_ids"] = position_ids

    omit_attention_mask = _mask_omission_is_verified(attention_mask)
    spatial_merge_size = vision_config.spatial_merge_size
    request_plan = _build_qwen3_inference_plan(
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        spatial_merge_size=spatial_merge_size,
        num_position_embeddings=num_position_embeddings,
        omit_attention_mask=omit_attention_mask,
    )
    if input_ids is not None and target_device.type == "npu":
        image_indices, video_indices, visual_indices = _build_qwen3_placeholder_indices(
            input_ids,
            image_token_id=qwen_model.config.image_token_id,
            video_token_id=qwen_model.config.video_token_id,
            image_plan=request_plan.image,
            video_plan=request_plan.video,
        )
        request_plan = replace(
            request_plan,
            image_token_indices=image_indices,
            video_token_indices=video_indices,
            visual_token_indices=visual_indices,
        )
    _stage_qwen3_plan_tensors(
        batch_inputs,
        request_plan.image,
        _PUMA_QWEN3_IMAGE_PLAN_TENSOR_KEYS,
    )
    _stage_qwen3_plan_tensors(
        batch_inputs,
        request_plan.video,
        _PUMA_QWEN3_VIDEO_PLAN_TENSOR_KEYS,
    )
    for field, key in (
        ("image_token_indices", _PUMA_QWEN3_IMAGE_TOKEN_INDICES_KEY),
        ("video_token_indices", _PUMA_QWEN3_VIDEO_TOKEN_INDICES_KEY),
        ("visual_token_indices", _PUMA_QWEN3_VISUAL_TOKEN_INDICES_KEY),
    ):
        tensor = getattr(request_plan, field)
        if tensor is not None:
            batch_inputs[key] = tensor
    if omit_attention_mask:
        batch_inputs.pop("attention_mask", None)

    batch_inputs = batch_inputs.to(target_device)
    image_plan = _take_qwen3_plan_tensors(
        batch_inputs,
        request_plan.image,
        _PUMA_QWEN3_IMAGE_PLAN_TENSOR_KEYS,
        batch_inputs.get("image_grid_thw"),
    )
    video_plan = _take_qwen3_plan_tensors(
        batch_inputs,
        request_plan.video,
        _PUMA_QWEN3_VIDEO_PLAN_TENSOR_KEYS,
        batch_inputs.get("video_grid_thw"),
    )
    batch_inputs[_PUMA_QWEN3_PLAN_KEY] = replace(
        request_plan,
        image=image_plan,
        video=video_plan,
        bound_input_ids=batch_inputs.get("input_ids"),
        image_token_indices=batch_inputs.pop(_PUMA_QWEN3_IMAGE_TOKEN_INDICES_KEY, None),
        video_token_indices=batch_inputs.pop(_PUMA_QWEN3_VIDEO_TOKEN_INDICES_KEY, None),
        visual_token_indices=batch_inputs.pop(_PUMA_QWEN3_VISUAL_TOKEN_INDICES_KEY, None),
    )
    return batch_inputs


@contextmanager
def qwen3_ascend_inference_plan_context(plan):
    """Activate an Ascend request plan for one model invocation."""
    if plan is None:
        yield None
        return
    with _activate_qwen3_inference_plan(plan) as active_plan:
        yield active_plan


def prepare_qwen3_ascend_inference_inputs(batch_inputs, qwen_model, device):
    """Build and transfer the Ascend-only Qwen3-VL request plan."""
    return _prepare_qwen3_npu_inference_inputs(batch_inputs, qwen_model, device)


__all__ = [
    "QWEN3_ASCEND_INFERENCE_PLAN_KEY",
    "Qwen3AscendInferencePlan",
    "configure_qwen3_ascend_inference",
    "prepare_qwen3_ascend_inference_inputs",
    "qwen3_ascend_inference_plan_context",
]
