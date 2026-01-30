# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from synerfuse.core.fusions.fused_bias_dropout import get_bias_dropout_add
from synerfuse.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from synerfuse.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from synerfuse.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from synerfuse.core.transformer.enums import AttnMaskType
from synerfuse.core.transformer.identity_op import IdentityOp
from synerfuse.core.transformer.mlp import MLP, MLPSubmodules
from synerfuse.core.transformer.spec_utils import ModuleSpec
from synerfuse.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_vit_layer_with_transformer_engine_spec() -> ModuleSpec:
    mlp = _get_mlp_module_spec(use_te=True)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={
                    "attn_mask_type": AttnMaskType.causal
                },  # TODO: This should be no_mask when CI is upgraded
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(use_te: bool = True,) -> ModuleSpec:
    # Dense MLP w/ or w/o TE modules.
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
            linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
        ),
    )
