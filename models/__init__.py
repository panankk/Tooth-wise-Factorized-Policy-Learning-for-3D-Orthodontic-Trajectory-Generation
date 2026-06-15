import torch.nn as nn


from .components import (
    MiniPointNet, IdentityBackbone,
    UncertainRegressionHead, DeterministicHead, IdentityHead,
    AdvancedDualStreamMaskHead, IdentityGate,
    SingleStreamGate
)

# ==========================================
# 1. 组件注册表 (Registry)
# ==========================================
BACKBONE_REGISTRY = {
    "MiniPointNet": MiniPointNet,
    "IdentityBackbone": IdentityBackbone,
}

STRATEGY_REGISTRY = {
    "AdvancedDualStreamMaskHead": AdvancedDualStreamMaskHead,
    "IdentityGate": IdentityGate,
    "SingleStreamGate": SingleStreamGate,
}

HEAD_REGISTRY = {
    "UncertainRegressionHead": UncertainRegressionHead,
    "DeterministicHead": DeterministicHead,
    "IdentityHead": IdentityHead,
}

# ==========================================
# 2. 构建器函数 (Builders)
# ==========================================
def build_backbone(config_backbone, input_dim=3, embed_dim=512):
    name = config_backbone.NAME
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"未知的 Backbone: {name}")
    return BACKBONE_REGISTRY[name](input_dim=input_dim, embed_dim=embed_dim)

def build_strategy(config_strategy, input_dim=10, static_dim=4, embed_dim=32):
    name = config_strategy.NAME
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知的 Strategy: {name}")
    return STRATEGY_REGISTRY[name](input_dim=input_dim, static_dim=static_dim, embed_dim=embed_dim)

def build_head(config_head, input_dim=512, output_dim=3):
    name = config_head.NAME
    if name not in HEAD_REGISTRY:
        raise ValueError(f"未知的 Head: {name}")
    return HEAD_REGISTRY[name](input_dim=input_dim, output_dim=output_dim)

# ==========================================
# 3. 顶层模型构建 (Build Model)
# ==========================================
def build_model(config):
    """
    根据 Config 构建完整的 DualGatedOrthoGPT
    """
    # 延迟导入以避免循环依赖
    from .model import DualGatedOrthoGPT 
    return DualGatedOrthoGPT(config)
