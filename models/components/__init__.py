"""
组件聚合模块 (Components Aggregator)

职责：
    将分散在 backbone, head, strategy 等子目录下的具体实现
    统一暴露在这个命名空间下，方便顶层工厂 (models/__init__.py) 进行导入。
"""

# ==========================================
# 1. 导入 Backbone 组件
# ==========================================
# 核心特征提取器
from .backbone.pointnet import MiniPointNet
# 消融实验：空 Backbone
from .backbone.identity import IdentityBackbone

# ==========================================
# 2. 导入 Head 组件
# ==========================================
# 核心预测头：不确定性回归
from .head.uncertain import UncertainRegressionHead
# 消融实验：确定性预测头
from .head.deterministic import DeterministicHead
# 消融实验：空 Head
from .head.identity import IdentityHead

# ==========================================
# 3. 导入 Strategy 组件
# ==========================================
# 核心门控机制：双流解耦
from .strategy.dual_gate import AdvancedDualStreamMaskHead
# 消融实验：直通门控
from .strategy.identity import IdentityGate
# 假设你的 strategy 导出在这里
from .strategy.single_gate import SingleStreamGate

# 记得把它加到 __all__ 列表里（如果有的话）

# ==========================================
# 4. 导出列表 (Export List)
# ==========================================
# 定义 __all__ 可以防止使用 `from components import *` 时污染命名空间
# 同时也明确了哪些是公开接口
__all__ = [
    # Backbone
    'MiniPointNet',
    'IdentityBackbone',
    
    # Head
    'UncertainRegressionHead',
    'DeterministicHead',
    'IdentityHead',
    
    # Strategy
    'AdvancedDualStreamMaskHead',
    'IdentityGate',
    'SingleStreamGate',
]