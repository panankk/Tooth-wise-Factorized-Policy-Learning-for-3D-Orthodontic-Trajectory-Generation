import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# ==========================================
# 1. 抽象基类：BaseBackbone
# ==========================================
class BaseBackbone(ABC, nn.Module):
    """
    所有 Backbone (特征提取器) 的抽象基类。
    
    职责：
        定义统一的接口，确保所有子类都能处理点云数据并输出几何特征。
        用于规范化输入输出形状，方便模型工厂动态组装。
    """
    def __init__(self, config=None):
        super(BaseBackbone, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        """
        前向传播接口
        
        Args:
            x (torch.Tensor): 输入的点云数据。
                              期望形状通常为 [B, N, P, C]，其中：
                              - B: Batch Size (批次大小)
                              - N: Num_Teeth (牙齿数量，如 32)
                              - P: Num_Points (每颗牙的点数，如 1024)
                              - C: Channels (坐标维度，通常为 3)
        
        Returns:
            features (torch.Tensor): 提取后的几何特征向量。
                                     期望形状为 [B, N, Embed_Dim]，其中：
                                     - Embed_Dim: 特征维度 (如 512)
        """
        raise NotImplementedError("子类必须实现 forward 方法")

    def get_output_dim(self):
        """
        获取输出特征的维度。
        用于自动构建后续的 Transformer 或 Head 层。
        """
        if hasattr(self, 'embed_dim'):
            return self.embed_dim
        raise NotImplementedError("子类应实现 get_output_dim 或定义 embed_dim 属性")