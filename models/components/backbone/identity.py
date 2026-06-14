import torch
import torch.nn as nn
from .base import BaseBackbone

# ==========================================
# 1. 消融实验组件：IdentityBackbone
# ==========================================
class IdentityBackbone(BaseBackbone):
    """
    用于消融实验的“空”Backbone (No-Op / Placeholder)。
    
    目的：
        验证 PointNet 提取的几何特征是否有效。
        
    行为：
        忽略输入的点云数据 (shape_points)，直接返回全零张量。
        这迫使模型仅依赖“战略特征”、“时序特征”和“位置编码”进行预测。
        如果模型性能大幅下降，则证明几何特征是必要的。
    """
    def __init__(self, config=None, input_dim=3, embed_dim=512):
        """
        Args:
            config: 配置对象 (可选)
            input_dim: 输入维度 (仅用于接口兼容，实际不使用)
            embed_dim: 输出特征维度。必须与 Transformer/Head 的输入维度匹配。
        """
        super(IdentityBackbone, self).__init__(config)
        self.embed_dim = embed_dim
        self.input_dim = input_dim

    def forward(self, x):
        """
        Args:
            x: [B, N, 1024, 3] 输入点云（将被忽略）
        
        Returns:
            zeros: [B, N, Embed_Dim] 全零特征向量
        """
        # 获取 Batch (B) 和 Num_Teeth (N)
        if x.dim() == 4:
            B, N = x.shape[0], x.shape[1]
        else:
            # 兼容 3D 输入情况
            B, N = 1, x.shape[0]
            
        # 返回与输入设备 (GPU/CPU) 和精度 (Float/Half) 一致的全零张量
        return torch.zeros(B, N, self.embed_dim, device=x.device, dtype=x.dtype)

    def get_output_dim(self):
        """
        返回输出特征维度，供后续模块（如 Head）查询。
        """
        return self.embed_dim