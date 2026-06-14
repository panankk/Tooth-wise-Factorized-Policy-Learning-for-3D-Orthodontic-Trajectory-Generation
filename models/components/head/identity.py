import torch
import torch.nn as nn
from .base import BaseHead

# ==========================================
# 1. 消融实验组件：IdentityHead
# ==========================================
class IdentityHead(BaseHead):
    """
    用于消融实验的“空”Head (No-Op / Placeholder)。
    
    目的：
        验证预测头（无论是确定性还是不确定性）是否有效。
        
    行为：
        忽略输入的特征数据，直接返回全零张量。
        这迫使模型仅依赖“Backbone”和“Transformer”的特征进行预测（实际上无法预测）。
        如果模型性能大幅下降，则证明预测头是必要的。
    """
    def __init__(self, config=None, input_dim=512, output_dim=3):
        """
        Args:
            config: 配置对象 (可选)
            input_dim: 输入维度 (仅用于接口兼容，实际不使用)
            output_dim: 输出目标维度。必须与任务匹配（Pos=3, Rot=6）。
        """
        super(IdentityHead, self).__init__(config)
        self.output_dim = output_dim
        self.input_dim = input_dim

    def forward(self, x):
        """
        Args:
            x: [B, N, D] 输入特征（将被忽略）
        
        Returns:
            mu: [B, N, Output_Dim] 全零预测值
            log_var: None (确定性输出)
        """
        # 获取 Batch (B) 和 Num_Teeth (N)
        if x.dim() == 3:
            B, N = x.shape[0], x.shape[1]
        else:
            # 兼容其他维度情况
            B, N = 1, x.shape[0]
            
        # 返回与输入设备 (GPU/CPU) 和精度 (Float/Half) 一致的全零张量
        mu = torch.zeros(B, N, self.output_dim, device=x.device, dtype=x.dtype)
        
        # 返回 None 表示没有不确定性（确定性回归）
        return mu, None

    def get_output_dim(self):
        """
        返回输出目标维度，供 Loss 函数查询。
        """
        return self.output_dim

    def get_loss_type(self):
        """
        返回损失函数类型。
        IdentityHead 是确定性的，所以使用 MSE。
        """
        return "MSE"