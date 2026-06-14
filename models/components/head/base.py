import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# ==========================================
# 1. 抽象基类：BaseHead
# ==========================================
class BaseHead(ABC, nn.Module):
    """
    所有 Prediction Head (预测头) 的抽象基类。
    
    职责：
        定义统一的输出接口，无论内部是简单的线性层还是复杂的不确定性网络，
        对外都提供一致的 (mu, log_var) 输出格式。
    """
    def __init__(self, config=None):
        super(BaseHead, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        """
        前向传播接口
        
        Args:
            x (torch.Tensor): 来自 Transformer 的特征向量。
                              形状: [B, N, D]
                              - B: Batch Size
                              - N: Num_Teeth (如 32)
                              - D: Feature Dimension (如 512)
        
        Returns:
            mu (torch.Tensor): 预测的均值 (物理量)。
                               形状: [B, N, Output_Dim]
            log_var (torch.Tensor or None): 预测的对数方差 (不确定性)。
                                            - 如果是 Uncertainty Head: 返回 [B, N, Output_Dim]
                                            - 如果是 Deterministic Head: 返回 None
        """
        raise NotImplementedError("子类必须实现 forward 方法")

    @abstractmethod
    def get_output_dim(self):
        """
        获取该 Head 输出的目标维度。
        用于验证配置文件的正确性（例如 Pos 应该是 3，Rot 应该是 6）。
        """
        raise NotImplementedError("子类必须实现 get_output_dim 方法")

    def get_loss_type(self):
        """
        获取该 Head 对应的损失函数类型。
        用于自动选择 Loss 函数 (例如 'MSE' 或 'NLL').
        """
        return "MSE" # 默认返回 MSE，UncertaintyHead 会重写为 'NLL'