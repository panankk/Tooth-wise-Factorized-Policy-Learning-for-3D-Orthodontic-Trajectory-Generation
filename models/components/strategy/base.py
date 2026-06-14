import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# ==========================================
# 1. 抽象基类：BaseStrategy
# ==========================================
class BaseStrategy(ABC, nn.Module):
    """
    所有 Strategy (战略门控机制) 的抽象基类。
    
    职责：
        定义统一的接口，确保所有子类都能处理战略特征向量并输出门控信号。
        用于规范化输入输出形状，方便模型工厂动态组装。
    """
    def __init__(self, config=None):
        super(BaseStrategy, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, strat_vec):
        """
        前向传播接口
        
        Args:
            strat_vec (torch.Tensor): 输入的战略特征向量。
                                      期望形状: [B, N, S]
                                      - B: Batch Size
                                      - N: Num_Teeth (如 32)
                                      - S: Strategy_Dim (如 10，包含类型、间隙、阻力等)
        
        Returns:
            logits (torch.Tensor): 门控 logits (未经过 Sigmoid)。
                                   期望形状: [B, N] 或 [B, N, 1]
                                   - 正值表示“允许移动”
                                   - 负值表示“抑制移动”
        """
        raise NotImplementedError("子类必须实现 forward 方法")