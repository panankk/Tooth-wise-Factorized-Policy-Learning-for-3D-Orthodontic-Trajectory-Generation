import torch
import torch.nn as nn
from .base import BaseStrategy

# ==========================================
# 1. 消融实验组件：SingleStreamGate
# ==========================================
class SingleStreamGate(BaseStrategy):
    """
    单流门控机制 (用于 ab_04 消融实验)。
    
    目的：
        作为 Baseline，与 AdvancedDualStreamMaskHead (双流解耦) 进行对比。
    
    行为：
        不区分静态特征(类型)和动态特征(距离、状态)，
        直接将 10 维的战略特征拼接成一个向量，用一个普通的 MLP 暴力预测概率。
        以此证明解耦架构的优越性。
    """
    def __init__(self, config=None, input_dim=10, static_dim=4, embed_dim=32):
        """
        为了保证消融实验参数量的公平性，
        单流网络的隐藏层容量(64 -> 32)与双流融合后的容量对齐。
        """
        super(SingleStreamGate, self).__init__(config)
        self.input_dim = input_dim

        # 直接使用一个多层感知机 (MLP) 处理所有输入特征
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),  # 10 -> 64
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),  # 64 -> 32
            nn.ReLU(),
            nn.Linear(embed_dim, 1)               # 32 -> 1 (输出门控 Logit)
        )

    def forward(self, strat_vec):
        """
        Args:
            strat_vec: [B, N, 10] 混合的战略特征向量
        
        Returns:
            logits: [B, N] 门控 Logits
        """
        # 直接通过网络并去掉最后一个维度
        logits = self.net(strat_vec).squeeze(-1) 
        return logits