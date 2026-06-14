import torch
import torch.nn as nn
from .base import BaseHead

# ==========================================
# 1. 消融实验组件：DeterministicHead
# ==========================================
class DeterministicHead(BaseHead):
    """
    确定性回归头 (用于消融实验)。
    
    目的：
        仅预测物理量（位移/旋转）的均值，不预测不确定性（方差）。
        用于对比验证：引入不确定性估计（Uncertainty Estimation）是否真的能提升极难病例的鲁棒性。
        
    输出：
        mu: 预测的均值 (物理轨迹)
        log_var: None (无方差输出，Loss 模块将自动降级为纯 MSE)
    """
    def __init__(self, config=None, input_dim=512, output_dim=3):
        """
        Args:
            config: 配置对象 (可选)
            input_dim: 输入特征维度 (来自 Transformer)
            output_dim: 输出目标维度 (Pos=3, Rot=6)
        """
        super(DeterministicHead, self).__init__(config)
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 为了消融实验的绝对公平，这里的网络容量（隐藏层维度、层数）
        # 必须与 UncertainRegressionHead 的均值预测分支完全一致。
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            # 直接输出最终维度，不分离方差头
            nn.Linear(256, output_dim) 
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        初始化最后一层，使其输出初始接近 0。
        防止训练第一步步子迈得太大导致物理崩溃。
        """
        nn.init.uniform_(self.net[-1].weight, -0.001, 0.001)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, N, D] 输入特征 (来自 Transformer)
        
        Returns:
            mu: [B, N, Output_Dim] 预测均值
            log_var: None (直接返回 None，引擎会自动捕获并切换 Loss)
        """
        mu = self.net(x)
        
        # 兼容基类设定的 (mu, log_var) 接口
        return mu, None

    def get_output_dim(self):
        """
        返回输出目标维度。
        """
        return self.output_dim

    def get_loss_type(self):
        """
        告诉工厂或 Loss 模块，这个 Head 只能使用普通的均方误差 (MSE) Loss。
        """
        return "MSE"