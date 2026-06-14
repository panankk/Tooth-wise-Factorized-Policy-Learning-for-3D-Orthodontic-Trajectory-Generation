import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseHead

# ==========================================
# 1. 核心组件：UncertainRegressionHead
# ==========================================
class UncertainRegressionHead(BaseHead):
    """
    不确定性回归头。
    
    目的：
        不仅预测物理量（位移/旋转），还预测该预测的不确定性（方差）。
        使用异方差不确定性（Heteroscedastic Uncertainty）建模。
        
    输出：
        mu: 预测的均值 (物理量)
        log_var: 预测的对数方差 (用于 NLL Loss)
    """
    def __init__(self, config=None, input_dim=512, output_dim=3, init_log_var=-4.0):
        """
        Args:
            config: 配置对象 (可选)
            input_dim: 输入特征维度 (来自 Transformer)
            output_dim: 输出目标维度 (Pos=3, Rot=6)
            init_log_var: 对数方差的初始值。
                          -4.0 对应方差约为 0.018，表示初始时模型比较自信。
        """
        super(UncertainRegressionHead, self).__init__(config)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_log_var = init_log_var

        # 1. 共享特征提取层 (Shared Feature Extractor)
        # 使用 GELU 激活函数和 Dropout 提高泛化能力
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 2. 均值预测分支 (Mean Head)
        # 预测物理量的期望值
        self.mu_head = nn.Linear(256, output_dim)

        # 3. 方差预测分支 (Variance Head)
        # 预测对数方差 log(σ²)
        # 注意：这里输出的是 log_var，而不是直接输出 σ，为了保证数值稳定性
        self.log_var_head = nn.Linear(256, output_dim)

        # 4. 权重初始化
        self._init_weights()

    def _init_weights(self):
        """
        初始化权重，特别是方差分支的偏置，以确保训练初期的稳定性。
        """
        # 均值分支：小随机初始化，接近 0
        nn.init.uniform_(self.mu_head.weight, -0.001, 0.001)
        nn.init.constant_(self.mu_head.bias, 0)
        
        # 方差分支：初始化为固定值 (init_log_var)
        # 这样训练开始时，模型对所有预测都有一个基准的置信度
        nn.init.constant_(self.log_var_head.weight, 0)
        nn.init.constant_(self.log_var_head.bias, self.init_log_var)

    def forward(self, x):
        """
        Args:
            x: [B, N, D] 输入特征 (来自 Transformer)
        
        Returns:
            mu: [B, N, Output_Dim] 预测均值
            log_var: [B, N, Output_Dim] 预测对数方差
        """
        # 1. 提取共享特征
        feat = self.shared_net(x)
        
        # 2. 分别预测均值和方差
        mu = self.mu_head(feat)
        log_var = self.log_var_head(feat)
        
        # 3. 限制方差范围 (可选，防止数值溢出)
        # 这里我们不做 clamp，让 Loss 函数去处理极端值，或者让网络自己学习
        # 但为了数值稳定性，通常 log_var 不会让其超过一定范围
        
        return mu, log_var

    def get_output_dim(self):
        """
        返回输出目标维度。
        """
        return self.output_dim

    def get_loss_type(self):
        """
        告诉工厂，这个 Head 需要使用 NLL (Negative Log Likelihood) Loss。
        """
        return "NLL"