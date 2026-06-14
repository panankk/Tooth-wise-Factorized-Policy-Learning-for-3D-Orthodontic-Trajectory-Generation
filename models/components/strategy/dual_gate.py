import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseStrategy

# ==========================================
# 1. 核心组件：AdvancedDualStreamMaskHead
# ==========================================
class AdvancedDualStreamMaskHead(BaseStrategy):
    """
    双流解耦门控机制。
    
    目的：
        模拟医生决策过程，将“静态身份”与“动态状态”分离处理，
        精准控制哪些牙齿该动，哪些该静。
        
    输入分解 (假设输入维度为 10):
        - 静态特征 (Static, dim=4): 牙齿类型 (Type Embedding)
        - 动态特征 (Dynamic, dim=6): 拔牙间隙、邻牙阻力、历史移动趋势等
        
    输出：
        logits: 门控信号 (未经过 Sigmoid)
    """
    def __init__(self, config=None, input_dim=10, static_dim=4, embed_dim=32):
        """
        Args:
            config: 配置对象 (可选)
            input_dim: 战略特征总维度 (默认 10)
            static_dim: 静态特征维度 (默认 4，对应牙齿类型)
            embed_dim: 隐藏层嵌入维度 (默认 32)
        """
        super(AdvancedDualStreamMaskHead, self).__init__(config)
        
        self.input_dim = input_dim
        self.static_dim = static_dim
        self.dynamic_dim = input_dim - static_dim
        self.embed_dim = embed_dim

        # ==========================================
        # 1. 静态通道 (身份识别)
        # ==========================================
        # 处理牙齿类型信息，这部分信息在正畸过程中通常是不变的
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, embed_dim),
            nn.LayerNorm(embed_dim), # 使用 LayerNorm 提高稳定性
            nn.GELU(),               # GELU 激活函数
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        # ==========================================
        # 2. 动态通道 (状态感知)
        # ==========================================
        # 处理拔牙间隙、拥挤度等随时间变化的信息
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(self.dynamic_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        # ==========================================
        # 3. 融合决策层
        # ==========================================
        # 将静态身份和动态状态融合，做出最终决策
        # 输入维度: embed_dim (static) + embed_dim (dynamic) = 64
        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # 输出单个 Logit，表示“移动概率”
        )

    def forward(self, strat_vec):
        """
        Args:
            strat_vec: [B, N, 10] 战略特征向量
        
        Returns:
            logits: [B, N] 门控 Logits
                    - 正值表示“允许移动”
                    - 负值表示“抑制移动”
        """
        # --- 步骤 1: 特征解耦 ---
        # 切片分离静态和动态特征
        static_input = strat_vec[..., :self.static_dim]       # [B, N, 4]
        dynamic_input = strat_vec[..., self.static_dim:]      # [B, N, 6]
        
        # --- 步骤 2: 独立编码 ---
        # 分别通过各自的 MLP 提取特征
        feat_static = self.static_encoder(static_input)       # [B, N, 32]
        feat_dynamic = self.dynamic_encoder(dynamic_input)    # [B, N, 32]
        
        # --- 步骤 3: 特征融合 ---
        # 拼接静态和动态特征
        feat_fused = torch.cat([feat_static, feat_dynamic], dim=-1) # [B, N, 64]
        
        # --- 步骤 4: 决策输出 ---
        # 通过融合头输出 Logit
        # squeeze(-1) 将形状从 [B, N, 1] 变为 [B, N]
        logits = self.fusion_head(feat_fused).squeeze(-1)
        
        return logits