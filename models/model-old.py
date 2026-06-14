import torch
import torch.nn as nn
import math

from . import build_backbone, build_strategy, build_head
from .components.backbone.base import BaseBackbone
from .components.strategy.base import BaseStrategy
from .components.head.base import BaseHead

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# ==========================================
# 核心总装车间：模块化 V39 模型
# ==========================================
class DualGatedOrthoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 通用超参数
        self.d_model = config.MODEL.D_MODEL # 比如 512
        self.nhead = config.MODEL.NHEAD     # 比如 8
        self.num_layers = config.MODEL.NUM_LAYERS # 比如 6
        self.dropout = config.MODEL.DROPOUT

        # 2. 物理通道特征提取 (Backbone: PointNet)
        self.shape_encoder: BaseBackbone = build_backbone(config.MODEL.BACKBONE)
        
        # 3. 基础信息编码
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.d_model),
            nn.Linear(self.d_model, self.d_model * 2), nn.GELU(), nn.Linear(self.d_model * 2, self.d_model)
        )
        self.tooth_embedding = nn.Embedding(32, self.d_model)
        self.type_embedding = nn.Embedding(4, self.d_model)

        # 4. 物理交互中心 (Transformer 塔 - 极度重要，不能丢！)
        # 位移塔
        self.pos_embed = nn.Sequential(
            nn.Linear(6, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, self.d_model), nn.LayerNorm(self.d_model), nn.Dropout(self.dropout)
        )
        pos_enc = nn.TransformerEncoderLayer(self.d_model, self.nhead, 2048, self.dropout, batch_first=True, norm_first=True)
        self.pos_transformer = nn.TransformerEncoder(pos_enc, self.num_layers)

        # 旋转塔
        self.rot_embed = nn.Sequential(
            nn.Linear(12, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, self.d_model), nn.LayerNorm(self.d_model), nn.Dropout(self.dropout)
        )
        rot_enc = nn.TransformerEncoderLayer(self.d_model, self.nhead, 2048, self.dropout, batch_first=True, norm_first=True)
        self.rot_transformer = nn.TransformerEncoder(rot_enc, self.num_layers)

        # 5. 预测头 (Head: UncertainRegressionHead)
        self.pos_head: BaseHead = build_head(config.MODEL.HEAD.POS, output_dim=3) # 👈 明确位移是 3 维
        self.rot_head: BaseHead = build_head(config.MODEL.HEAD.ROT, output_dim=6) # 👈 明确旋转是 6 维

        # 6. 战略门控机制 (Strategy: AdvancedDualStreamMaskHead)
        self.gate_pos_head: BaseStrategy = build_strategy(config.MODEL.STRATEGY)
        self.gate_rot_head: BaseStrategy = build_strategy(config.MODEL.STRATEGY)

    def forward(self, shape_points, current_state, timestep, tooth_types, teeth_mask, strat_vec_pos, strat_vec_rot):
        """
        参数已严格对齐 train.py 里的传参顺序
        """
        batch_size, num_teeth, _ = current_state.shape
        
        # ==========================================
        # 步骤 1: 物理通道前向传播
        # ==========================================
        t_emb = self.time_mlp(timestep).unsqueeze(1)
        
        # Backbone 提特征 (注意你的 MiniPointNet 内部已经处理了展平逻辑)
        x_shape = self.shape_encoder(shape_points) 
        
        ids = torch.arange(num_teeth, device=current_state.device).unsqueeze(0).expand(batch_size, -1)
        identity = self.tooth_embedding(ids) + self.type_embedding(tooth_types)
        
        src_mask = (teeth_mask == 0) if teeth_mask is not None else None
        state_pos = current_state[..., :6] 
        state_rot = current_state[..., 6:] 

        # Transformer 物理交互 (防碰撞核心)
        token_p = self.pos_embed(state_pos) + x_shape + identity + t_emb
        feat_p = self.pos_transformer(token_p, src_key_padding_mask=src_mask) 
        
        token_r = self.rot_embed(state_rot) + x_shape + identity + t_emb
        feat_r = self.rot_transformer(token_r, src_key_padding_mask=src_mask) 
        
        # 通过预测头 (Head 输出均值和方差)
        mu_pos, log_var_pos = self.pos_head(feat_p)
        mu_pos = torch.clamp(mu_pos, -1.0, 1.0) # 🌟 恢复 V17 的防爆保护伞，防止均值漂移爆炸

        mu_rot, log_var_rot = self.rot_head(feat_r)
        mu_rot = torch.clamp(mu_rot, -5.0, 5.0) # 🌟 恢复 V17 的防爆保护伞，防止均值漂移爆炸

        # ==========================================
        # 步骤 2: 战略通道前向传播
        # ==========================================
        # 通过 Strategy 生成门控 Logits
        logits_pos = self.gate_pos_head(strat_vec_pos).squeeze(-1)
        logits_rot = self.gate_rot_head(strat_vec_rot).squeeze(-1)

        # ==========================================
        # 步骤 3: 返回 (不在模型内直接相乘，保留解耦供 Loss 评估)
        # ==========================================
        # 注意：在 V39 的架构中，我们为了在 Loss 中利用 NLL 计算方差，
        # 不能直接在模型内部用 mask 把 mu 乘 0，而是让推理引擎 (infer.py) 去乘 mask。
        # 这样训练时才能让不确定的牙齿也输出合理的 sigma！
        
        # 严格按照 train.py 接收的顺序返回：
        return mu_pos, log_var_pos, mu_rot, log_var_rot, logits_pos, logits_rot