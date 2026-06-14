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
# 核心总装车间：模块化 V39 模型 (支持时空解耦注意力)
# ==========================================
class DualGatedOrthoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 通用超参数与消融开关 (Ablation Switches)
        self.d_model = config.MODEL.D_MODEL 
        self.nhead = config.MODEL.NHEAD     
        self.num_layers = config.MODEL.NUM_LAYERS 
        self.dropout = config.MODEL.DROPOUT

        # 🌟 核心升级：全局消融控制开关 (通过 YAML 获取，默认开启)
        self.window_size = config.MODEL.get('WINDOW_SIZE', 1)
        self.use_temporal_attn = config.MODEL.get('USE_WINDOW_ATTN', True)
        self.use_spatial_attn = config.MODEL.get('USE_SPATIAL_ATTN', True)
        self.use_tooth_type = config.MODEL.get('USE_TOOTH_TYPE', True)
        self.use_global_time = config.MODEL.get('USE_GLOBAL_TIME', True)

        # 2. 物理通道特征提取 (Backbone: PointNet)
        self.shape_encoder: BaseBackbone = build_backbone(config.MODEL.BACKBONE)
        
        # 3. 基础信息编码
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.d_model),
            nn.Linear(self.d_model, self.d_model * 2), nn.GELU(), nn.Linear(self.d_model * 2, self.d_model)
        )
        self.tooth_embedding = nn.Embedding(32, self.d_model)
        self.type_embedding = nn.Embedding(4, self.d_model)

        # ==========================================
        # 🌟 4. 时空双塔 Transformer (Spatio-Temporal Decoupled Attention)
        # ==========================================
        
        # [新增] A. 时序注意力塔 (Temporal Transformer) - 只看自己的过去
        if self.use_temporal_attn and self.window_size > 1:
            # 时序位置编码 (区分过去第 5 帧和第 1 帧)
            self.temp_pos_emb = nn.Parameter(torch.randn(1, self.window_size, 1, self.d_model) * 0.02)
            
            # 时序网络不需要太深，2层即可捕获惯性
            temp_enc_p = nn.TransformerEncoderLayer(self.d_model, self.nhead, 2048, self.dropout, batch_first=True, norm_first=True)
            self.temp_transformer_pos = nn.TransformerEncoder(temp_enc_p, num_layers=2)

            temp_enc_r = nn.TransformerEncoderLayer(self.d_model, self.nhead, 2048, self.dropout, batch_first=True, norm_first=True)
            self.temp_transformer_rot = nn.TransformerEncoder(temp_enc_r, num_layers=2)

        # B. 空间注意力塔 (Spatial Transformer) - 牙齿间互相看 (原版核心防撞机制)
        self.pos_embed = nn.Sequential(
            nn.Linear(6, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, self.d_model), nn.LayerNorm(self.d_model), nn.Dropout(self.dropout)
        )
        pos_enc = nn.TransformerEncoderLayer(self.d_model, self.nhead, 2048, self.dropout, batch_first=True, norm_first=True)
        self.pos_transformer = nn.TransformerEncoder(pos_enc, self.num_layers)

        self.rot_embed = nn.Sequential(
            nn.Linear(12, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, self.d_model), nn.LayerNorm(self.d_model), nn.Dropout(self.dropout)
        )
        rot_enc = nn.TransformerEncoderLayer(self.d_model, self.nhead, 2048, self.dropout, batch_first=True, norm_first=True)
        self.rot_transformer = nn.TransformerEncoder(rot_enc, self.num_layers)

        # 5. 预测头 (Head: UncertainRegressionHead)
        self.pos_head: BaseHead = build_head(config.MODEL.HEAD.POS, output_dim=3) 
        self.rot_head: BaseHead = build_head(config.MODEL.HEAD.ROT, output_dim=6) 

        # 6. 战略门控机制 (Strategy: AdvancedDualStreamMaskHead)
        self.gate_pos_head: BaseStrategy = build_strategy(config.MODEL.STRATEGY)
        self.gate_rot_head: BaseStrategy = build_strategy(config.MODEL.STRATEGY)

    def forward(self, shape_points, current_state, timestep, tooth_types, teeth_mask, strat_vec_pos, strat_vec_rot):
        # -----------------------------------------------------------
        # 🚀 维度动态解包 (The Dimension Reshape Trick)
        # train.py 传来的是 [B*W, 32, 18], 我们需要动态还原出真实的 B 和 W
        # -----------------------------------------------------------
        B = shape_points.shape[0]                # 真实 Batch Size
        num_teeth = current_state.shape[1]       # 牙齿数 (32)
        W = current_state.shape[0] // B          # 动态计算当前数据的窗口大小 W

        # 还原物理状态 [B, W, N, Dim]
        state_pos = current_state[..., :6].view(B, W, num_teeth, 6)
        state_rot = current_state[..., 6:].view(B, W, num_teeth, 12)

        # 1. 基础物理映射
        token_p_base = self.pos_embed(state_pos) # [B, W, N, D]
        token_r_base = self.rot_embed(state_rot) # [B, W, N, D]

        # 2. 获取其他全局/静态特征
        x_shape = self.shape_encoder(shape_points) # [B, N, D]
        if self.use_global_time:
            t_emb = self.time_mlp(timestep)            # [B, D]
        else:
            t_emb = torch.zeros(B, self.d_model, device=current_state.device) # 彻底致盲全局时间
        
        # 🩸 牙齿类别消融控制
        ids = torch.arange(num_teeth, device=current_state.device).unsqueeze(0).expand(B, -1)
        if self.use_tooth_type:
            identity = self.tooth_embedding(ids) + self.type_embedding(tooth_types) # [B, N, D]
        else:
            identity = self.tooth_embedding(ids) + torch.zeros_like(self.type_embedding(tooth_types))
            # 顺便抹除传递给门控的战略通道类别特征
            strat_vec_pos[..., :4] = 0.0
            strat_vec_rot[..., :4] = 0.0

        # 3. 将全局特征广播 (Broadcast) 注入到每一个历史帧中
        # x_shape/identity: [B, N, D] -> [B, 1, N, D]
        # t_emb: [B, D] -> [B, 1, 1, D]
        token_p = token_p_base + x_shape.unsqueeze(1) + identity.unsqueeze(1) + t_emb.unsqueeze(1).unsqueeze(2)
        token_r = token_r_base + x_shape.unsqueeze(1) + identity.unsqueeze(1) + t_emb.unsqueeze(1).unsqueeze(2)

        # ==========================================
        # 🌟 步骤 1.5: 时序注意力 (Temporal Attention)
        # 逻辑：让每颗牙齿沿着时间轴 W 回顾自己的历史
        # ==========================================
        if self.use_temporal_attn and W > 1 and hasattr(self, 'temp_transformer_pos'):
            # 加上时序位置编码 (切片 [:W] 防止越界)
            token_p = token_p + self.temp_pos_emb[:, :W, :, :]
            token_r = token_r + self.temp_pos_emb[:, :W, :, :]

            # 重塑为 [Batch * NumTeeth, Window, Dim]
            # 使得 Transformer 只在时间轴 W 上做 Attention
            token_p_temp = token_p.transpose(1, 2).contiguous().view(B * num_teeth, W, self.d_model)
            token_r_temp = token_r.transpose(1, 2).contiguous().view(B * num_teeth, W, self.d_model)

            feat_p_temp = self.temp_transformer_pos(token_p_temp) 
            feat_r_temp = self.temp_transformer_rot(token_r_temp) 

            # 核心提取：只提取最后一帧 (当前状态 t) 的融合特征
            curr_token_p = feat_p_temp[:, -1, :].view(B, num_teeth, self.d_model)
            curr_token_r = feat_r_temp[:, -1, :].view(B, num_teeth, self.d_model)
        else:
            # 退化/消融逻辑：直接取序列最后一帧 (等同于老代码 W=1 的逻辑)
            curr_token_p = token_p[:, -1, :, :] # [B, N, D]
            curr_token_r = token_r[:, -1, :, :] # [B, N, D]

        # ==========================================
        # 🛡️ 步骤 2: 空间注意力 (Spatial Attention - 防碰撞)
        # ==========================================
        src_mask = (teeth_mask == 0) if teeth_mask is not None else None
        
        # 🩸 空间特征消融控制
        if self.use_spatial_attn:
            feat_p = self.pos_transformer(curr_token_p, src_key_padding_mask=src_mask) 
            feat_r = self.rot_transformer(curr_token_r, src_key_padding_mask=src_mask) 
        else:
            # 消融：绕过 Transformer，互盲，直接碰撞率爆表
            feat_p = curr_token_p
            feat_r = curr_token_r
        
        # 预测头 (Head)
        mu_pos, log_var_pos = self.pos_head(feat_p)
        mu_pos = torch.clamp(mu_pos, -1.0, 1.0)
        mu_rot, log_var_rot = self.rot_head(feat_r)
        mu_rot = torch.clamp(mu_rot, -5.0, 5.0)
        # ==========================================
        # 🧠 步骤 3: 战略通道前向传播 (Strategy Gate)
        # ==========================================
        # 注意：战略通道永远只基于当前局势 (strat_vec)，与历史 W 无关
        logits_pos = self.gate_pos_head(strat_vec_pos).squeeze(-1)
        logits_rot = self.gate_rot_head(strat_vec_rot).squeeze(-1)

        # ==========================================
        # 返回结果 (严格匹配 train.py 的接收顺序)
        # ==========================================
        return mu_pos, log_var_pos, mu_rot, log_var_rot, logits_pos, logits_rot