import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 核心组件：MiniPointNet
# ==========================================
class MiniPointNet(nn.Module):
    """
    轻量级 PointNet，用于提取单颗牙齿的几何形状特征。
    
    输入: [B, N, 1024, 3] (Batch, Num_Teeth, Num_Points, XYZ)
    输出: [B, N, Embed_Dim] (Batch, Num_Teeth, Feature_Dim)
    """
    def __init__(self, input_dim=3, embed_dim=512):
        super(MiniPointNet, self).__init__()
        
        # 保存配置以便调试
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # 1. 局部特征提取 (Conv1D 处理点云)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.conv4 = nn.Conv1d(256, embed_dim, 1)
        self.bn4 = nn.BatchNorm1d(embed_dim)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Args:
            x: 输入张量
               - 情况A (训练时): [B, N, 1024, 3]
               - 情况B (推理/中间态): [B*N, 1024, 3]
        
        Returns:
            features: [B, N, Embed_Dim] 几何特征向量
        """
        # --- 步骤 1: 标准化输入形状 ---
        # 如果输入是 4D [B, N, P, C]，我们需要将其展平为 [B*N, P, C] 以并行处理所有牙齿
        original_shape = x.shape
        if x.dim() == 4:
            B, N, P, C = x.shape
            x = x.view(-1, P, C)  # [B*N, 1024, 3]
        else:
            # 已经是 3D，假设 B=1 或已预处理
            pass

        # --- 步骤 2: PointNet 前向传播 ---
        # PointNet 需要 [Batch, Channels, Points] 格式
        x = x.transpose(2, 1)  # [B*N, 3, 1024]

        # Layer 1
        x = F.relu(self.bn1(self.conv1(x)))
        # Layer 2
        x = F.relu(self.bn2(self.conv2(x)))
        # Layer 3
        x = F.relu(self.bn3(self.conv3(x)))
        # Layer 4 (Embedding)
        x = self.bn4(self.conv4(x))
        
        # --- 步骤 3: 全局最大池化 (Global Max Pooling) ---
        # 将 [B*N, Embed_Dim, 1024] 压缩为 [B*N, Embed_Dim]
        # 这是 PointNet 的核心：提取整体形状特征
        x = torch.max(x, 2, keepdim=False)[0]
        
        # --- 步骤 4: 恢复原始 Batch 形状 ---
        # 将 [B*N, Embed_Dim] 恢复为 [B, N, Embed_Dim]
        if len(original_shape) == 4:
            x = x.view(original_shape[0], original_shape[1], -1)
            
        return x

# ==========================================
# 2. 消融实验组件：IdentityBackbone
# ==========================================
class IdentityBackbone(nn.Module):
    """
    用于消融实验的“空”Backbone。
    
    目的：验证 PointNet 提取的几何特征是否有效。
    行为：忽略输入的点云数据，直接返回全零张量。
          这迫使模型仅依赖“战略特征”和“时序特征”进行预测。
    """
    def __init__(self, input_dim=3, embed_dim=512):
        super(IdentityBackbone, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Args:
            x: [B, N, 1024, 3] 输入点云（将被忽略）
        Returns:
            zeros: [B, N, Embed_Dim] 全零特征
        """
        if x.dim() == 4:
            B, N = x.shape[0], x.shape[1]
        else:
            B, N = 1, x.shape[0]
            
        # 返回与输入设备一致的零张量
        return torch.zeros(B, N, self.embed_dim, device=x.device)