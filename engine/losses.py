import torch
import torch.nn as nn

class SequentialFusionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # log_vars: 仅用于战略层的动态加权 (Focal, Pattern, Bin)
        self.log_vars = nn.Parameter(torch.zeros(3)) 
        
        self.nll_loss = nn.GaussianNLLLoss(reduction='none', full=True) 
        self.mse = nn.MSELoss(reduction='none')
        self.cosine = nn.CosineSimilarity(dim=-1)
        
        # 方差的全局与冻结限制范围
        self.global_min_log_var = -5.0  
        self.global_max_log_var = 0.5 
        self.freeze_min_log_var = -4.0
        self.freeze_max_log_var = -3.5
        
        # ==========================================
        # 🌟 核心修复：精准恢复物理梯度权重 (打破梯度饥饿)
        # ==========================================
        # 针对消融实验 (ab_05, 纯 MSE 模式): 
        # 赋予 50 倍高优权重，完美复原老版本 log_var 自动学到的强度
        self.lambda_mse_pos = 50.0 
        self.lambda_mse_rot = 0.2 
        self.lambda_cos = 5.0 
        
        # 针对满血版 (V39, NLL 模式):
        # 因为 NLL 公式内部有 1/(2*sigma^2) 起到约 50 倍的放大作用，
        # 外部乘数必须恢复成 1.0，绝不能是原来的 0.05！
        self.lambda_nll_pos = 1.0
        self.lambda_nll_rot = 1.0
        self.lambda_nll_freeze = 0.1 
        
        self.lambda_var_reg = 0.1 

        self.current_stage = "STAGE_1" 
        self.rot_freeze_mode = False

    def set_stage(self, stage_name):
        self.current_stage = stage_name
        if stage_name == "STAGE_3_FREEZE":
            self.rot_freeze_mode = True
            self.log_vars.requires_grad_(False) 
        else:
            self.rot_freeze_mode = False
            self.log_vars.requires_grad_(True)

    def forward(self, mu_pos, log_var_pos, mu_rot, log_var_rot, logits_pos, logits_rot,
                gt_pos_mu, gt_rot_mu, gt_mask_pos, gt_mask_rot, valid_mask,
                feat_prev_pos, feat_prev_rot):
        
        device = mu_pos.device
        denom = valid_mask.sum() + 1e-6
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        stats = {}
        
        # ==========================================
        # 🔥 核心重构：动态路由 (Adaptive Late-Fusion)
        # ==========================================
        is_pos_deterministic = (log_var_pos is None)
        is_rot_deterministic = (log_var_rot is None)

        # 提前计算门控概率 
        probs_pos = torch.sigmoid(logits_pos).unsqueeze(-1)
        probs_rot = torch.sigmoid(logits_rot).unsqueeze(-1)

        # 🌟 智能分流：
        # 如果是消融版 (无不确定性)，必须使用概率作为保护伞，防止静止牙齿毒害均值。
        # 如果是满血版 (有不确定性)，利用 sigma 自动降噪，让 mu 保持纯净物理意义。
        if is_pos_deterministic:
            effective_mu_pos = mu_pos * probs_pos
        else:
            effective_mu_pos = mu_pos

        if is_rot_deterministic:
            effective_mu_rot = mu_rot * probs_rot
        else:
            effective_mu_rot = mu_rot

        # --- 1. 动态处理方差 (NLL 模式) ---
        if is_pos_deterministic:
            use_nll_pos = False
            var_pos = torch.zeros_like(mu_pos) # Dummy
        else:
            if self.current_stage == "STAGE_1":
                log_var_pos = torch.full_like(log_var_pos, -4.0) 
                use_nll_pos = False
            else:
                log_var_pos = torch.clamp(log_var_pos, self.global_min_log_var, self.global_max_log_var)
                use_nll_pos = True
            var_pos = torch.exp(log_var_pos)

        if is_rot_deterministic:
            use_nll_rot = False
            var_rot = torch.zeros_like(mu_rot) # Dummy
            log_var_rot = torch.zeros_like(mu_rot) # Dummy for reg
        else:
            if self.current_stage in ["STAGE_1", "STAGE_2"]:
                log_var_rot = torch.full_like(log_var_rot, -4.0)
                use_nll_rot = False
            elif self.current_stage == "STAGE_3_FREEZE":
                log_var_rot = torch.clamp(log_var_rot, self.freeze_min_log_var, self.freeze_max_log_var)
                use_nll_rot = True
            elif self.current_stage == "STAGE_3_NORMAL":
                log_var_rot = torch.clamp(log_var_rot, self.global_min_log_var, self.global_max_log_var)
                use_nll_rot = True
            else:
                use_nll_rot = False
            var_rot = torch.exp(log_var_rot)

        # --- 2. Pos Loss ---
        raw_mse_pos = self.mse(effective_mu_pos, gt_pos_mu).sum(dim=-1)
        weights_pos = torch.where(gt_mask_pos > 0.5, 10.0, 1.0)
        l_mse_pos = (raw_mse_pos * weights_pos * valid_mask).sum() / denom
        
        if use_nll_pos:
            nll_per_tooth_pos = self.nll_loss(effective_mu_pos, gt_pos_mu, var_pos).sum(dim=-1)
            l_nll_pos = (nll_per_tooth_pos * weights_pos * valid_mask).sum() / denom
            term_pos = self.lambda_nll_pos * l_nll_pos
            stats['Raw_P_NLL'] = l_nll_pos.item(); stats['W_P_NLL'] = term_pos.item()
            stats['Raw_P_MSE'] = 0.0; stats['W_P_MSE'] = 0.0
        else:
            term_pos = self.lambda_mse_pos * l_mse_pos
            stats['Raw_P_MSE'] = l_mse_pos.item(); stats['W_P_MSE'] = term_pos.item()
            stats['Raw_P_NLL'] = 0.0; stats['W_P_NLL'] = 0.0
        total_loss = total_loss + term_pos

        # Pos Cosine (只算动的地方)
        moving_pos = (gt_mask_pos > 0.5) & (valid_mask > 0.5)
        l_cos_pos = torch.tensor(0.0, device=device)
        if moving_pos.sum() > 0:
            cos_sim = self.cosine(effective_mu_pos, gt_pos_mu) 
            l_cos_pos = ((1.0 - cos_sim) * moving_pos).sum() / (moving_pos.sum() + 1e-6)
        total_loss = total_loss + self.lambda_cos * l_cos_pos
        stats['P_Cos'] = l_cos_pos.item()

        # --- 3. Rot Loss ---
        raw_mse_rot = self.mse(effective_mu_rot, gt_rot_mu).sum(dim=-1)
        weights_rot = torch.where(gt_mask_rot > 0.5, 10.0, 1.0)
        l_mse_rot = (raw_mse_rot * weights_rot * valid_mask).sum() / denom
        
        if use_nll_rot:
            nll_per_tooth_rot = self.nll_loss(effective_mu_rot, gt_rot_mu, var_rot).sum(dim=-1)
            l_nll_rot = (nll_per_tooth_rot * weights_rot * valid_mask).sum() / denom
            lambda_nll_rot_weight = self.lambda_nll_freeze if self.rot_freeze_mode else self.lambda_nll_rot
            term_rot = lambda_nll_rot_weight * l_nll_rot
            stats['Raw_R_NLL'] = l_nll_rot.item(); stats['W_R_NLL'] = term_rot.item()
            stats['Raw_R_MSE'] = 0.0; stats['W_R_MSE'] = 0.0
            
            if not self.rot_freeze_mode:
                l_var_reg_rot = (log_var_rot ** 2 * valid_mask.unsqueeze(-1)).sum() / denom
                term_var_reg = self.lambda_var_reg * l_var_reg_rot
                total_loss = total_loss + term_var_reg
                stats['Var_Reg_Rot'] = term_var_reg.item()
            else:
                stats['Var_Reg_Rot'] = 0.0
        else:
            term_rot = self.lambda_mse_rot * l_mse_rot
            stats['Raw_R_MSE'] = l_mse_rot.item(); stats['W_R_MSE'] = term_rot.item()
            stats['Raw_R_NLL'] = 0.0; stats['W_R_NLL'] = 0.0
            stats['Var_Reg_Rot'] = 0.0
        total_loss = total_loss + term_rot

        # --- 4. Strategic Loss ---
        def calculate_focal_context_loss(logits, gt_mask, prev_active):
            probs = torch.sigmoid(logits)
            probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
            bce_loss = -(gt_mask * torch.log(probs) + (1.0 - gt_mask) * torch.log(1.0 - probs))
            pt = torch.where(gt_mask > 0.5, probs, 1.0 - probs)
            focal_weight = (1.0 - pt) ** 2.0 
            if prev_active.dim() == 3: prev_active = prev_active.squeeze(-1)
            context_weight = torch.ones_like(gt_mask)
            context_weight[(prev_active < 0.5) & (gt_mask > 0.5)] = 50.0 
            context_weight[(prev_active > 0.5) & (gt_mask < 0.5)] = 5.0 
            return (bce_loss * focal_weight * context_weight * valid_mask).sum() / denom

        logits_pos_s = 10.0 * torch.tanh(logits_pos / 10.0)
        logits_rot_s = 10.0 * torch.tanh(logits_rot / 10.0)
        
        l_focal_pos = calculate_focal_context_loss(logits_pos_s, gt_mask_pos, feat_prev_pos)
        l_focal_rot = calculate_focal_context_loss(logits_rot_s, gt_mask_rot, feat_prev_rot)
        l_focal_total = (l_focal_pos + l_focal_rot) * 0.5
        
        w_focal = torch.exp(-self.log_vars[0]) * l_focal_total + 0.5 * self.log_vars[0]
        total_loss = total_loss + w_focal
        stats['Raw_Focal'] = l_focal_total.item(); stats['W_Focal'] = w_focal.item()

        probs_pos_flat = probs_pos.squeeze(-1)
        probs_rot_flat = probs_rot.squeeze(-1)

        def calculate_soft_dice(probs, targets):
            smooth = 1e-5
            intersection = (probs * targets).sum(dim=1)
            union = probs.sum(dim=1) + targets.sum(dim=1)
            return (1.0 - (2. * intersection + smooth) / (union + smooth)).mean()

        l_pattern_total = (calculate_soft_dice(probs_pos_flat, gt_mask_pos) + 
                           calculate_soft_dice(probs_rot_flat, gt_mask_rot) + 
                           (1.0 - self.cosine(probs_pos_flat, gt_mask_pos)).mean() + 
                           (1.0 - self.cosine(probs_rot_flat, gt_mask_rot)).mean()) * 0.25
        
        w_pattern = torch.exp(-self.log_vars[1]) * l_pattern_total + 0.5 * self.log_vars[1]
        total_loss = total_loss + w_pattern
        stats['Raw_Pattern'] = l_pattern_total.item(); stats['W_Pattern'] = w_pattern.item()

        l_bin_pos = (probs_pos_flat * (1.0 - probs_pos_flat) * valid_mask).sum() / denom
        l_bin_rot = (probs_rot_flat * (1.0 - probs_rot_flat) * valid_mask).sum() / denom
        l_bin_total = (l_bin_pos + l_bin_rot) * 0.5
        w_bin = torch.exp(-self.log_vars[2]) * l_bin_total + 0.5 * self.log_vars[2]
        total_loss = total_loss + w_bin
        stats['Raw_Bin'] = l_bin_total.item(); stats['W_Bin'] = w_bin.item()

        # --- 5. Stats & Safety Check ---
        stats['Total'] = total_loss.item()
        
        with torch.no_grad():
            stats['Sigma_Focal'] = torch.sqrt(torch.exp(self.log_vars[0])).item()
            stats['Sigma_Pattern'] = torch.sqrt(torch.exp(self.log_vars[1])).item()
            stats['Sigma_Bin'] = torch.sqrt(torch.exp(self.log_vars[2])).item()
            
            stats['Avg_Sigma_Pos'] = 0.0 if is_pos_deterministic else torch.sqrt(var_pos).mean().item()
            stats['Avg_Sigma_Rot'] = 0.0 if is_rot_deterministic else torch.sqrt(var_rot).mean().item()
            
            acc_pos = (((probs_pos_flat > 0.5).float() == gt_mask_pos) * valid_mask).sum() / denom * 100.0
            acc_rot = (((probs_rot_flat > 0.5).float() == gt_mask_rot) * valid_mask).sum() / denom * 100.0
            stats['P_Acc'] = acc_pos.item()
            stats['R_Acc'] = acc_rot.item()

        return total_loss, stats