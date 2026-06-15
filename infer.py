import torch
import torch.nn.functional as F
import numpy as np
import os
import csv
import argparse
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


from models import build_model


class DummyConfig:
    class DATA:
        PROCESSED_ROOT = "/path/to/test_data_processed"
    
    class MODEL:
        class BACKBONE:
            NAME = "MiniPointNet" 
        class STRATEGY:
            NAME = "AdvancedDualStreamMaskHead"
        class HEAD:
            class POS:
                NAME = "UncertainRegressionHead"
            class ROT:
                NAME = "UncertainRegressionHead"
        
        D_MODEL = 512
        NHEAD = 8
        NUM_LAYERS = 6
        DROPOUT = 0.1
        NUM_TEETH = 32

    # 推理专属配置
    CKPT_PATH = "/path/to/checkpoints"  
    SAVE_ROOT = "/path/to/inference_results" 
    DEVICE_ID = "0"
    
    MAX_STEPS = 150
    STOP_THRES_POS = 0.2   # 0.2mm
    STOP_THRES_DEG = 3.0   # 3.0度
    MASK_THRESHOLD = 0.5 
    
  
    UNCERTAINTY_SCALE_FACTOR = 0.0 

# ==========================================
# 牙位定义 & 类型映射
# ==========================================
FDI_MAP = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28, 
           48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

TOOTH_TYPE_MAP = {
    11:0, 12:0, 21:0, 22:0, 31:0, 32:0, 41:0, 42:0, 
    13:1, 23:1, 33:1, 43:1,                         
    14:2, 15:2, 24:2, 25:2, 34:2, 35:2, 44:2, 45:2, 
    16:3, 17:3, 18:3, 26:3, 27:3, 28:3,             
    36:3, 37:3, 38:3, 46:3, 47:3, 48:3
}

# ==========================================
#  辅助函数
# ==========================================
def normalize_ortho6d(ortho6d):
    x_raw = ortho6d[..., 0:3]; y_raw = ortho6d[..., 3:6]
    x = F.normalize(x_raw, dim=-1, eps=1e-8)
    y = F.normalize(y_raw - (x * y_raw).sum(dim=-1, keepdim=True) * x, dim=-1, eps=1e-8)
    return torch.cat([x, y], dim=-1)

def compute_rotation_matrix_from_ortho6d(ortho6d):
    ortho6d = normalize_ortho6d(ortho6d)
    x = ortho6d[..., 0:3]; y = ortho6d[..., 3:6]
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-1)

def compute_metrics(curr_pose, goal_pose, mask):
    pos_err = torch.norm(curr_pose[..., :3] - goal_pose[..., :3], dim=-1)
    mp = (pos_err * mask).max().item() 
    
    R_curr = compute_rotation_matrix_from_ortho6d(curr_pose[..., 3:9])
    R_goal = compute_rotation_matrix_from_ortho6d(goal_pose[..., 3:9])
    R_diff = torch.matmul(R_curr.transpose(-1, -2), R_goal)
    trace = torch.clamp(R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2], -1.0, 3.0)
    angle_deg = torch.rad2deg(torch.acos((trace - 1) / 2))
    
    mr = (angle_deg * mask).max().item() 
    return mp, mr

def save_step_to_txt(pose, step, save_dir, mask):
    pose_np = pose.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    rot_mat = compute_rotation_matrix_from_ortho6d(pose[..., 3:9]).detach().cpu().numpy()
    lines = []
    for i in range(32):
        if mask_np[i] < 0.5: continue
        q = R.from_matrix(rot_mat[i]).as_quat()
        lines.append(f"{FDI_MAP[i]} {pose_np[i,0]:.6f} {pose_np[i,1]:.6f} {pose_np[i,2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    with open(os.path.join(save_dir, f"step_{step:03d}.txt"), 'w') as f: f.write("\n".join(lines))

def make_inference_state(curr_p, goal_p, teeth_mask):
    is_batch = (curr_p.dim() == 3)
    cp = curr_p if is_batch else curr_p.unsqueeze(0)
    gp = goal_p if is_batch else goal_p.unsqueeze(0)
    tm = teeth_mask if is_batch else teeth_mask.unsqueeze(0)

    curr_pos = cp[..., :3]
    mask_expand = tm.unsqueeze(-1)
    sum_pos = (curr_pos * mask_expand).sum(dim=1, keepdim=True)
    count = mask_expand.sum(dim=1, keepdim=True) + 1e-6
    centroid = sum_pos / count
    local_pos = curr_pos - centroid 
    diff_pos = gp[..., :3] - cp[..., :3]
    state_pos = torch.cat([local_pos, diff_pos], dim=-1)

    curr_rot = cp[..., 3:9]
    goal_rot = gp[..., 3:9]
    diff_rot = goal_rot - curr_rot 
    state_rot = torch.cat([curr_rot, diff_rot], dim=-1)
    
    state = torch.cat([state_pos, state_rot], dim=-1)
    return state.squeeze(0) if not is_batch else state

# ==========================================
#  核心推理逻辑
# ==========================================
def run_case(case_id, model, device, cfg):
    case_dir = os.path.join(cfg.DATA.PROCESSED_ROOT, case_id)
    save_dir = os.path.join(cfg.SAVE_ROOT, case_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 加载数据
    try:
        poses_9d = torch.load(os.path.join(case_dir, 'poses_9d.pt'), map_location=device, weights_only=True)
        shape_emb = torch.load(os.path.join(case_dir, 'shape_feature.pt'), map_location=device, weights_only=True)
        meta = torch.load(os.path.join(case_dir, 'meta.pt'), map_location=device, weights_only=True)
    except Exception as e:
        print(f"Error loading {case_id}: {e}")
        return {"status": "Fail_Load", "max_p": 99.9, "max_r": 99.9, "steps": 0}

    mask = meta['mask'].to(device).unsqueeze(0)
    
    curr_pose = poses_9d[0].unsqueeze(0).clone()
    goal_pose = poses_9d[-1].unsqueeze(0).clone()
    shape = shape_emb.unsqueeze(0)

    # 预处理: 剔除 Ghost
    init_pos = curr_pose[0, :, :3]
    dist_from_origin = torch.norm(init_pos, dim=-1)
    is_ghost = (mask[0] > 0.5) & (dist_from_origin < 1e-4)
    if is_ghost.any(): mask[0, is_ghost] = 0.0
    
    # 构建 Tooth Types
    type_ids_list = [TOOTH_TYPE_MAP.get(fdi, 3) for fdi in FDI_MAP]
    tooth_types = torch.tensor(type_ids_list, dtype=torch.long, device=device).unsqueeze(0)
    
    # 构建 One-Hot Type Feature
    feat_my_type = torch.zeros(1, 32, 4, device=device)
    feat_my_type.scatter_(2, tooth_types.unsqueeze(2), 1.0)

    metrics_log = []
    save_step_to_txt(curr_pose.squeeze(0), 0, save_dir, mask.squeeze(0))
    status = "Timeout"
    
    # 初始化双轨制惯性
    prev_active_pos = torch.zeros(1, 32, 1, device=device)
    prev_active_rot = torch.zeros(1, 32, 1, device=device)
    
    type_masks = []
    for i in range(4):
        type_masks.append((tooth_types == i).float())

    # ================= 循环生成 =================
    for step in range(1, cfg.MAX_STEPS + 1):
        
        state = make_inference_state(curr_pose, goal_pose, mask).to(device)
        time_tensor = torch.tensor([float(step)], device=device)

        diff_pos_vec = goal_pose[..., :3] - curr_pose[..., :3]
        res_pos = torch.norm(diff_pos_vec, dim=-1, keepdim=True)
        
        diff_rot_vec = goal_pose[..., 3:9] - curr_pose[..., 3:9]
        res_rot = torch.norm(diff_rot_vec, dim=-1, keepdim=True)
        
        is_finished_pos = (res_pos < 0.2).float() 
        is_finished_rot = (res_rot < 0.05).float()
        
        group_rates_pos_list = []
        group_rates_rot_list = []
        
        for i in range(4):
            g_mask = type_masks[i].unsqueeze(-1)
            total = g_mask.sum() + 1e-6
            group_rates_pos_list.append((is_finished_pos * g_mask).sum() / total)
            group_rates_rot_list.append((is_finished_rot * g_mask).sum() / total)
            
        feat_group_pos = torch.stack(group_rates_pos_list).view(1, 1, 4).expand(1, 32, 4)
        feat_group_rot = torch.stack(group_rates_rot_list).view(1, 1, 4).expand(1, 32, 4)
        
        strat_vec_pos = torch.cat([feat_my_type, feat_group_pos, res_pos, prev_active_pos], dim=-1)
        strat_vec_rot = torch.cat([feat_my_type, feat_group_rot, res_rot, prev_active_rot], dim=-1)

        with torch.no_grad():
            pred_mu_pos, pred_log_var_pos, pred_mu_rot, pred_log_var_rot, logits_pos, logits_rot = model(
                shape, state, time_tensor, tooth_types, mask,
                strat_vec_pos=strat_vec_pos, 
                strat_vec_rot=strat_vec_rot
            )
            
        prob_pos = torch.sigmoid(logits_pos).unsqueeze(-1)
        prob_rot = torch.sigmoid(logits_rot).unsqueeze(-1)
        
        mask_move_pos = (prob_pos > cfg.MASK_THRESHOLD).float()
        mask_move_rot = (prob_rot > cfg.MASK_THRESHOLD).float()
        
        prev_active_pos = mask_move_pos
        prev_active_rot = mask_move_rot
        
       
        effective_pred_mu_pos = pred_mu_pos * prob_pos
        effective_pred_mu_rot = pred_mu_rot * prob_rot
        
      
        base_pos_delta = effective_pred_mu_pos * mask_move_pos
        if cfg.UNCERTAINTY_SCALE_FACTOR > 0 and pred_log_var_pos is not None:
            sigma_pos = torch.exp(0.5 * pred_log_var_pos)
            scale_pos = 1.0 / (1.0 + sigma_pos * cfg.UNCERTAINTY_SCALE_FACTOR)
            final_pos_delta = base_pos_delta * scale_pos
        else:
            final_pos_delta = base_pos_delta

       
        base_rot_delta = (effective_pred_mu_rot / 100.0) * mask_move_rot
        if cfg.UNCERTAINTY_SCALE_FACTOR > 0 and pred_log_var_rot is not None:
            sigma_rot = torch.exp(0.5 * pred_log_var_rot)
            scale_rot = 1.0 / (1.0 + sigma_rot * cfg.UNCERTAINTY_SCALE_FACTOR)
            final_rot_delta = base_rot_delta * scale_rot
        else:
            final_rot_delta = base_rot_delta

      
        is_pos_converged = res_pos < 0.05 
        final_pos_delta = final_pos_delta.masked_fill(is_pos_converged, 0.0)
        
        is_rot_converged = res_rot < 0.02 
        final_rot_delta = final_rot_delta.masked_fill(is_rot_converged, 0.0)
        
        step_dist_pos = torch.norm(final_pos_delta, dim=-1, keepdim=True)
        final_pos_delta = final_pos_delta.masked_fill(step_dist_pos < 0.02, 0.0)

        step_dist_rot = torch.norm(final_rot_delta, dim=-1, keepdim=True)
        final_rot_delta = final_rot_delta.masked_fill(step_dist_rot < 0.005, 0.0)

        
        curr_pose[..., :3] += final_pos_delta
        curr_pose[..., 3:9] += final_rot_delta
        curr_pose[..., 3:9] = normalize_ortho6d(curr_pose[..., 3:9])
        
       
        mp, mr = compute_metrics(curr_pose, goal_pose, mask)
        metrics_log.append({"step": step, "max_pos": mp, "max_rot": mr})
        save_step_to_txt(curr_pose.squeeze(0), step, save_dir, mask.squeeze(0))
        
        if mp < cfg.STOP_THRES_POS and mr < cfg.STOP_THRES_DEG:
            status = " Success"
            break
            
    with open(os.path.join(save_dir, "metrics.csv"), 'w') as f:
        w = csv.DictWriter(f, fieldnames=["step", "max_pos", "max_rot"])
        w.writeheader(); w.writerows(metrics_log)
        
    return {"status": status, "max_p": mp, "max_r": mr, "steps": step}

def main():
    cfg = DummyConfig()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.DEVICE_ID
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f" Modular Inference Engine Started")
    print(f" Checkpoint: {cfg.CKPT_PATH}")
    
   
    model = build_model(cfg).to(device)
    
    if os.path.exists(cfg.CKPT_PATH):
        print(f"Loading Model Weights...")
        ckpt = torch.load(cfg.CKPT_PATH, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt: 
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else: 
            model.load_state_dict(ckpt, strict=False)
        model.eval()
        print("Model Loaded Successfully.")
    else: 
        print(f" Checkpoint not found: {cfg.CKPT_PATH}")
        return

    cases = sorted([c for c in os.listdir(cfg.DATA.PROCESSED_ROOT) if os.path.isdir(os.path.join(cfg.DATA.PROCESSED_ROOT, c))])
    pbar = tqdm(cases)
    
    suc = 0; tot_p = 0; tot_r = 0
    fail_reasons = {"Pos": 0, "Rot": 0, "Both": 0}
    
    for c in pbar:
        res = run_case(c, model, device, cfg)
        tot_p += res['max_p']; tot_r += res['max_r']
        
        if "S" in res['status']:
            suc += 1
        else:
            is_pos_fail = res['max_p'] >= cfg.STOP_THRES_POS
            is_rot_fail = res['max_r'] >= cfg.STOP_THRES_DEG
            if is_pos_fail: fail_reasons["Pos"] += 1
            if is_rot_fail: fail_reasons["Rot"] += 1
            if is_pos_fail and is_rot_fail: fail_reasons["Both"] += 1 
            
        pbar.set_postfix({"S": res['status'], "Steps": res['steps'], "P": f"{res['max_p']:.2f}", "R": f"{res['max_r']:.1f}"})
        
    total_cases = len(cases)
    if total_cases == 0:
        print("No cases found.")
        return

    total_fails = total_cases - suc
    
    print("\n" + "="*60)
    print(f"📊 Final Modular Inference Report")
    print("="*60)
    print(f" Success Rate: {suc/total_cases*100:.1f}% ({suc}/{total_cases})")
    print(f" Total Failures: {total_fails}")
    print("-" * 30)
    if total_fails > 0:
        print(f"   Due to Position (> {cfg.STOP_THRES_POS}mm): {fail_reasons['Pos']} cases")
        print(f"   Due to Rotation (> {cfg.STOP_THRES_DEG}°):  {fail_reasons['Rot']} cases")
        print(f"   Both Failed: {fail_reasons['Both']} cases")
    print("-" * 30)
    print(f" Avg Final Error -> Pos: {tot_p/total_cases:.4f}mm | Rot: {tot_r/total_cases:.4f}°")
    print("="*60)

if __name__ == "__main__": 
    main()
